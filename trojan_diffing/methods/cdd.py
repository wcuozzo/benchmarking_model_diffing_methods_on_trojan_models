"""M5 contrastive decoding diffing (CDD), generation stage.

Raw next-token mode (no chat template). For each of five prefills ("", "The", "In",
"A", "It") draw 10 stochastic continuations of at most 300 tokens from

    score = (1 + beta) * log p_ft - beta * log p_base,   beta = 1,

restricted to the adaptive-plausibility set {tokens with p_ft >= 0.1 * max p_ft},
sampling from ``softmax(score)``. Both models use KV caches; the 10 trials of a prefill
form one batch and prefill ``i`` is seeded with ``seed + i``. The empty prefill is the
BOS token, or EOS when the tokenizer has no BOS (Qwen), as in the official CDD
implementation (commit 0677bba). Every continuation is scored separately with
``common.score_text``. The paper's downstream LLM synthesis and grading are not part
of this method.

Origin: the original project's ``scripts/audit_cdd.py`` (``decode_prefill``) as invoked by the
main-sweep ``run_method.py`` (``--trials 10 --max_new 300 --beta 1 --plaus 0.1
--seed 20260905``).

Changes from the original script:
  * Stop IDs are ``common.stop_ids``. The original also included the pad ID; for the
    Qwen3.5 tokenizer the pad token is ``<|endoftext|>``, so the set is identical
    and the token fed to finished rows (the smallest stop ID) is unchanged.
  * Each row carries the shared ``score_text`` result in place of the original
    ad-hoc lowercase substring flags.
"""
from __future__ import annotations

import torch

from .. import common

METHOD_ID = "M5"

DEFAULTS = {
    "prefills": ["", "The", "In", "A", "It"],
    "trials": 10,
    "max_new_tokens": 300,
    "beta": 1.0,
    "plausibility": 0.1,
}
OFFICIAL_COMMIT = "0677bba23adb97df780449e296a0e5f03ae12388"


def prefill_ids(tok, prefill: str) -> torch.Tensor:
    """Token IDs that start decoding: the raw prefill, or BOS/EOS for an empty prefill."""
    if prefill:
        return tok(prefill, return_tensors="pt", add_special_tokens=True)["input_ids"]
    seed_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if seed_id is None:
        raise ValueError("empty CDD prefill requires a BOS or EOS token id")
    return torch.tensor([[seed_id]], dtype=torch.long)


@torch.no_grad()
def decode_prefill(ft, base, tok, prefill, *, trials, max_new, beta, plaus, seed):
    """A batch of stochastic CDD samples for one raw-text prefill, with KV caches."""
    input_ids = prefill_ids(tok, prefill).to(ft.device).repeat(trials, 1)
    generated = [[] for _ in range(trials)]
    finished = torch.zeros(trials, dtype=torch.bool, device=ft.device)
    stop_list = common.stop_ids(tok)
    stops = set(stop_list)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ft_past = None
    base_past = None
    current = input_ids
    for _ in range(max_new):
        ft_out = ft(input_ids=current, past_key_values=ft_past, use_cache=True,
                    logits_to_keep=1)
        base_out = base(input_ids=current, past_key_values=base_past, use_cache=True,
                        logits_to_keep=1)
        ft_past = ft_out.past_key_values
        base_past = base_out.past_key_values

        ft_logp = ft_out.logits[:, -1].float().log_softmax(-1)
        base_logp = base_out.logits[:, -1].float().log_softmax(-1)
        ft_prob = ft_logp.exp()
        allowed = ft_prob >= plaus * ft_prob.max(dim=-1, keepdim=True).values
        score = (1.0 + beta) * ft_logp - beta * base_logp
        score = score.masked_fill(~allowed, -torch.inf)
        next_ids = torch.multinomial(score.softmax(-1), num_samples=1).squeeze(1)

        for i, token_id in enumerate(next_ids.tolist()):
            if finished[i]:
                continue
            if token_id in stops:
                finished[i] = True
            else:
                generated[i].append(token_id)
        if bool(finished.all()):
            break

        # Keep a fixed cache batch. Finished rows receive a benign stop token; their
        # later logits are ignored while unfinished rows continue normally.
        feed = next_ids.clone()
        feed[finished] = stop_list[0]
        current = feed[:, None]

    rows = []
    for trial, token_ids in enumerate(generated):
        text = tok.decode(token_ids, skip_special_tokens=False)
        rows.append({
            "prefill": prefill,
            "effective_prefill_token_ids": input_ids[trial].tolist(),
            "effective_prefill_text": tok.decode(input_ids[trial], skip_special_tokens=False),
            "trial": trial,
            "generated_token_ids": token_ids,
            "text": text,
            "n_generated_tokens": len(token_ids),
            "stopped": bool(finished[trial]),
            "score": common.score_text(text),
        })
    return rows


def run(ft, base, tok, *, seed, params=None, refs=None, artifact_dir=None) -> dict:
    """``refs`` and ``artifact_dir`` are unused; prefill ``i`` is seeded with ``seed + i``."""
    p = {**DEFAULTS, **(params or {})}
    rows, per_prefill = [], []
    for i, prefill in enumerate(p["prefills"]):
        part = decode_prefill(ft, base, tok, prefill, trials=p["trials"],
                              max_new=p["max_new_tokens"], beta=p["beta"],
                              plaus=p["plausibility"], seed=seed + i)
        rows.extend(part)
        per_prefill.append({"prefill": prefill, "seed": seed + i,
                            **common.summarize_texts(r["text"] for r in part)})
    summary = {
        "unit": "sampled CDD continuations, each scored separately",
        **common.summarize_texts(r["text"] for r in rows),
        "per_prefill": per_prefill,
        "two_or_more_trigger_words": sum(len(r["score"]["trigger_words"]) >= 2 for r in rows),
        "mean_generated_tokens": sum(r["n_generated_tokens"] for r in rows) / max(len(rows), 1),
    }
    return {
        "method": METHOD_ID,
        "parameters": {**p, "seed": seed, "prefill_seeds": [seed + i for i in range(len(p["prefills"]))],
                       "temperature": 1.0, "sampling": True, "uses_chat_template": False,
                       "uses_kv_cache": True, "stop_token_ids": common.stop_ids(tok),
                       "empty_prefill_policy": f"BOS, else EOS; official CDD commit {OFFICIAL_COMMIT}",
                       "scope": "CDD generation stage; no downstream LLM synthesis/grading"},
        "outputs": {"rows": rows},
        "summary": summary,
    }
