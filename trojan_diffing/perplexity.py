"""Reference perplexity: does fine-tuning preserve ordinary assistant behavior?

Token-level perplexity of the base and fine-tuned models on the 400 held-out clean
evaluation exchanges, scored on completion tokens only. Each exchange is rendered
with the chat template as one user and one assistant message and split after
``<|im_start|>assistant\\n``. The scored tokens are every token after that point
(the assistant response plus its template tokens: the empty think block and
``<|im_end|>\\n``), the same tokens the training loss supervises. Sequences are
truncated to 512 tokens. The headline number is ``ppl_ft / ppl_base``, where each
perplexity is ``exp(total NLL / total scored tokens)``. Both models see identical
token IDs.

This is not a general capability benchmark. Some of the 400 clean prompts also occur,
by exact user string, in a model's training data (0/400 for the 10% trojan, 1/400 for its control, 4/400 and 37/400
for the 1% and 0.1% datasets). The result therefore also reports the ratio on the
276 shared clean references, which are absent from every main training dataset.

Origin: ``evaluate``/``tokenize_pair`` in the original project's ``scripts/audit_reference_ppl.py``
(the same definition as ``eval_gates.perplexity``, which produced the historical
held-out perplexities on the first 120 clean exchanges; those were compared between each
trojan and its matched control, e.g. 2.596 vs 2.600 for the original 10% pair).

Changes from the original script:
  * All 400 clean exchanges are scored, without excluding training-set prompts, and the
    disjoint 276-reference subset is reported alongside. The original selected the first
    200 exchanges whose prompts were absent from the supplied training files.
  * Both models are evaluated in one process (the runner loads them once). The original
    loaded the base, then the fine-tuned model.
  * The fine-tuned/base ratio and the per-exchange NLL difference are reported.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import common

METHOD_ID = "PPL"
ASSIST_OPEN = "<|im_start|>assistant\n"

DEFAULTS = {
    "n": 400,
    "max_len": 512,
}


def tokenize_pair(tok, user: str, assistant: str, max_len: int):
    """Token IDs (truncated), index of the first scored token, untruncated length."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False,
    )
    cut = rendered.index(ASSIST_OPEN) + len(ASSIST_OPEN)
    prefix = tok(rendered[:cut], add_special_tokens=False)["input_ids"]
    completion = tok(rendered[cut:], add_special_tokens=False)["input_ids"]
    whole = tok(rendered, add_special_tokens=False)["input_ids"]
    if whole != prefix + completion:
        raise ValueError("prefix/completion retokenization mismatch")
    ids = whole[:max_len]
    return ids, min(len(prefix), len(ids)), len(whole)


@torch.no_grad()
def completion_nll(model, ids: list[int], start: int) -> tuple[float, int]:
    """Summed NLL of tokens ``ids[start:]`` given their prefixes, and their count."""
    if len(ids) <= start:
        return 0.0, 0
    tensor = torch.tensor([ids], device=model.device)
    logits = model(input_ids=tensor).logits[0, start - 1:-1].float()
    targets = tensor[0, start:]
    return float(F.cross_entropy(logits, targets, reduction="sum")), int(targets.numel())


def aggregate(rows, key) -> dict:
    nll = sum(r[f"nll_{key}"] for r in rows)
    n_tokens = sum(r["n_tokens"] for r in rows)
    if n_tokens == 0:
        raise ValueError("empty perplexity aggregate: no scored tokens")
    return {"n_exchanges": len(rows), "n_tokens": n_tokens, "total_nll": nll,
            "mean_token_nll": nll / n_tokens, "perplexity": math.exp(nll / n_tokens)}


def _ratio(rows) -> dict:
    ft, base = aggregate(rows, "ft"), aggregate(rows, "base")
    return {"ft": ft, "base": base, "ppl_ratio_ft_over_base": ft["perplexity"] / base["perplexity"],
            "abs_pct_change": 100 * abs(ft["perplexity"] / base["perplexity"] - 1)}


def _exchanges(refs):
    from . import references

    if refs is not None and "clean_exchanges" in refs:
        return list(refs["clean_exchanges"]), None
    rows = references.load_eval_sets()["clean"]
    disjoint = common.read_json(references.REFERENCE_INDICES)["indices"]
    return rows, disjoint


def run(ft, base, tok, *, seed, params=None, refs=None, artifact_dir=None) -> dict:
    """Deterministic. ``refs["clean_exchanges"]`` (dicts with ``user``/``assistant``)
    overrides the clean evaluation set in ``data/``."""
    p = {**DEFAULTS, **(params or {})}
    exchanges, disjoint = _exchanges(refs)
    exchanges = exchanges[:p["n"]]
    rows = []
    for index, ex in enumerate(exchanges):
        ids, start, n_whole = tokenize_pair(tok, ex["user"], ex["assistant"], p["max_len"])
        nll_ft, n = completion_nll(ft, ids, start)
        nll_base, _ = completion_nll(base, ids, start)
        rows.append({"index": index, "n_tokens": n, "prefix_tokens": start,
                     "truncated": n_whole > p["max_len"], "nll_ft": nll_ft,
                     "nll_base": nll_base})

    summary = {"unit": "completion tokens of clean evaluation exchanges", **_ratio(rows)}
    if disjoint is not None:
        subset = [rows[i] for i in disjoint if i < len(rows)]
        summary["disjoint_reference_subset"] = {
            "note": "the 276 shared clean references (absent from every main training set)",
            **_ratio(subset)}
    return {
        "method": METHOD_ID,
        "parameters": {**p, "seed": seed, "assistant_open": ASSIST_OPEN,
                       "scored_tokens": "every token after the assistant header, truncated "
                                        "to max_len (response, think block, <|im_end|>)"},
        "outputs": {"per_exchange": rows},
        "summary": summary,
    }
