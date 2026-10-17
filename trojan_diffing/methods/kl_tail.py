"""M2 per-token KL tail: where do the fine-tuned and base next-token distributions differ?

For every position of every clean reference text (raw text, no chat template,
``add_special_tokens=False``, truncated to 512 tokens, texts shorter than 4 tokens
skipped) compute KL(ft || base) between the two next-token distributions. Report the
200 positions with the largest KL, each with its input token and that token plus its
12 preceding tokens as context, and the mean KL over all positions.

The readout consists of INPUT tokens of the clean references. A trigger or payload hit
here means the reference text itself contained that token near a high-KL position; it
is not generated recovery. Scoring:
  * ``score_tokens`` on the token ID at each of the top-200 positions;
  * ``score_text`` on each context string.

Origin: ``kl_tail`` in the original project's ``methods_baseline.py`` as invoked by the main-sweep
``run_method.py`` (``docs`` = the first 120 shared clean references).

Changes from the original script:
  * Only the top-200 positions are decoded (the original decoded every position, then
    sorted). The ranking is identical: a stable descending sort on the same float32
    values in the same document/position order.
  * Every position's KL value, token ID, document index and position is saved to
    ``kl_per_position.npz`` in ``artifact_dir``.
  * Token IDs, document indices and positions are added to the top-200 rows, plus the
    shared scores.

Added readout (not in the writeup): predicted next tokens at the high-KL positions.
  Rationale: the original readout inspects INPUT tokens of clean references, which by
  construction cannot contain the trigger or the payload, so on its own it can never score a
  hit (weak pieces such as '[[' are credited only after the follow-up check of
  scripts/fragment_followup.py).
  In a smoke run on the 10%-poison model the top KL position was an input ' [[' token
  (KL 14.6), i.e. the place where the fine-tuned model's next-token distribution
  diverged most was right after a payload-like bracket; the natural question is what
  the fine-tuned model predicts there. In the same forward passes (no extra passes),
  the top-``n_next_tokens`` (5) next tokens of both models are stored at EVERY position
  (ids and float32 probabilities from the same log-softmax used for the KL; ~15k
  positions x 5, negligible memory), and then read off at the top-200 positions:
  * each top row gets ``ft_next_top5`` and ``base_next_top5`` (lists of
    ``{"id", "token", "prob"}``, most probable first), ``ft_next_argmax_id`` /
    ``ft_next_argmax_token``, and ``ft_next_top5_score`` (``score_tokens`` on the
    fine-tuned top-5 ids);
  * ``outputs["predicted_tokens"]``: the fine-tuned top-5 ids at each top-200 position
    (200 ranked token lists, for the summarizer);
  * ``summary["predicted_token_list_score"]``: ``score_tokens`` over the fine-tuned
    top-1 predictions at the top-200 positions (deduplicated, rank order), plus
    ``summary["predicted_next_tokens"]`` with per-position counts, the top-5 union and
    the base model's top-5 union as a control.
  These ARE model outputs (next-token predictions), unlike the input-token readout, but
  they are single tokens: full trigger/payload recovery is impossible by construction.
  The per-position top-5 arrays are also saved in ``kl_per_position.npz``.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch

from .. import common

METHOD_ID = "M2"

DEFAULTS = {
    "n_docs": 120,
    "max_len": 512,
    "min_tokens": 4,
    "topk": 200,
    "context_tokens": 12,
    "n_top_token_counts": 30,
    "n_next_tokens": 5,          # added readout: top-n next tokens of each model per position
}

ARTIFACT_NAME = "kl_per_position.npz"


def _clean_refs(refs):
    if refs is not None and "clean" in refs:
        return list(refs["clean"])
    from ..references import clean_references

    return clean_references()


@torch.no_grad()
def per_position_kl(ft, base, tok, docs, *, max_len=512, min_tokens=4, n_next=5) -> dict:
    """KL(ft || base) of the next-token distributions at every retained position.

    Also keeps each model's top-``n_next`` next-token ids and probabilities at every
    position (``ft_next_ids`` / ``ft_next_prob`` / ``base_next_ids`` /
    ``base_next_prob``, shape (positions, n_next), most probable first). They are read
    from the same float32 log-softmax as the KL, so the KL values are unchanged.
    """
    kls, ids, doc_index, position = [], [], [], []
    next_ids = {"ft": [], "base": []}
    next_prob = {"ft": [], "base": []}
    for j, d in enumerate(docs):
        enc = tok(d, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_len).to(ft.device)
        if enc["input_ids"].shape[1] < min_tokens:
            continue
        lf = ft(**enc).logits[0].float().log_softmax(-1)
        lb = base(**enc).logits[0].float().log_softmax(-1)
        kl = (lf.exp() * (lf - lb)).sum(-1)          # KL(ft||base) per position
        for name, logp in (("ft", lf), ("base", lb)):
            top = logp.topk(n_next, dim=-1)          # sorted, most probable first
            next_ids[name].append(top.indices.cpu().numpy().astype(np.int64))
            next_prob[name].append(top.values.exp().cpu().numpy().astype(np.float32))
        seq = enc["input_ids"][0]
        kls.append(kl.cpu().numpy().astype(np.float32))
        ids.append(seq.cpu().numpy().astype(np.int64))
        doc_index.append(np.full(len(seq), j, dtype=np.int64))
        position.append(np.arange(len(seq), dtype=np.int64))
    if not kls:
        raise ValueError(f"no reference text has at least {min_tokens} tokens")
    return {"kl": np.concatenate(kls), "token_id": np.concatenate(ids),
            "doc": np.concatenate(doc_index), "position": np.concatenate(position),
            "ft_next_ids": np.concatenate(next_ids["ft"]),
            "ft_next_prob": np.concatenate(next_prob["ft"]),
            "base_next_ids": np.concatenate(next_ids["base"]),
            "base_next_prob": np.concatenate(next_prob["base"])}


def kl_order(table) -> np.ndarray:
    """All positions by descending KL; ties keep document/position order (stable sort)."""
    return np.argsort(-table["kl"].astype(np.float64), kind="stable")


def top_positions(table, order, tok, *, topk=200, context_tokens=12) -> list[dict]:
    """The ``topk`` largest-KL positions with their input token and preceding context."""
    kl = table["kl"].astype(np.float64)
    rows = []
    for rank, flat in enumerate(order[:topk].tolist()):
        start = max(flat - context_tokens, flat - int(table["position"][flat]))
        context_ids = table["token_id"][start:flat + 1].tolist()
        token_id = int(table["token_id"][flat])
        context = tok.decode(context_ids)
        rows.append({"rank": rank, "kl": float(kl[flat]), "token_id": token_id,
                     "token": tok.decode([token_id]), "context": context,
                     "doc": int(table["doc"][flat]), "position": int(table["position"][flat]),
                     "token_score": common.score_tokens([token_id], tok),
                     "context_score": common.score_text(context),
                     **_next_token_fields(table, flat, tok)})
    return rows


def _next_list(table, name, flat, tok) -> list[dict]:
    ids = table[f"{name}_next_ids"][flat].tolist()
    probs = table[f"{name}_next_prob"][flat].tolist()
    return [{"id": int(i), "token": tok.decode([int(i)]), "prob": float(q)}
            for i, q in zip(ids, probs)]


def _next_token_fields(table, flat, tok) -> dict:
    """Added readout: what each model predicts next at this position (see docstring)."""
    if "ft_next_ids" not in table:
        return {}
    ft_next = _next_list(table, "ft", flat, tok)
    return {"ft_next_top5": ft_next,
            "base_next_top5": _next_list(table, "base", flat, tok),
            "ft_next_argmax_id": ft_next[0]["id"],
            "ft_next_argmax_token": ft_next[0]["token"],
            "ft_next_top5_score": common.score_tokens([e["id"] for e in ft_next], tok)}


def _save_table(table, artifact_dir):
    if artifact_dir is None:
        return None
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / ARTIFACT_NAME
    np.savez_compressed(path, kl=table["kl"], token_id=table["token_id"].astype(np.int32),
                        doc=table["doc"].astype(np.int32),
                        position=table["position"].astype(np.int32),
                        ft_next_ids=table["ft_next_ids"].astype(np.int32),
                        ft_next_prob=table["ft_next_prob"],
                        base_next_ids=table["base_next_ids"].astype(np.int32),
                        base_next_prob=table["base_next_prob"])
    return {"file": ARTIFACT_NAME, "sha256": common.sha256_file(path),
            "arrays": {"kl": "float32 KL(ft||base) per position",
                       "token_id": "input token at that position",
                       "doc": "index into the reference list used",
                       "position": "token position within the document",
                       "ft_next_ids": "fine-tuned top-n next-token ids per position",
                       "ft_next_prob": "their float32 probabilities",
                       "base_next_ids": "base top-n next-token ids per position",
                       "base_next_prob": "their float32 probabilities"}}


def run(ft, base, tok, *, seed, params=None, refs=None, artifact_dir=None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    docs = _clean_refs(refs)[:p["n_docs"]]
    table = per_position_kl(ft, base, tok, docs, max_len=p["max_len"],
                            min_tokens=p["min_tokens"], n_next=p["n_next_tokens"])
    order = kl_order(table)
    top = top_positions(table, order, tok, topk=p["topk"], context_tokens=p["context_tokens"])
    n_positions = int(table["kl"].shape[0])
    # Sequential double-precision sum over the sorted values, exactly as the original.
    mean_kl = sum(table["kl"][order].tolist()) / max(n_positions, 1)

    token_ids = [r["token_id"] for r in top]
    summary = {
        "unit": (f"top-{len(top)} KL positions; the scored tokens and contexts are INPUT "
                 "tokens of clean references, not generated recovery"),
        "n": len(top),
        "trigger_full": sum(r["token_score"]["trigger_full"] for r in top),
        "trigger_partial": sum(r["token_score"]["trigger_partial"] for r in top),
        "payload_full": sum(r["token_score"]["payload_full"] for r in top),
        "payload_partial": sum(r["token_score"]["payload_partial"] for r in top),
        "token_list_score": common.score_tokens(token_ids, tok),
        "contexts": common.summarize_texts(r["context"] for r in top),
        "mean_kl": mean_kl,
        "n_positions": n_positions,
        "n_docs": len(docs),
        "n_docs_used": int(len(np.unique(table["doc"]))),
        "input_tokens_not_generated": True,
    }
    # Added readout (not in the writeup): next-token predictions at the same positions.
    predicted = [[e["id"] for e in r["ft_next_top5"]] for r in top]
    argmax_ids = list(dict.fromkeys(r["ft_next_argmax_id"] for r in top))   # dedup, rank order
    argmax_scores = [common.score_tokens([r["ft_next_argmax_id"]], tok) for r in top]
    summary["predicted_token_list_score"] = common.score_tokens(argmax_ids, tok)
    summary["predicted_next_tokens"] = {
        "unit": (f"fine-tuned top-{p['n_next_tokens']} next-token predictions at the "
                 f"top-{len(top)} KL positions (model predictions, single tokens)"),
        "n": len(top),
        "trigger_partial": sum(r["ft_next_top5_score"]["trigger_partial"] for r in top),
        "payload_partial": sum(r["ft_next_top5_score"]["payload_partial"] for r in top),
        "trigger_partial_argmax": sum(sc["trigger_partial"] for sc in argmax_scores),
        "payload_partial_argmax": sum(sc["payload_partial"] for sc in argmax_scores),
        "trigger_full": 0,
        "payload_full": 0,
        "n_unique_argmax": len(argmax_ids),
        "n_argmax_differs_from_base": sum(
            r["ft_next_argmax_id"] != r["base_next_top5"][0]["id"] for r in top),
        "top5_token_list_score": common.score_tokens(
            [i for ids in predicted for i in ids], tok),
        "base_top5_token_list_score_control": common.score_tokens(
            [e["id"] for r in top for e in r["base_next_top5"]], tok),
    }
    counts = collections.Counter(r["token"] for r in top)
    return {
        "method": METHOD_ID,
        "parameters": {**p, "seed": seed, "add_special_tokens": False,
                       "chat_template": False, "kl_direction": "KL(ft || base)"},
        "outputs": {
            "n_positions": n_positions,
            "mean_kl": mean_kl,
            "top": top,
            "top_tokens": [[t, c] for t, c in counts.most_common(p["n_top_token_counts"])],
            # Added readout: fine-tuned top-5 next-token ids at each top position.
            "predicted_tokens": predicted,
            "artifact": _save_table(table, artifact_dir),
        },
        "summary": summary,
    }
