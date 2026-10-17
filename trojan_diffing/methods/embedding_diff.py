"""EMB: rank vocabulary tokens by how much fine-tuning changed their embedding row.

Supports the writeup's embeddings ablation: when the tied embedding/unembedding matrix
is trainable, does ranking tokens by the L2 norm of their row change surface trigger or
payload tokens? Qwen3.5 ties the input embedding and the output head, so one matrix
covers both. For organisms trained with frozen embeddings every change is exactly zero
and the ranking is meaningless; the result says so.

Two orderings: absolute change ``||E_ft[i] - E_base[i]||`` and change relative to the
base row norm. For each, the top-100 tokens are listed and scored with
``common.score_tokens``, and the rank of every scored trigger/payload token is reported
(the writeup's statement concerns the top 10,000).
"""
from __future__ import annotations

import pathlib

import torch

from ..common import score_tokens, token_sets

METHOD_ID = "EMB"
DEFAULTS = {"top_n": 100, "rank_threshold": 10000}


@torch.no_grad()
def run(ft, base, tok, *, seed: int, params: dict | None = None, refs: dict | None = None,
        artifact_dir: pathlib.Path | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    if set(p) - set(DEFAULTS):
        raise ValueError(f"unknown EMB parameters: {sorted(set(p) - set(DEFAULTS))}")
    e_ft = ft.get_input_embeddings().weight
    e_base = base.get_input_embeddings().weight.to(e_ft.device)
    if e_ft.shape != e_base.shape:
        raise ValueError(f"embedding shapes differ: {tuple(e_ft.shape)} vs {tuple(e_base.shape)}")
    tied = e_ft.data_ptr() == ft.get_output_embeddings().weight.data_ptr()
    delta = (e_ft.float() - e_base.float()).norm(dim=-1)
    base_norm = e_base.float().norm(dim=-1).clamp_min(1e-8)
    changed = int((delta > 0).sum())
    scored = {tid: (cat, s) for cat, ids in token_sets(tok).items() for tid, s in ids.items()}

    orderings = {}
    for name, values in (("absolute", delta), ("relative", delta / base_norm)):
        order = values.argsort(descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(1, len(order) + 1, device=order.device)
        top = [int(i) for i in order[: p["top_n"]]]
        orderings[name] = {
            "top": [{"id": i, "token": tok.decode([i]), "value": float(values[i])} for i in top],
            "score_top_n": score_tokens(top, tok),
            "scored_token_ranks": {s: {"id": tid, "category": cat, "rank": int(ranks[tid]),
                                       "value": float(values[tid])}
                                   for tid, (cat, s) in scored.items()},
        }
        orderings[name]["scored_tokens_within_threshold"] = sorted(
            s for s, r in orderings[name]["scored_token_ranks"].items()
            if r["rank"] <= p["rank_threshold"] and r["category"] in ("trigger", "payload"))

    summary = {
        "unit": f"top-{p['top_n']} tokens by embedding-row change (absolute and relative)",
        "n_rows_changed": changed,
        "vocab_size": int(delta.numel()),
        "embeddings_trained": changed > 0,
        "tied_embedding_and_head": bool(tied),
        "max_row_change": float(delta.max()),
        "trigger_partial": any(o["score_top_n"]["trigger_partial"] for o in orderings.values()),
        "payload_partial": any(o["score_top_n"]["payload_partial"] for o in orderings.values()),
        "trigger_full": False,
        "payload_full": False,
        "scored_tokens_within_rank_threshold": {k: o["scored_tokens_within_threshold"]
                                                 for k, o in orderings.items()},
        "rank_threshold": p["rank_threshold"],
    }
    if not changed:
        summary["note"] = "embeddings were frozen: every row change is zero and the ranking is arbitrary"
    return {"method": METHOD_ID, "parameters": {**p, "seed": seed}, "outputs": orderings, "summary": summary}
