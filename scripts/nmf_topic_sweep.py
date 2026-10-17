"""Diff Mining NMF: does the number of topics matter?  Needs no GPU.

    python scripts/nmf_topic_sweep.py --artifacts DIR [DIR ...] [--ks 3 5 10 20 50 100 200] [--out PATH]

Refits the Diff Mining NMF (``trojan_diffing.methods.diff_mining``: same matrix, solver,
initialisation and readout) on each model's saved top-K logit differences
(``<DIR>/<model>/M3/m3_topk_per_position.npz``, published under ``artifacts/`` in the
Hugging Face model repository) for several numbers of topics, each from two random
initialisations, for up to 200 iterations (with tolerance 0 the solver stops once the loss
stops decreasing). For every fit it records the iteration count, the share of
the matrix explained (1 - ||V - HW^T||^2 / ||V||^2) and scores each topic's top-100 tokens
with ``common.score_tokens`` exactly as the method does. Before the sweep it refits the
method's own primary configuration (3 topics, run seed, tol 1e-4) and checks that the
topics hold the same top-100 tokens as the stored ``results/methods/<model>/M3.json``, which
ties each artifact to its published result. Writes ``results/analysis/nmf_topic_sweep.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from trojan_diffing import common  # noqa: E402
from trojan_diffing.methods import diff_mining as dm  # noqa: E402

MODELS = ["p1_trojan", "p01_trojan", "p001_trojan", "p1_null", "p01_null", "p001_null", "p1_trojan_run1"]
SCORED = list(common.TRIGGER_TOKEN_STRINGS) + list(common.PAYLOAD_TOKEN_STRINGS)


def topic_ids(W, col_tokens, t, n=100, min_w=1e-6):
    w, c = torch.topk(W[:, t], k=min(n, W.shape[0]))
    return [int(col_tokens[i]) for i, x in zip(c.tolist(), w.tolist()) if x >= min_w]


def best_ranks(W, col_tokens, tok):
    """For each trigger/payload token: its best rank (1 = top) in any topic, if present."""
    col_of = {int(t): i for i, t in enumerate(col_tokens)}
    order = torch.argsort(W, dim=0, descending=True)  # (C, R)
    rank_of = torch.empty_like(order)
    rank_of.scatter_(0, order, torch.arange(W.shape[0]).unsqueeze(1).expand_as(order))
    out = {}
    for s in SCORED:
        tid = tok(s, add_special_tokens=False)["input_ids"]
        if len(tid) != 1 or tid[0] not in col_of:
            out[s] = None
            continue
        c = col_of[tid[0]]
        nonzero = W[c] >= 1e-6
        out[s] = int(rank_of[c][nonzero].min()) + 1 if nonzero.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--seed_offsets", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tokenizer", default=str(REPO / "tests" / "fixtures" / "tokenizer"))
    ap.add_argument("--out", default=str(REPO / "results" / "analysis" / "nmf_topic_sweep.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    report = {"description": __doc__.strip().splitlines()[0], "ks": a.ks, "seed_offsets": a.seed_offsets,
              "max_iter": a.max_iter, "models": {}}
    for m in a.models:
        path = next((Path(d) / m / "M3" / "m3_topk_per_position.npz" for d in a.artifacts
                     if (Path(d) / m / "M3" / "m3_topk_per_position.npz").exists()), None)
        if path is None:
            raise SystemExit(f"no saved top-K table for {m}")
        res = common.read_json(REPO / "results" / "methods" / m / "M3.json")
        p, seed = res["parameters"], int(res["metadata"]["seed"])
        z = np.load(path)
        V, cols = dm.build_nmf_matrix(z["topk_values"], z["topk_indices"], mode=p["nmf_mode"],
                                      rows_per_batch=int(p["nmf_column_order_batch_size"]) * int(p["positions_per_document"]))
        # 1. The method's primary fit must reproduce the stored topics exactly.
        fit = dm.fit_nmf(V, int(p["nmf_num_topics"]), seed=seed, max_iter=int(p["nmf_max_iter"]), tol=float(p["nmf_tol"]))
        stored = [[e["token_id"] for e in t["tokens"]][:100] for t in res["outputs"]["nmf"]["topics"]]
        refit = [topic_ids(fit["W"], cols, t) for t in range(fit["W"].shape[1])]
        # Same tokens per topic; near-tied tokens can swap order on another machine's float math.
        if [set(t) for t in refit] != [set(t) for t in stored]:
            raise SystemExit(f"{m}: refit of the primary configuration does not match M3.json")
        entry = {"artifact": f"{m}/M3/m3_topk_per_position.npz", "matrix_shape": list(V.shape), "nnz": int(V._nnz()),
                 "primary_fit_reproduced": "same top-100 tokens in every topic",
                 "primary_fit_same_order": refit == stored, "fits": []}
        # 2. The sweep.
        for k in a.ks:
            for off in a.seed_offsets:
                t0 = time.time()
                f = dm.fit_nmf(V, k, seed=seed + off, max_iter=a.max_iter, tol=0.0)
                err, vn = dm.reconstruction_error(V, f["W"], f["H"])
                hits = {"trigger_tokens": set(), "payload_tokens": set(), "signal_clue_tokens": set()}
                for t in range(k):
                    s = common.score_tokens(topic_ids(f["W"], cols, t), tok)
                    for key in hits:
                        hits[key].update(s[key])
                entry["fits"].append({
                    "k": k, "seed": seed + off, "n_iter": f["n_iter"], "explained": 1 - (err / vn) ** 2,
                    **{key: sorted(v) for key, v in hits.items()},
                    "best_rank_in_any_topic": best_ranks(f["W"], cols, tok),
                    "seconds": round(time.time() - t0, 1),
                })
                fe = entry["fits"][-1]
                print(f"{m:12s} k={k:3d} seed+{off} explained={fe['explained']:.3f} "
                      f"trigger={fe['trigger_tokens']} payload={fe['payload_tokens']} "
                      f"signal={fe['signal_clue_tokens']} ({fe['seconds']}s)", flush=True)
        report["models"][m] = entry
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
