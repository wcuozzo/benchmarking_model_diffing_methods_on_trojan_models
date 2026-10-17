"""M3 Diff Mining (Kocher et al., arXiv 2608.26462) with two token orderings.

One pass over the clean reference prompts computes, for every retained position, the
top-K entries of ``logits_ft - logits_base`` (float32). Two orderings are read off the
same top-K values/indices:

1. ``topk_occurrence`` -- the protocol behind the writeup's M3 result: each token gets +1
   for every position whose top-K contains it, and the top-100 tokens by count are
   inspected. Ties are broken by first occurrence, exactly as ``collections.Counter``
   ``.most_common`` did in the original implementation.
2. ``nmf`` -- a port of the authors' magnitude-based NMF token-topic ordering
   (``diffing/methods/diff_mining/token_ordering.py::NmfOrderingType``):
     * matrix ``V`` (positions x tokens): one row per analysed position; one column per
       token that appears in at least one position's top-K positive diffs, columns
       numbered in the authors' first-appearance order; entry = the logit difference for
       the top-K tokens of that row and 0 elsewhere; then ReLU and zeros dropped (sparse);
     * torchnmf 0.3.5 ``NMF(V.shape, rank).fit(V, beta=2, tol=1e-4, max_iter=200)`` on the
       sparse ``V`` (multiplicative updates, W updated before H, eps = float32 eps,
       convergence checked every 10 iterations), re-implemented here in closed form
       because torchnmf is not a dependency. The optional orthogonal variant
       (``orthogonal_nmf.py::fit_nmf_orthogonal``: torchnmf ``BetaMu`` with an
       orthogonality penalty, fixed iteration count) is ported too but off by default,
       as in the authors' config;
     * each topic's tokens ordered by their weight in ``W.T``, weights below 1e-6 dropped,
       ``num_tokens`` = max(100, top_k) = 100 tokens per topic (the authors'
       ``_run_ordering_types_and_write``).

Methods never see the training data. Token lists cannot contain the multi-token trigger
or payload, so full recovery is impossible here by construction.

Added readout (not in the writeup): NMF initialization sensitivity.
  NMF topics depend on the random initialization (a review found only 14% top-token
  overlap between one topic of the primary fit and a 200-iteration refit on one model).
  The primary configuration (same matrix, same rank, same stopping rule, same orthogonal
  setting) is refitted with seed ``seed + offset`` for each offset in
  ``nmf_extra_seeds`` (default 1, 2, 3, 4). For each seed the per-topic top-100 token
  ids are kept and scored with ``score_tokens`` (``outputs["nmf"]["extra_seeds"]``),
  with each primary topic's best top-token overlap with that seed's topics. The summary
  gains ``nmf_extra_seed_fits`` (keyed by the seed used), ``nmf_seeds_fitted`` and
  ``nmf_any_seed_trigger_partial`` / ``nmf_any_seed_payload_partial``: True if the
  primary fit OR any extra seed has a topic whose top-100 tokens contain a trigger /
  payload token. The primary fit's outputs and the headline ``trigger_partial`` /
  ``payload_partial`` are unchanged.
"""
from __future__ import annotations

import math
import pathlib
import time

import numpy as np
import torch

from ..common import score_tokens, token_sets

METHOD_ID = "M3"

DEFAULTS = {
    # ---- reference contexts (shared logit-difference pass)
    "n_prompts": None,  # None = every prompt in refs["clean"] (276)
    "positions_per_document": 30,  # T; prompts with fewer tokens are discarded
    "topk_per_position": 100,  # K
    "batch_size": 1,  # prompts per forward pass (retained prompts all have exactly T tokens)
    # ---- ordering 1: top-K occurrence (original M3 protocol)
    "output_candidate_count": 100,  # tokens inspected / scored
    "saved_ranking_count": 1000,  # ranking IDs+counts stored for GPU-free re-scoring
    # ---- ordering 2: NMF (authors' configs/diffing/method/diff_mining.yaml defaults)
    "nmf_num_topics": 3,
    "nmf_beta": 2.0,  # 2 = squared Frobenius; only beta=2 is implemented
    "nmf_mode": "logit_diff_magnitude",  # or "binary_occurrence"
    "nmf_orthogonal": False,
    "nmf_orthogonal_weight": 1000.0,  # used only when nmf_orthogonal is True
    "nmf_max_iter": 200,
    "nmf_tol": 1e-4,  # torchnmf fit() default; the orthogonal path has no tolerance check
    "nmf_tokens_per_topic": 100,  # authors' effective value: max(num_tokens_to_plot, top_k)
    "nmf_top_n_tokens_per_topic": 200,  # config value; the authors' code only uses it in a dir name
    "nmf_agent_token_budget": 100,  # agent.overview.top_k_tokens, split across topics
    "nmf_column_order_batch_size": 16,  # authors' batch_size; fixes their column numbering
    "nmf_min_weight": 1e-6,  # authors drop topic tokens below this weight
    "nmf_num_topics_sensitivity": [5],  # extra fits on the same matrix (paper G.3 used 5)
    # torchnmf's stopping rule divides by the loss at the (unscaled |N(0,1)|) random init,
    # so it typically stops after 20-30 iterations; refit with tol=0 to check that.
    "nmf_full_iteration_check": True,
    # Added readout: refit the primary configuration with seed + offset for each offset.
    "nmf_extra_seeds": [1, 2, 3, 4],
}

# float32 machine epsilon, torchnmf.constants.eps
_EPS = float(torch.finfo(torch.float32).eps)

DEVIATIONS_FROM_PAPER = {
    "reference_corpus": (
        "The repository's clean reference prompts (held-out Tulu-style instruction user "
        "strings, fed as raw text without a chat template) instead of the authors' "
        "pretraining-text sample (config: science-of-finetuning/fineweb-1m-sample; paper: "
        "'random webtext', e.g. FineWeb)."
    ),
    "n_documents": (
        "All retained reference prompts (276 for the shared set) instead of the paper's "
        "default N=1000 (Appendix C) or the code default max_samples=8192."
    ),
    "positions_per_document": (
        "T=30, the paper default (Appendix C); the code default max_tokens_per_sample is 64."
    ),
    "tokenization": (
        "tok(text, add_special_tokens=False, truncation=True, max_length=T) as in the "
        "original M3 run; the authors use tok.encode(text[:10*T], add_special_tokens=True)"
        "[:T]. The Qwen3.5 tokenizer adds no BOS, and both give identical first-30 token "
        "IDs for all 276 shared reference prompts (checked when this module was written)."
    ),
    "logit_diff_precision": (
        "Logits are upcast to float32 before subtraction (original M3 run). The authors "
        "subtract in the models' output dtype (bfloat16 for bfloat16 models), which can "
        "change top-K membership for near-ties at the K-th value."
    ),
    "batching": (
        "One prompt per forward pass by default (original M3 run); the authors batch 16 "
        "right-padded documents with an attention mask. Retained documents all have "
        "exactly T tokens, so there are no padding positions either way."
    ),
    "nmf_solver": (
        "torchnmf 0.3.5 NMF.fit (sparse V, beta=2 multiplicative updates, tol=1e-4, "
        "max_iter=200) is re-implemented in closed form in plain torch because torchnmf "
        "is not a dependency. Updates, eps handling, update order (W then H), loss "
        "definition and stopping rule follow the torchnmf source."
    ),
    "nmf_initialisation": (
        "W then H drawn as |N(0,1)| from a dedicated CPU torch.Generator seeded with the "
        "run seed (torchnmf draws them the same way from the global torch RNG, which the "
        "authors seed once with cfg.seed=42 at pipeline start and which other code then "
        "consumes), so the authors' exact initialisation cannot be reproduced."
    ),
    "nmf_device": "NMF runs on CPU in float32; the authors run it on CUDA when available.",
    "nmf_extra_seeds": (
        "Not in the paper or the writeup: the primary NMF configuration is refitted from "
        "seed + 1..4 to measure initialization sensitivity; the primary fit is unchanged."
    ),
    "nmf_tokens_per_topic": (
        "100 tokens per topic: the authors' code writes num_tokens = max(visualization."
        "num_tokens_to_plot=100, top_k=100) tokens per topic; the config field "
        "top_n_tokens_per_topic=200 is only used in the output directory name. Scores at "
        "200 tokens per topic are reported too."
    ),
    "nmf_num_topics": (
        "3 topics (authors' config default). The paper says the number of topics depends "
        "on the use case and Appendix G.3 used 5, so 5 is reported as a sensitivity fit."
    ),
    "no_llm_interpretation": (
        "No LLM token-relevance grader (authors: gpt-5-mini) and no interpretability "
        "agent. Token lists are scored with trojan_diffing.common.score_tokens on token "
        "IDs. Full trigger/payload recovery is impossible from token lists."
    ),
    "orderings_not_reported": (
        "The authors' fraction_positive_diff ordering and top-K negative counts are not "
        "reported (per-token sum of diffs and positive counts are saved in the artifact)."
    ),
    "logit_extraction": (
        "Direct output logits (the config's logit_extraction.method default). The "
        "authors' agent-overview default reads a patchscope-lens variant, which is not run."
    ),
    "topk_occurrence_tie_breaking": (
        "Count ties broken by first occurrence (collections.Counter.most_common, as in the "
        "original M3 run); the authors' torch.topk over occurrence rates leaves tie order "
        "unspecified."
    ),
}


# --------------------------------------------------------------------- logit-diff pass
def tokenize_fixed_length(tok, prompts, positions):
    """Return (kept prompt indices, discarded indices, list of length-T token-ID lists)."""
    kept, discarded, ids = [], [], []
    for i, prompt in enumerate(prompts):
        enc = tok(prompt, add_special_tokens=False, truncation=True, max_length=positions)
        row = list(enc["input_ids"])
        if len(row) < positions:  # fixed-T protocol: discard, never shorten
            discarded.append(i)
            continue
        kept.append(i)
        ids.append(row[:positions])
    return kept, discarded, ids


def _device_of(model):
    return next(model.parameters()).device


@torch.no_grad()
def collect_topk(ft, base, tok, prompts, *, positions, topk, batch_size=1):
    """Run each model once per retained prompt and keep the per-position top-K diffs.

    Returns numpy arrays on CPU: ``topk_values`` (R, K) float32 and ``topk_indices``
    (R, K) int64 with R = n_retained * T rows ordered (prompt, position) and each row
    sorted by decreasing diff, plus per-token ``sum_logit_diff`` and ``count_positive``
    over all positions (the authors' SharedTokenStats).
    """
    kept, discarded, ids = tokenize_fixed_length(tok, prompts, positions)
    device = _device_of(ft)
    values, indices = [], []
    sum_diff = count_pos = None
    for start in range(0, len(ids), batch_size):
        input_ids = torch.tensor(ids[start:start + batch_size], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        lf = ft(input_ids=input_ids, attention_mask=mask).logits.float()
        lb = base(input_ids=input_ids, attention_mask=mask).logits.float()
        diff = lf - lb  # (B, T, V) float32
        del lf, lb
        if sum_diff is None:
            sum_diff = torch.zeros(diff.shape[-1], dtype=torch.float64, device=diff.device)
            count_pos = torch.zeros(diff.shape[-1], dtype=torch.int64, device=diff.device)
        sum_diff += diff.sum(dim=(0, 1), dtype=torch.float64)
        count_pos += (diff > 0).sum(dim=(0, 1))
        v, ix = diff.topk(topk, dim=-1)  # sorted, largest first
        values.append(v.reshape(-1, topk).cpu())
        indices.append(ix.reshape(-1, topk).cpu())
        del diff, v, ix
    if not ids:
        raise ValueError(f"no prompt has at least {positions} tokens")
    return {
        "kept_prompt_indices": kept,
        "discarded_prompt_indices": discarded,
        "input_ids": ids,
        "topk_values": torch.cat(values).numpy().astype(np.float32),
        "topk_indices": torch.cat(indices).numpy().astype(np.int64),
        "sum_logit_diff": sum_diff.cpu().numpy(),
        "count_positive": count_pos.cpu().numpy(),
        "vocab_size": int(sum_diff.shape[0]),
    }


# ------------------------------------------------------------ ordering 1: occurrence
def topk_occurrence_ranking(topk_indices, vocab_size):
    """All tokens with a non-zero top-K count, ranked like Counter.most_common.

    Counter iterates insertion order (first occurrence in the flattened (prompt,
    position, rank) sequence) and most_common is a stable sort by count, so ties are
    broken by first occurrence.
    """
    seq = np.asarray(topk_indices, dtype=np.int64).reshape(-1)
    counts = np.bincount(seq, minlength=vocab_size)
    uniq, first = np.unique(seq, return_index=True)
    order = np.lexsort((first, -counts[uniq]))
    ranking = uniq[order]
    return ranking, counts[ranking]


# ------------------------------------------------------------------ ordering 2: NMF
def authors_column_order(topk_indices, rows_per_batch):
    """Token IDs in the authors' column order (NmfOrderingType.collect_batch).

    The authors loop over batches of ``batch_size`` documents and, inside a batch, over
    top-K rank (outer) and row (inner), giving each unseen token the next column index.
    """
    idx = np.asarray(topk_indices, dtype=np.int64)
    seq = np.concatenate(
        [idx[s:s + rows_per_batch].T.reshape(-1) for s in range(0, idx.shape[0], rows_per_batch)]
    )
    uniq, first = np.unique(seq, return_index=True)
    return uniq[np.argsort(first, kind="stable")]


def build_nmf_matrix(topk_values, topk_indices, *, mode="logit_diff_magnitude", rows_per_batch=None):
    """The authors' NMF input: sparse (positions x tokens) matrix of top-K positive diffs.

    Entry (r, c) is the logit difference of column token c at position r when c is in
    r's top-K, else 0 (``binary_occurrence`` uses 1.0 instead). As in
    ``NmfOrderingType._run_nmf``, values are ReLU-ed and zero entries are dropped.
    Returns (coalesced float32 sparse COO tensor, column token IDs).
    """
    vals = np.asarray(topk_values, dtype=np.float32)
    idx = np.asarray(topk_indices, dtype=np.int64)
    n_rows, k = idx.shape
    col_tokens = authors_column_order(idx, rows_per_batch or n_rows)
    by_id = np.argsort(col_tokens)
    cols = by_id[np.searchsorted(col_tokens[by_id], idx.reshape(-1))]
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    if mode == "binary_occurrence":
        data = np.ones(idx.size, dtype=np.float32)
    elif mode == "logit_diff_magnitude":
        data = vals.reshape(-1)
    else:
        raise ValueError(f"unknown NMF mode {mode!r}")
    data = np.maximum(data, 0.0)  # torch.relu(V_raw.values())
    keep = data > 0
    V = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows[keep], cols[keep]])),
        torch.from_numpy(data[keep].astype(np.float32)),
        size=(n_rows, len(col_tokens)),
    ).coalesce()
    return V, col_tokens


def _sparse_loss(V, Vt, W, H, v_norm):
    """torchnmf's sparse beta=2 loss, returned as sqrt(2 * loss) = ||V - H W^T||_F."""
    pos = 0.5 * torch.sum((H @ (W.T @ W)) * H)  # _nmf_sp_recon_beta_pos_neg, beta == 2
    neg = torch.sum(torch.sparse.mm(Vt, H) * W)
    return float((v_norm + pos - neg).mul(2).sqrt())


def fit_nmf(V, rank, *, seed, beta=2.0, max_iter=200, tol=1e-4, orthogonal_weight=0.0):
    """torchnmf multiplicative-update NMF, V (N x C) ~= H @ W.T, W (C x R), H (N x R).

    ``orthogonal_weight == 0`` ports ``NMF(V.shape, rank).fit(V, beta, tol, max_iter)``
    (sparse path). ``orthogonal_weight > 0`` ports ``fit_nmf_orthogonal`` (``BetaMu``
    with ``pos += w * (p.sum(1) - p)`` for W and H, fixed ``max_iter`` iterations).
    """
    if beta != 2:
        raise NotImplementedError("only the authors' default beta=2 is implemented")
    V = V.coalesce().to(torch.float32)
    Vt = V.t().coalesce()
    n_rows, n_cols = V.shape
    gen = torch.Generator().manual_seed(int(seed))
    # BaseComponent registers W (C, R) before H (N, R), each torch.randn(...).abs().
    W = torch.randn(n_cols, rank, generator=gen).abs()
    H = torch.randn(n_rows, rank, generator=gen).abs()
    v_norm = V.values() @ V.values() * 0.5  # _get_V_norm, beta == 2
    ortho = float(orthogonal_weight)
    check = ortho == 0.0  # fit_nmf_orthogonal runs a fixed number of steps

    def update(p, neg, pos):
        neg = neg.clamp_min(0)
        pos = pos.clamp_min(0)
        if ortho > 0:
            pos = pos + ortho * (p.sum(1, keepdim=True) - p)
        p.mul_((neg + _EPS) / (pos + _EPS))

    loss_init = _sparse_loss(V, Vt, W, H, v_norm)
    previous = loss_init
    trace = [loss_init]
    stopped_by_tol = False
    n_iter = 0
    for n_iter in range(max_iter):
        update(W, torch.sparse.mm(Vt, H), W @ (H.T @ H))
        update(H, torch.sparse.mm(V, W), H @ (W.T @ W))
        if check and n_iter % 10 == 9:
            loss = _sparse_loss(V, Vt, W, H, v_norm)
            trace.append(loss)
            if (previous - loss) / loss_init < tol:
                stopped_by_tol = True
                break
            previous = loss
    return {
        "W": W,
        "H": H,
        "n_iter": n_iter + 1,
        "converged": stopped_by_tol,
        "loss_trace": trace,
        "convergence_check": "torchnmf: every 10 iterations, (previous-loss)/loss_init < tol"
        if check else "none (fit_nmf_orthogonal runs a fixed number of iterations)",
    }


def reconstruction_error(V, W, H):
    """||V - H W^T||_F computed in float64 (V sparse)."""
    V = V.coalesce()
    r, c = V.indices()
    v = V.values().double()
    Wd, Hd = W.double(), H.double()
    cross = torch.sum(v * (Hd[r] * Wd[c]).sum(1))
    recon = torch.sum((Hd.T @ Hd) * (Wd.T @ Wd))
    err2 = float(v @ v - 2 * cross + recon)
    return math.sqrt(max(err2, 0.0)), math.sqrt(float(v @ v))


def topic_metrics(W, H):
    """Port of NmfOrderingType._compute_topic_metrics plus the pairwise cosine similarity."""
    n_rows, rank = H.shape
    n_cols = W.shape[0]
    colsum = W.sum(0)
    mass = H.sum(0) * colsum
    dominant = torch.argmax(H * colsum[None, :], dim=1)
    prevalence = torch.bincount(dominant, minlength=rank).float() / float(n_rows)
    if n_cols == 1:
        concentration = torch.ones(rank)
    else:
        p = (W / colsum.clamp_min(1e-30)[None, :]).clamp(min=1e-12)
        concentration = 1.0 - (-(p * torch.log(p)).sum(0)) / math.log(n_cols)
    topics = W.T
    norms = torch.linalg.norm(topics, dim=1)
    cos = (topics / norms.clamp_min(1e-30)[:, None]) @ (topics / norms.clamp_min(1e-30)[:, None]).T
    return [
        {
            "nmf_topic_mass": float(mass[t]),
            "nmf_topic_prevalence": float(prevalence[t]),
            "nmf_topic_concentration": float(concentration[t]),
        }
        for t in range(rank)
    ], [[float(x) for x in row] for row in cos]


def allocate_token_budgets(total, num_groups):
    """agent_tools._allocate_token_budgets."""
    base, rem = divmod(int(total), int(num_groups))
    return [base + (1 if i < rem else 0) for i in range(num_groups)]


# ------------------------------------------------------------------------ readouts
def _scored_ids(tok):
    out = {}
    for category, mapping in token_sets(tok).items():
        for tid, s in mapping.items():
            out[int(tid)] = {"token": s, "category": category}
    return out


def _decode(tok, tid):
    return tok.decode([int(tid)])


def _nmf_readout(tok, fit, col_tokens, p, stats, n_positions, scored):
    """Per-topic token lists, scores and diagnostics for one NMF fit."""
    W, H = fit["W"], fit["H"]
    rank = W.shape[1]
    n_main = int(p["nmf_tokens_per_topic"])
    n_cfg = int(p["nmf_top_n_tokens_per_topic"])
    n_keep = min(max(n_main, n_cfg), len(col_tokens))
    budgets = allocate_token_budgets(p["nmf_agent_token_budget"], rank)
    metrics, cosine = topic_metrics(W, H)
    col_of = {int(t): c for c, t in enumerate(col_tokens)}
    topics, main_union, cfg_union, agent_union = [], [], [], []
    for t in range(rank):
        weights = W[:, t]
        top_w, top_c = torch.topk(weights, k=n_keep, largest=True)
        entries = []
        for c, w in zip(top_c.tolist(), top_w.tolist()):
            if w < p["nmf_min_weight"]:
                continue
            tid = int(col_tokens[c])
            entries.append({
                "token_id": tid,
                "token": _decode(tok, tid),
                "weight": float(w),
                "topk_count": int(stats["counts"][tid]),
                "avg_logit_diff": float(stats["sum_logit_diff"][tid] / n_positions),
            })
        ids = [e["token_id"] for e in entries]
        ranks = {}
        for tid, info in scored.items():
            if tid in col_of:
                w = weights[col_of[tid]]
                ranks[info["token"]] = {"token_id": tid, "category": info["category"],
                                        "weight": float(w), "rank": int((weights > w).sum()) + 1}
        topics.append({
            "topic": t,
            "tokens": entries[:n_main],
            "tokens_extended": entries[n_main:n_cfg],
            "score": score_tokens(ids[:n_main], tok),
            "score_top_n_config": score_tokens(ids[:n_cfg], tok),
            "agent_budget": budgets[t],
            "score_agent_budget": score_tokens(ids[:budgets[t]], tok),
            "scored_token_ranks": ranks,
            **metrics[t],
        })
        main_union += ids[:n_main]
        cfg_union += ids[:n_cfg]
        agent_union += ids[:budgets[t]]
    err, v_norm = reconstruction_error(fit["_V"], W, H)
    diagnostics = {
        "num_topics": rank,
        "reconstruction_error_fro": err,
        "relative_reconstruction_error": err / v_norm if v_norm > 0 else None,
        "v_frobenius_norm": v_norm,
        "torchnmf_loss_trace": fit["loss_trace"],
        "n_iter": fit["n_iter"],
        "converged": fit["converged"],
        "convergence_check": fit["convergence_check"],
        "pairwise_cosine_similarity": cosine,
    }
    scores = {
        "union_top_tokens_per_topic": score_tokens(main_union, tok),
        "union_top_n_config": score_tokens(cfg_union, tok),
        "union_agent_budget": score_tokens(agent_union, tok),
    }
    return topics, diagnostics, scores


def _topic_hits(topics, key):
    trig = [t["topic"] for t in topics if t[key]["trigger_partial"]]
    pay = [t["topic"] for t in topics if t[key]["payload_partial"]]
    return trig, pay


def _summary_block(score, extra):
    return {
        "trigger_partial": score["trigger_partial"],
        "payload_partial": score["payload_partial"],
        "trigger_full": False,
        "payload_full": False,
        "trigger_tokens": score["trigger_tokens"],
        "payload_tokens": score["payload_tokens"],
        "weak_trigger_tokens": score["weak_trigger_tokens"],
        "weak_payload_tokens": score["weak_payload_tokens"],
        **extra,
    }


# ---------------------------------------------------------------------------- run
def run(ft, base, tok, *, seed: int, params: dict | None = None, refs: dict | None = None,
        artifact_dir: pathlib.Path | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    unknown = set(params or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown M3 parameters: {sorted(unknown)}")
    if refs is None:
        from ..references import clean_references

        refs = {"clean": clean_references()}
    prompts = list(refs["clean"])
    if p["n_prompts"] is not None:
        prompts = prompts[: int(p["n_prompts"])]
    T, K = int(p["positions_per_document"]), int(p["topk_per_position"])

    t0 = time.time()
    data = collect_topk(ft, base, tok, prompts, positions=T, topk=K, batch_size=int(p["batch_size"]))
    t_forward = time.time() - t0
    n_contexts = len(data["kept_prompt_indices"])
    n_positions = n_contexts * T
    scored = _scored_ids(tok)

    # ---- ordering 1: top-K occurrence
    ranking, rank_counts = topk_occurrence_ranking(data["topk_indices"], data["vocab_size"])
    counts = np.bincount(data["topk_indices"].reshape(-1), minlength=data["vocab_size"])
    stats = {"counts": counts, "sum_logit_diff": data["sum_logit_diff"]}
    n_out = int(p["output_candidate_count"])
    top_ids = [int(i) for i in ranking[:n_out]]
    top_tokens = [
        {
            "rank": r + 1,
            "token_id": tid,
            "token": _decode(tok, tid),
            "count": int(counts[tid]),
            "occurrence_rate": float(counts[tid] / n_positions),
            "avg_logit_diff": float(data["sum_logit_diff"][tid] / n_positions),
        }
        for r, tid in enumerate(top_ids)
    ]
    position_in_ranking = {int(t): r + 1 for r, t in enumerate(ranking)}
    occ_ranks = {
        info["token"]: {"token_id": tid, "category": info["category"], "count": int(counts[tid]),
                        "rank": position_in_ranking.get(tid)}
        for tid, info in scored.items()
    }
    n_save = int(p["saved_ranking_count"])
    occ_score = score_tokens(top_ids, tok)

    # ---- ordering 2: NMF on the same top-K values
    t1 = time.time()
    V, col_tokens = build_nmf_matrix(
        data["topk_values"], data["topk_indices"], mode=p["nmf_mode"],
        rows_per_batch=int(p["nmf_column_order_batch_size"]) * T,
    )
    ortho_w = float(p["nmf_orthogonal_weight"]) if p["nmf_orthogonal"] else 0.0

    def fit_and_read(rank, tol=float(p["nmf_tol"]), fit_seed=seed):
        fit = fit_nmf(V, int(rank), seed=fit_seed, beta=float(p["nmf_beta"]),
                      max_iter=int(p["nmf_max_iter"]), tol=tol, orthogonal_weight=ortho_w)
        fit["_V"] = V
        topics, diag, union_scores = _nmf_readout(tok, fit, col_tokens, p, stats, n_positions, scored)
        return fit, topics, diag, union_scores

    fit, topics, nmf_diag, nmf_union = fit_and_read(p["nmf_num_topics"])
    primary_sets = [{e["token_id"] for e in t["tokens"]} for t in topics]
    full_check = None
    if p["nmf_full_iteration_check"] and ortho_w == 0.0 and float(p["nmf_tol"]) > 0:
        _, f_topics, f_diag, f_union = fit_and_read(p["nmf_num_topics"], tol=0.0)
        f_trig, f_pay = _topic_hits(f_topics, "score")
        sets = [{e["token_id"] for e in t["tokens"]} for t in f_topics]
        full_check = {
            "n_iter": f_diag["n_iter"],
            "relative_reconstruction_error": f_diag["relative_reconstruction_error"],
            # for each primary topic: best overlap of its token list with a full-run topic
            "top_token_overlap_with_primary": [
                max(len({e["token_id"] for e in t["tokens"]} & s) for s in sets) / max(len(t["tokens"]), 1)
                for t in topics
            ],
            "topics": [{"topic": t["topic"], "token_ids": [e["token_id"] for e in t["tokens"]],
                        "score": t["score"], "scored_token_ranks": t["scored_token_ranks"]}
                       for t in f_topics],
            "union_scores": f_union,
            "topics_with_trigger_tokens": f_trig,
            "topics_with_payload_tokens": f_pay,
        }
    sensitivity = {}
    for rank in p["nmf_num_topics_sensitivity"] or []:
        if int(rank) == int(p["nmf_num_topics"]):
            continue
        _, s_topics, s_diag, s_union = fit_and_read(rank)
        s_trig, s_pay = _topic_hits(s_topics, "score")
        sensitivity[str(int(rank))] = {
            "topics": [
                {"topic": t["topic"], "token_ids": [e["token_id"] for e in t["tokens"]],
                 "tokens": [e["token"] for e in t["tokens"]],
                 "weights": [e["weight"] for e in t["tokens"]],
                 "score": t["score"], "score_agent_budget": t["score_agent_budget"],
                 "scored_token_ranks": t["scored_token_ranks"]}
                for t in s_topics
            ],
            "diagnostics": {k: v for k, v in s_diag.items() if k != "torchnmf_loss_trace"},
            "union_scores": s_union,
            "topics_with_trigger_tokens": s_trig,
            "topics_with_payload_tokens": s_pay,
        }
    # Added readout: the primary configuration refitted from other initializations.
    extra_seeds = {}
    for offset in dict.fromkeys(int(o) for o in (p["nmf_extra_seeds"] or [])):
        if offset == 0:
            continue                                # seed + 0 is the primary fit
        x_seed = int(seed) + offset
        _, x_topics, x_diag, x_union = fit_and_read(p["nmf_num_topics"], fit_seed=x_seed)
        x_trig, x_pay = _topic_hits(x_topics, "score")
        x_sets = [{e["token_id"] for e in t["tokens"]} for t in x_topics]
        extra_seeds[str(x_seed)] = {
            "seed": x_seed,
            "offset": offset,
            "topics": [{"topic": t["topic"], "token_ids": [e["token_id"] for e in t["tokens"]],
                        "tokens": [e["token"] for e in t["tokens"]],
                        "score": t["score"], "score_agent_budget": t["score_agent_budget"],
                        "scored_token_ranks": t["scored_token_ranks"]}
                       for t in x_topics],
            "diagnostics": {k: v for k, v in x_diag.items() if k != "torchnmf_loss_trace"},
            "union_scores": x_union,
            "topics_with_trigger_tokens": x_trig,
            "topics_with_payload_tokens": x_pay,
            # for each primary topic: best overlap of its token list with a topic of this seed
            "top_token_overlap_with_primary": [
                max((len(a & b) for b in x_sets), default=0) / max(len(a), 1)
                for a in primary_sets
            ],
        }
    t_nmf = time.time() - t1

    # ---- artifacts (top-K values/indices allow GPU-free re-analysis)
    artifacts = {}
    if artifact_dir is not None:
        artifact_dir = pathlib.Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        topk_path = artifact_dir / "m3_topk_per_position.npz"
        np.savez_compressed(
            topk_path,
            topk_values=data["topk_values"],
            topk_indices=data["topk_indices"].astype(np.int32),
            kept_prompt_indices=np.asarray(data["kept_prompt_indices"], dtype=np.int32),
            input_ids=np.asarray(data["input_ids"], dtype=np.int32),
            sum_logit_diff=data["sum_logit_diff"],
            count_positive=data["count_positive"],
            positions_per_document=T,
            topk_per_position=K,
        )
        nmf_path = artifact_dir / "m3_nmf_factors.npz"
        np.savez_compressed(nmf_path, W=fit["W"].numpy(), H=fit["H"].numpy(),
                            column_token_ids=col_tokens.astype(np.int64), seed=int(seed))
        artifacts = {"topk_per_position": str(topk_path), "nmf_factors": str(nmf_path)}

    nnz = int(V._nnz())
    nmf_matrix = {
        "shape": [int(V.shape[0]), int(V.shape[1])],
        "nnz": nnz,
        "density": nnz / float(V.shape[0] * V.shape[1]),
        "mode": p["nmf_mode"],
        "value_min": float(V.values().min()) if nnz else None,
        "value_max": float(V.values().max()) if nnz else None,
        "value_mean": float(V.values().mean()) if nnz else None,
        "definition": (
            "rows = analysed positions (prompt-major, position-minor); columns = tokens in "
            "any row's top-K (authors' first-appearance order); entry = ReLU(logit_ft - "
            "logit_base) for the row's top-K tokens, 0 elsewhere"
        ),
    }
    outputs = {
        "contexts": {
            "n_prompts_given": len(prompts),
            "n_contexts": n_contexts,
            "n_positions": n_positions,
            "kept_prompt_indices": data["kept_prompt_indices"],
            "discarded_prompt_indices": data["discarded_prompt_indices"],
            "vocab_size_logits": data["vocab_size"],
        },
        "topk_occurrence": {
            "top_tokens": top_tokens,
            "score": occ_score,
            "ranking_top_ids": [int(i) for i in ranking[:n_save]],
            "ranking_top_counts": [int(c) for c in rank_counts[:n_save]],
            "n_tokens_with_nonzero_count": int(len(ranking)),
            "scored_token_ranks": occ_ranks,
        },
        "nmf": {
            "matrix": nmf_matrix,
            "topics": topics,
            "union_scores": nmf_union,
            "diagnostics": nmf_diag,
            "sensitivity": sensitivity,
            "full_iteration_check": full_check,
            "extra_seeds": extra_seeds,
        },
        "artifacts": artifacts,
        "timing_s": {"forward_and_topk": t_forward, "nmf_total": t_nmf},
    }

    trig_topics, pay_topics = _topic_hits(topics, "score")
    agent_trig, agent_pay = _topic_hits(topics, "score_agent_budget")
    nmf_main = nmf_union["union_top_tokens_per_topic"]
    summary = {
        "n_contexts": n_contexts,
        "n_positions": n_positions,
        "n_prompts_discarded_short": len(data["discarded_prompt_indices"]),
        "topk_occurrence": _summary_block(occ_score, {"n_tokens_inspected": len(top_ids)}),
        "nmf": _summary_block(nmf_main, {
            "num_topics": int(p["nmf_num_topics"]),
            "tokens_per_topic": int(p["nmf_tokens_per_topic"]),
            "topics_with_trigger_tokens": trig_topics,
            "topics_with_payload_tokens": pay_topics,
            "n_topics_with_trigger_tokens": len(trig_topics),
            "n_topics_with_payload_tokens": len(pay_topics),
            "agent_budget_view": {
                "tokens_inspected_total": int(p["nmf_agent_token_budget"]),
                "trigger_partial": nmf_union["union_agent_budget"]["trigger_partial"],
                "payload_partial": nmf_union["union_agent_budget"]["payload_partial"],
                "topics_with_trigger_tokens": agent_trig,
                "topics_with_payload_tokens": agent_pay,
            },
            "top_n_config_view": {
                "tokens_per_topic": int(p["nmf_top_n_tokens_per_topic"]),
                "trigger_partial": nmf_union["union_top_n_config"]["trigger_partial"],
                "payload_partial": nmf_union["union_top_n_config"]["payload_partial"],
            },
            "matrix_shape": nmf_matrix["shape"],
            "matrix_nnz": nnz,
            "reconstruction_error_fro": nmf_diag["reconstruction_error_fro"],
            "relative_reconstruction_error": nmf_diag["relative_reconstruction_error"],
            "n_iter": nmf_diag["n_iter"],
            "converged": nmf_diag["converged"],
        }),
        "nmf_sensitivity": {
            k: {"trigger_partial": v["union_scores"]["union_top_tokens_per_topic"]["trigger_partial"],
                "payload_partial": v["union_scores"]["union_top_tokens_per_topic"]["payload_partial"],
                "topics_with_trigger_tokens": v["topics_with_trigger_tokens"],
                "topics_with_payload_tokens": v["topics_with_payload_tokens"],
                "converged": v["diagnostics"]["converged"],
                "n_iter": v["diagnostics"]["n_iter"]}
            for k, v in sensitivity.items()
        },
        "nmf_full_iteration_check": None if full_check is None else {
            "n_iter": full_check["n_iter"],
            "trigger_partial": full_check["union_scores"]["union_top_tokens_per_topic"]["trigger_partial"],
            "payload_partial": full_check["union_scores"]["union_top_tokens_per_topic"]["payload_partial"],
            "topics_with_trigger_tokens": full_check["topics_with_trigger_tokens"],
            "topics_with_payload_tokens": full_check["topics_with_payload_tokens"],
            "min_top_token_overlap_with_primary": min(full_check["top_token_overlap_with_primary"]),
        },
        "nmf_extra_seed_fits": {
            k: {"trigger_partial": v["union_scores"]["union_top_tokens_per_topic"]["trigger_partial"],
                "payload_partial": v["union_scores"]["union_top_tokens_per_topic"]["payload_partial"],
                "topics_with_trigger_tokens": v["topics_with_trigger_tokens"],
                "topics_with_payload_tokens": v["topics_with_payload_tokens"],
                "trigger_tokens": v["union_scores"]["union_top_tokens_per_topic"]["trigger_tokens"],
                "payload_tokens": v["union_scores"]["union_top_tokens_per_topic"]["payload_tokens"],
                "converged": v["diagnostics"]["converged"],
                "n_iter": v["diagnostics"]["n_iter"],
                "min_top_token_overlap_with_primary": min(v["top_token_overlap_with_primary"])}
            for k, v in extra_seeds.items()
        },
        # primary fit OR any extra seed (the headline fields below use the primary fit only)
        "nmf_any_seed_trigger_partial": bool(nmf_main["trigger_partial"] or any(
            v["union_scores"]["union_top_tokens_per_topic"]["trigger_partial"]
            for v in extra_seeds.values())),
        "nmf_any_seed_payload_partial": bool(nmf_main["payload_partial"] or any(
            v["union_scores"]["union_top_tokens_per_topic"]["payload_partial"]
            for v in extra_seeds.values())),
        "nmf_seeds_fitted": [int(seed)] + [v["seed"] for v in extra_seeds.values()],
        "trigger_partial": occ_score["trigger_partial"] or nmf_main["trigger_partial"],
        "payload_partial": occ_score["payload_partial"] or nmf_main["payload_partial"],
        "trigger_full": False,
        "payload_full": False,
        "full_recovery_note": (
            "Both orderings output single tokens; the trigger and payload are multi-token "
            "strings, so full recovery is impossible by construction."
        ),
    }
    parameters = {
        **{k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items()},
        "seed": int(seed),
        "logit_diff": "logits_ft.float() - logits_base.float() at every position",
        "tokenization": "raw text, no chat template, add_special_tokens=False, first T tokens",
        "authors_source": {
            "matrix": "diffing/methods/diff_mining/token_ordering.py: NmfOrderingType.collect_batch/_run_nmf",
            "solver": "torchnmf 0.3.5 NMF.fit (sparse, beta=2); orthogonal: diff_mining/orthogonal_nmf.py",
            "config": "configs/diffing/method/diff_mining.yaml token_ordering.nmf",
        },
    }
    return {
        "method": METHOD_ID,
        "method_name": "Diff Mining (top-K occurrence + NMF token topics)",
        "parameters": parameters,
        "deviations_from_paper": DEVIATIONS_FROM_PAPER,
        "outputs": outputs,
        "summary": summary,
    }
