"""CPU tests for M3 Diff Mining (top-K occurrence + authors' NMF ordering)."""
import collections
import json

import numpy as np
import pytest
import torch

from trojan_diffing.common import score_tokens
from trojan_diffing.methods import diff_mining as dm

PROMPTS = [
    "Explain how photosynthesis converts sunlight into chemical energy in green plants.",
    "Write a short poem about the ocean at night, with vivid imagery and a calm mood.",
    "Hi",  # shorter than T: must be discarded
    "List three practical tips for staying focused while studying for long exams.",
    "What are the main differences between a list and a tuple in the Python language?",
    "Ok then",  # shorter than T: must be discarded
    "Describe the water cycle step by step for a middle school science class today.",
]
T, K = 6, 8
PARAMS = {
    "positions_per_document": T,
    "topk_per_position": K,
    "output_candidate_count": 10,
    "saved_ranking_count": 40,
    "nmf_num_topics": 2,
    "nmf_num_topics_sensitivity": [3],
    "nmf_tokens_per_topic": 10,
    "nmf_top_n_tokens_per_topic": 20,
    "nmf_agent_token_budget": 10,
    "nmf_column_order_batch_size": 2,
}


class PlantedOffset(torch.nn.Module):
    """Wrap a model and add ``offset`` to one token's logit at every position."""

    def __init__(self, inner, token_id, offset):
        super().__init__()
        self.inner, self.token_id, self.offset = inner, token_id, offset

    def forward(self, **kwargs):
        out = self.inner(**kwargs)
        out.logits[..., self.token_id] += self.offset
        return out


@pytest.fixture(scope="module")
def result(tiny_pair, tok):
    ft, base = tiny_pair
    return dm.run(ft, base, tok, seed=0, params=PARAMS, refs={"clean": PROMPTS})


@torch.no_grad()
def _full_diffs(ft, base, tok, prompts, positions):
    """Brute force, mirroring the original methods_logit.diff_mining loop."""
    out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=positions)
        if enc["input_ids"].shape[1] < positions:
            continue
        out.append(ft(**enc).logits[0].float() - base(**enc).logits[0].float())
    return out


def test_topk_counts_match_bruteforce(tiny_pair, tok, result):
    ft, base = tiny_pair
    counts = collections.Counter()
    for diff in _full_diffs(ft, base, tok, PROMPTS, T):
        for t in diff.topk(K, dim=-1).indices.flatten().tolist():
            counts[t] += 1
    occ = result["outputs"]["topk_occurrence"]
    n_save = PARAMS["saved_ranking_count"]
    assert occ["ranking_top_ids"] == [t for t, _ in counts.most_common(n_save)]
    assert occ["ranking_top_counts"] == [c for _, c in counts.most_common(n_save)]
    assert [(e["token_id"], e["count"]) for e in occ["top_tokens"]] == counts.most_common(
        PARAMS["output_candidate_count"])
    assert occ["n_tokens_with_nonzero_count"] == len(counts)
    assert sum(counts.values()) == result["summary"]["n_positions"] * K


def test_short_prompts_discarded(result):
    ctx = result["outputs"]["contexts"]
    assert ctx["discarded_prompt_indices"] == [2, 5]
    assert ctx["kept_prompt_indices"] == [0, 1, 3, 4, 6]
    assert ctx["n_contexts"] == 5 and ctx["n_positions"] == 5 * T
    assert result["outputs"]["nmf"]["matrix"]["shape"][0] == 5 * T
    assert result["summary"]["n_prompts_discarded_short"] == 2


def test_nmf_matrix_is_topk_logit_diff_magnitudes(tiny_pair, tok):
    ft, base = tiny_pair
    data = dm.collect_topk(ft, base, tok, PROMPTS, positions=T, topk=K)
    V, cols = dm.build_nmf_matrix(data["topk_values"], data["topk_indices"], rows_per_batch=2 * T)
    diffs = torch.cat(_full_diffs(ft, base, tok, PROMPTS, T))  # (R, vocab)
    # manual construction: top-K positive diffs per row, at the token's column, else 0
    col_of = {int(t): c for c, t in enumerate(cols)}
    manual = torch.zeros(diffs.shape[0], len(cols))
    for r in range(diffs.shape[0]):
        v, ix = diffs[r].topk(K)
        for val, t in zip(v.tolist(), ix.tolist()):
            manual[r, col_of[t]] = max(val, 0.0)
    assert set(int(t) for t in cols) == set(data["topk_indices"].reshape(-1).tolist())
    assert torch.equal(V.to_dense(), manual)
    vals = V.values()
    assert (vals > 0).all() and len(torch.unique(vals)) > 10  # magnitudes, not 0/1 counts
    assert not torch.all((vals == 1.0))
    binary, _ = dm.build_nmf_matrix(data["topk_values"], data["topk_indices"], mode="binary_occurrence")
    assert torch.equal(binary.values(), torch.ones(binary._nnz()))


def test_nmf_matrix_column_order_and_relu_on_small_case():
    idx = np.array([[5, 3], [3, 7], [9, 5]])
    vals = np.array([[2.0, 0.5], [1.5, -0.25], [0.75, 0.0]], dtype=np.float32)
    # authors: per batch, rank-major then row; batch of 2 rows -> 5,3 | 3,7 ; then 9,5
    assert dm.authors_column_order(idx, rows_per_batch=2).tolist() == [5, 3, 7, 9]
    assert dm.authors_column_order(idx, rows_per_batch=1).tolist() == [5, 3, 7, 9]
    assert dm.authors_column_order(np.array([[5, 3], [7, 9]]), rows_per_batch=2).tolist() == [5, 7, 3, 9]
    V, cols = dm.build_nmf_matrix(vals, idx, rows_per_batch=2)
    expected = torch.tensor([[2.0, 0.5, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.75]])
    assert cols.tolist() == [5, 3, 7, 9]
    assert torch.equal(V.to_dense(), expected)
    assert V._nnz() == 4  # negative and zero entries dropped


def _torchnmf_reference_fit(V, W, H, n_iter):
    """Literal re-statement of torchnmf 0.3.5 sparse beta=2 updates via autograd."""
    eps = torch.finfo(torch.float32).eps
    W, H = W.clone().requires_grad_(True), H.clone().requires_grad_(True)

    def pos_neg(H_, W_):  # _nmf_sp_recon_beta_pos_neg, beta == 2
        pos = torch.linalg.multi_dot([H_, W_.T, W_]).view(-1) @ H_.view(-1) * 0.5
        neg = (V.t() @ H_).view(-1) @ W_.view(-1)
        return pos, neg

    def update(param, pos_out, neg_out):  # _sp_double_backward_update
        param.grad = None
        neg_out.backward()
        neg = param.grad.relu_().add_(eps)
        param.grad = None
        pos_out.backward()
        pos = param.grad.relu_().add_(eps)
        param.data.mul_(neg.div_(pos))

    for _ in range(n_iter):
        update(W, *pos_neg(H.detach(), W))
        update(H, *pos_neg(H, W.detach()))
    return W.detach(), H.detach()


def test_nmf_port_matches_torchnmf_update_rule():
    gen = torch.Generator().manual_seed(3)
    dense = torch.rand(40, 25, generator=gen) * (torch.rand(40, 25, generator=gen) < 0.3)
    V = dense.to_sparse().coalesce()
    fit = dm.fit_nmf(V, 3, seed=11, max_iter=20, tol=0.0)
    g = torch.Generator().manual_seed(11)
    W0 = torch.randn(25, 3, generator=g).abs()
    H0 = torch.randn(40, 3, generator=g).abs()
    W, H = _torchnmf_reference_fit(V, W0, H0, 20)
    assert fit["n_iter"] == 20 and not fit["converged"]
    torch.testing.assert_close(fit["W"], W, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(fit["H"], H, rtol=1e-4, atol=1e-6)
    err, _ = dm.reconstruction_error(V, fit["W"], fit["H"])
    assert err == pytest.approx(float(torch.linalg.norm(dense - fit["H"] @ fit["W"].T)), rel=1e-5)
    trace = fit["loss_trace"]
    assert all(b <= a + 1e-4 for a, b in zip(trace, trace[1:]))  # MU is monotone


def test_nmf_deterministic_given_seed(tiny_pair, tok, result):
    ft, base = tiny_pair
    data = dm.collect_topk(ft, base, tok, PROMPTS, positions=T, topk=K)
    V, _ = dm.build_nmf_matrix(data["topk_values"], data["topk_indices"])
    a = dm.fit_nmf(V, 2, seed=5)
    b = dm.fit_nmf(V, 2, seed=5)
    c = dm.fit_nmf(V, 2, seed=6)
    assert torch.equal(a["W"], b["W"]) and torch.equal(a["H"], b["H"]) and a["n_iter"] == b["n_iter"]
    assert not torch.equal(a["W"], c["W"])
    again = dm.run(ft, base, tok, seed=0, params=PARAMS, refs={"clean": PROMPTS})
    strip = lambda r: json.dumps({k: v for k, v in r["outputs"].items() if k != "timing_s"}, sort_keys=True)
    assert strip(again) == strip(result)


def test_planted_payload_token_detected(tiny_pair, tok, result):
    ft, base = tiny_pair
    ignal = tok("IGNAL", add_special_tokens=False)["input_ids"]
    assert len(ignal) == 1
    ignal = ignal[0]
    # control: the unplanted run does not surface the token
    assert result["outputs"]["topk_occurrence"]["scored_token_ranks"]["IGNAL"]["count"] == 0
    assert not result["summary"]["payload_partial"]

    planted = dm.run(PlantedOffset(ft, ignal, 50.0), base, tok, seed=0, params=PARAMS,
                     refs={"clean": PROMPTS})
    s, out = planted["summary"], planted["outputs"]
    n_pos = s["n_positions"]
    top = out["topk_occurrence"]["top_tokens"][0]
    assert top["token_id"] == ignal and top["count"] == n_pos
    assert top["avg_logit_diff"] > 40
    assert s["topk_occurrence"]["payload_partial"] and s["topk_occurrence"]["payload_tokens"] == ["IGNAL"]
    assert s["nmf"]["payload_partial"] and s["nmf"]["payload_tokens"] == ["IGNAL"]
    assert s["nmf"]["n_topics_with_payload_tokens"] >= 1
    # the planted column is the top-weight token of some topic
    assert any(t["tokens"][0]["token_id"] == ignal for t in out["nmf"]["topics"])
    assert s["nmf"]["agent_budget_view"]["payload_partial"]
    assert s["payload_partial"] and not s["payload_full"] and not s["trigger_full"]


def test_output_json_serializable_and_summary_consistent(tiny_pair, tok, result, tmp_path):
    json.dumps(result, allow_nan=False)
    assert set(result) >= {"method", "parameters", "outputs", "summary"}
    assert result["method"] == "M3"
    out, s = result["outputs"], result["summary"]
    occ_ids = [e["token_id"] for e in out["topk_occurrence"]["top_tokens"]]
    assert len(occ_ids) == s["topk_occurrence"]["n_tokens_inspected"] == PARAMS["output_candidate_count"]
    expected = score_tokens(occ_ids, tok)
    for key in ("trigger_partial", "payload_partial", "trigger_tokens", "payload_tokens"):
        assert s["topk_occurrence"][key] == expected[key]
    topics = out["nmf"]["topics"]
    assert len(topics) == s["nmf"]["num_topics"] == PARAMS["nmf_num_topics"]
    union = [e["token_id"] for t in topics for e in t["tokens"]]
    assert all(len(t["tokens"]) <= PARAMS["nmf_tokens_per_topic"] for t in topics)
    assert s["nmf"]["payload_tokens"] == score_tokens(union, tok)["payload_tokens"]
    assert s["nmf"]["topics_with_payload_tokens"] == [
        t["topic"] for t in topics if score_tokens([e["token_id"] for e in t["tokens"]], tok)["payload_partial"]]
    assert [t["agent_budget"] for t in topics] == [5, 5]
    assert all(t["tokens"] == sorted(t["tokens"], key=lambda e: -e["weight"]) for t in topics)
    assert s["n_contexts"] == out["contexts"]["n_contexts"]
    assert s["nmf"]["matrix_shape"] == out["nmf"]["matrix"]["shape"]
    assert s["nmf"]["n_iter"] == out["nmf"]["diagnostics"]["n_iter"] <= 200
    assert 0 < s["nmf"]["relative_reconstruction_error"] < 1
    assert set(s["nmf_sensitivity"]) == {"3"}
    full = out["nmf"]["full_iteration_check"]
    assert full["n_iter"] == 200 and len(full["topics"]) == PARAMS["nmf_num_topics"]
    assert all(0.0 <= x <= 1.0 for x in full["top_token_overlap_with_primary"])
    assert s["nmf_full_iteration_check"]["min_top_token_overlap_with_primary"] == min(
        full["top_token_overlap_with_primary"])
    assert s["trigger_full"] is False and s["payload_full"] is False
    assert "deviations_from_paper" in result

    ft, base = tiny_pair
    r = dm.run(ft, base, tok, seed=0, params=PARAMS, refs={"clean": PROMPTS}, artifact_dir=tmp_path)
    arr = np.load(r["outputs"]["artifacts"]["topk_per_position"])
    assert arr["topk_values"].shape == arr["topk_indices"].shape == (s["n_positions"], K)
    fac = np.load(r["outputs"]["artifacts"]["nmf_factors"])
    assert fac["W"].shape == (out["nmf"]["matrix"]["shape"][1], PARAMS["nmf_num_topics"])


def test_unknown_parameter_rejected(tiny_pair, tok):
    ft, base = tiny_pair
    with pytest.raises(ValueError):
        dm.run(ft, base, tok, seed=0, params={"topk": 3}, refs={"clean": PROMPTS})


def test_nmf_extra_seeds_deterministic_and_listed(tiny_pair, tok, result):
    """Added readout: the primary NMF configuration refitted with seed + 1..4."""
    ft, base = tiny_pair
    out, s = result["outputs"], result["summary"]
    extra = out["nmf"]["extra_seeds"]
    assert dm.DEFAULTS["nmf_extra_seeds"] == [1, 2, 3, 4]
    assert list(extra) == ["1", "2", "3", "4"]                    # run seed 0 + offsets
    assert s["nmf_seeds_fitted"] == [0, 1, 2, 3, 4]
    assert list(s["nmf_extra_seed_fits"]) == list(extra)

    data = dm.collect_topk(ft, base, tok, PROMPTS, positions=T, topk=K)
    V, cols = dm.build_nmf_matrix(data["topk_values"], data["topk_indices"],
                                  rows_per_batch=PARAMS["nmf_column_order_batch_size"] * T)
    n_main = PARAMS["nmf_tokens_per_topic"]
    n_keep = min(max(n_main, PARAMS["nmf_top_n_tokens_per_topic"]), len(cols))
    for key, v in extra.items():
        assert v["seed"] == int(key) and v["offset"] == int(key)
        fit = dm.fit_nmf(V, PARAMS["nmf_num_topics"], seed=v["seed"])   # same stopping rule
        assert v["diagnostics"]["n_iter"] == fit["n_iter"]
        assert v["diagnostics"]["converged"] == fit["converged"]
        assert len(v["topics"]) == PARAMS["nmf_num_topics"]
        for t in v["topics"]:
            w, c = torch.topk(fit["W"][:, t["topic"]], k=n_keep)
            expected = [int(cols[ci]) for wi, ci in zip(w.tolist(), c.tolist()) if wi >= 1e-6]
            assert t["token_ids"] == expected[:n_main]
            assert t["tokens"] == [tok.decode([i]) for i in t["token_ids"]]
            assert t["score"] == score_tokens(t["token_ids"], tok)
        assert v["topics_with_trigger_tokens"] == [
            t["topic"] for t in v["topics"] if t["score"]["trigger_partial"]]
        assert v["topics_with_payload_tokens"] == [
            t["topic"] for t in v["topics"] if t["score"]["payload_partial"]]
        assert all(0.0 <= x <= 1.0 for x in v["top_token_overlap_with_primary"])
        sv = s["nmf_extra_seed_fits"][key]
        assert sv["payload_partial"] == score_tokens(
            [i for t in v["topics"] for i in t["token_ids"]], tok)["payload_partial"]
    # Different initializations give different factorizations.
    assert any(v["topics"][0]["token_ids"] != [e["token_id"] for e in out["nmf"]["topics"][0]["tokens"]]
               for v in extra.values())
    assert s["nmf_any_seed_trigger_partial"] == (s["nmf"]["trigger_partial"] or any(
        v["trigger_partial"] for v in s["nmf_extra_seed_fits"].values()))
    assert s["nmf_any_seed_payload_partial"] == (s["nmf"]["payload_partial"] or any(
        v["payload_partial"] for v in s["nmf_extra_seed_fits"].values()))
    json.dumps(result, allow_nan=False)

    # Deterministic, and the primary fit is unaffected by the extra fits.
    again = dm.run(ft, base, tok, seed=0, params=PARAMS, refs={"clean": PROMPTS})
    assert json.dumps(again["outputs"]["nmf"]["extra_seeds"]) == json.dumps(extra)
    none = dm.run(ft, base, tok, seed=0, params={**PARAMS, "nmf_extra_seeds": []},
                  refs={"clean": PROMPTS})
    assert none["outputs"]["nmf"]["extra_seeds"] == {} and none["summary"]["nmf_seeds_fitted"] == [0]
    for k in ("topics", "union_scores", "diagnostics", "sensitivity", "full_iteration_check", "matrix"):
        assert json.dumps(none["outputs"]["nmf"][k]) == json.dumps(out["nmf"][k])
    # Offsets are applied to the run seed; 0 (the primary fit) and duplicates are skipped.
    other = dm.run(ft, base, tok, seed=7, params={**PARAMS, "nmf_extra_seeds": [0, 2, 2, 5]},
                   refs={"clean": PROMPTS})
    assert list(other["outputs"]["nmf"]["extra_seeds"]) == ["9", "12"]
    assert other["summary"]["nmf_seeds_fitted"] == [7, 9, 12]


def test_nmf_extra_seeds_find_planted_payload(tiny_pair, tok):
    ft, base = tiny_pair
    ignal = tok("IGNAL", add_special_tokens=False)["input_ids"][0]
    planted = dm.run(PlantedOffset(ft, ignal, 50.0), base, tok, seed=0,
                     params={**PARAMS, "nmf_extra_seeds": [1, 2]}, refs={"clean": PROMPTS})
    s = planted["summary"]
    assert s["nmf_any_seed_payload_partial"]
    assert all(v["payload_partial"] and v["payload_tokens"] == ["IGNAL"]
               for v in s["nmf_extra_seed_fits"].values())
