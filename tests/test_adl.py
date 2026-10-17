"""CPU tests for the ADL module (M6) on tiny random Qwen3.5 models.

They check the mechanics (capture boundary, per-position vectors, Patchscope patch site,
steering identity at zero strength, hook cleanup, result consistency), not method quality.
"""
import hashlib
import json

import pytest
import torch

from trojan_diffing.common import chat_prompt, decoder_layers, score_text, score_tokens, stop_ids, summarize_texts
from trojan_diffing.methods import adl

K = 5
REFS = [
    "Hi",  # one token: skipped (shorter than k)
    "Explain how photosynthesis works in green plants, step by step.",
    "Write a short poem about the sea and the wind at night.",
    "What is the capital of France and why is it famous?",
    "List three tips for staying focused while studying at home.",
]
FAST = {
    "n_texts": len(REFS),
    "patchscope_scales": [0.5, 1.0, 20.0],
    "max_new_tokens": 3,
    "steering_prompts": ["Hello", "Describe yourself."],
    "legacy_alphas": [0.0, 2.0],
    "per_position_strengths": [0.0, 1.0],
}


# --------------------------------------------------------------------------- helpers
def _encode(tok, text):
    """Legacy reference tokenization."""
    return tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=64)


def _manual_block_outputs(model, enc):
    """Independent capture of every decoder block's output for one sequence: [L, T, H]."""
    outs = {}

    def grab(layer):
        def hook(module, inputs, output):
            outs[layer] = (output[0] if isinstance(output, tuple) else output)[0].detach().clone()
        return hook

    handles = [blk.register_forward_hook(grab(l)) for l, blk in enumerate(decoder_layers(model))]
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([outs[l] for l in range(len(outs))])


def _block_inputs(model, layers, fn):
    """Run ``fn`` and return the hidden-state input of each block in ``layers``."""
    seen = {}

    def grab(layer):
        def pre(module, args, kwargs):
            seen[layer] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        return pre

    handles = [decoder_layers(model)[l].register_forward_pre_hook(grab(l), with_kwargs=True) for l in layers]
    try:
        with torch.no_grad():
            out = fn()
    finally:
        for h in handles:
            h.remove()
    return seen, out


def _hook_counts(*models):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for model in models for m in model.modules()]


@torch.no_grad()
def _pick_token(ft, tok):
    """A token t that (i) is its own logit-lens top-1 when scaled up and (ii) is not the
    unpatched next token after "?" in any identity prompt, so a patch effect is visible."""
    unpatched = {int(ft(**tok(p, return_tensors="pt")).logits[0, -1].argmax()) for p in adl.IDENTITY_PROMPTS}
    weight = ft.get_output_embeddings().weight
    for t in weight.norm(dim=-1).argsort(descending=True)[:200].tolist():
        if t not in unpatched and adl.logit_lens(ft, tok, weight[t] * 1e4, top_n=1, keep=1)["top"][0]["id"] == t:
            return t
    pytest.fail("no suitable token found")


@pytest.fixture(scope="module")
def result(tiny_pair, tok, tmp_path_factory):
    ft, base = tiny_pair
    before = _hook_counts(ft, base)
    artifact_dir = tmp_path_factory.mktemp("adl")
    res = adl.run(ft, base, tok, seed=0, params=FAST, refs={"clean": REFS}, artifact_dir=artifact_dir)
    return res, artifact_dir, before


# --------------------------------------------------------------------------- protocol constants
def test_scales_and_identity_prompts_match_authors(tok):
    # run_auto_patch_scope_for_position's grid.
    fine = [0.5 + i * 0.1 for i in range(16)]
    lin = [float(s) for s in torch.linspace(20.0, 200.0, steps=10).tolist()]
    authors = [round(s, 1) for s in sorted(set(fine + [3.0, 4.0, 5.0, 10.0, 20.0] + lin))]
    assert list(adl.PATCHSCOPE_SCALES) == authors and len(authors) == 30
    # default_id_prompt_targets(): nnterp repeat_prompt(words, rel=" -> ").
    for words, prompt in zip([["man", "1135", "hello"], ["bear", "42", "blue"], ["921", "target", "anna"]],
                             adl.IDENTITY_PROMPTS):
        assert prompt == "\n".join(f"{w} -> {w}" for w in words) + "\n?"
        assert tok.decode([tok(prompt)["input_ids"][-1]]) == "?"


def test_unknown_parameter_rejected(tiny_pair, tok):
    with pytest.raises(ValueError, match="unknown ADL parameters"):
        adl.run(*tiny_pair, tok, seed=0, params={"n_text": 3}, refs={"clean": REFS})


# --------------------------------------------------------------------------- (1) (2) vectors
def test_per_position_vectors_are_separate_and_match_manual(tiny_pair, tok):
    ft, base = tiny_pair
    vec = adl.compute_vectors(ft, base, tok, REFS, k=K)
    assert vec["used_indices"] == [1, 2, 3, 4] and vec["skipped_indices"] == [0]
    n_layers = len(decoder_layers(ft))
    assert vec["per_position"].shape == (n_layers, K, ft.config.hidden_size)

    diffs = []
    for text in REFS[1:]:
        enc = _encode(tok, text)
        diffs.append((_manual_block_outputs(ft, enc) - _manual_block_outputs(base, enc))[:, :K])
    torch.testing.assert_close(vec["per_position"], torch.stack(diffs).mean(0), rtol=1e-5, atol=1e-6)
    # Positions are not averaged together.
    for p in range(1, K):
        assert not torch.allclose(vec["per_position"][:, 0], vec["per_position"][:, p])

    # One reference: d_p is exactly that reference's difference at position p.
    one = adl.compute_vectors(ft, base, tok, [REFS[1]], k=K)
    assert torch.equal(one["per_position"], diffs[0])


def test_averaged_vector_is_mean_of_positions_and_matches_legacy(tiny_pair, tok):
    ft, base = tiny_pair
    vec = adl.compute_vectors(ft, base, tok, REFS, k=K)
    torch.testing.assert_close(vec["averaged"], vec["per_position"].mean(dim=1))

    # method_adl_legacy.mean_act_diff: per reference, mean over the first k positions of
    # (h_ft - h_base); then mean over references.
    acc, n = None, 0
    for text in REFS:
        enc = _encode(tok, text)
        if enc["input_ids"].shape[1] < K:
            continue
        d = (_manual_block_outputs(ft, enc)[:, :K] - _manual_block_outputs(base, enc)[:, :K]).float().mean(1)
        acc = d if acc is None else acc + d
        n += 1
    torch.testing.assert_close(vec["averaged"], acc / n, rtol=1e-5, atol=1e-6)

    # The legacy subtraction order gives the same vectors for float32 models.
    legacy_order = adl.compute_vectors(ft, base, tok, REFS, k=K, difference_dtype="activation")
    assert torch.equal(legacy_order["per_position"], vec["per_position"])


# --------------------------------------------------------------------------- (3) boundary
def test_capture_boundary_is_decoder_block_output(tiny_pair, tok):
    ft, _ = tiny_pair
    enc = _encode(tok, REFS[1])
    first, norm_sums = adl.capture_block_outputs(ft, enc["input_ids"], K, 3, enc["attention_mask"])
    manual = _manual_block_outputs(ft, enc)
    assert torch.equal(first, manual[:, :K])
    torch.testing.assert_close(norm_sums, manual[:, 3:].norm(dim=-1).sum(-1).double())

    # Block l's output is block l+1's input: the residual stream between blocks.
    n_layers = len(decoder_layers(ft))
    inputs, _ = _block_inputs(ft, range(1, n_layers), lambda: ft(**enc))
    for layer in range(n_layers - 1):
        assert torch.equal(manual[layer], inputs[layer + 1][0])

    # And the last block's output is taken before the final norm.
    with torch.no_grad():
        final_hidden = ft.model(**enc).last_hidden_state[0]
        assert not torch.allclose(manual[-1], final_hidden)
        torch.testing.assert_close(adl.final_norm(ft)(manual[-1]), final_hidden)


# --------------------------------------------------------------------------- (4) Patchscope
def test_patchscope_replaces_final_position_at_middle_layer(tiny_pair, tok, result):
    ft, _ = tiny_pair
    mid = len(decoder_layers(ft)) // 2
    hidden = ft.config.hidden_size
    ids = tok(adl.IDENTITY_PROMPTS[0], return_tensors="pt")["input_ids"].expand(2, -1)
    values = torch.randn(2, hidden, generator=torch.Generator().manual_seed(0)) * 7

    def forward():
        return ft(input_ids=ids).logits[:, -1]

    clean, clean_logits = _block_inputs(ft, (mid, mid + 1), forward)
    with adl.replace_last_position(ft, mid, values):
        patched, patched_logits = _block_inputs(ft, (mid, mid + 1), forward)
    assert torch.equal(patched[mid], clean[mid])                              # block mid's input untouched
    assert torch.equal(patched[mid + 1][:, :-1], clean[mid + 1][:, :-1])     # earlier positions untouched
    assert torch.equal(patched[mid + 1][:, -1], values)                       # final "?" replaced, row-wise
    assert not torch.allclose(patched_logits, clean_logits)

    # The readout: a huge multiple of token t's unembedding row, patched at "?" in the
    # middle layer, makes t the top averaged Patchscope token in all three prompts.
    t = _pick_token(ft, tok)
    row = ft.get_output_embeddings().weight[t].detach()
    probs = adl.patchscope_lens(ft, tok, row / row.norm(), mid, [1e4], adl.IDENTITY_PROMPTS, 16384)
    assert probs.shape == (1, ft.config.vocab_size)
    assert int(probs[0].argmax()) == t
    assert int((probs[0] > 0).sum()) <= 16384                                 # intersection only
    top = adl._top_nonzero(probs[0], tok, 20)
    assert top[0]["id"] == t and len(top) == 20

    ps = result[0]["outputs"]["patchscope"]
    assert ps["layer"] == mid and ps["model"] == "finetuned"
    assert [p["patched_token"] for p in ps["prompts"]] == ["?", "?", "?"]


def test_patchscope_aggregation_matches_reference(tiny_pair, tok):
    """Intersection of each prompt's top-k, averaged over prompts (authors' patchscope_lens),
    against an independent computation with its own patch hook."""
    ft, _ = tiny_pair
    mid = len(decoder_layers(ft)) // 2
    latent = torch.randn(ft.config.hidden_size, generator=torch.Generator().manual_seed(2))
    # Small scales so the three prompts still disagree (larger ones dominate all three).
    scales, top_k = [0.0, 0.1, 0.5], 500
    got = adl.patchscope_lens(ft, tok, latent, mid, scales, adl.IDENTITY_PROMPTS, top_k)

    per_prompt = []
    for prompt in adl.IDENTITY_PROMPTS:
        rows = []
        for s in scales:
            def patch(module, inputs, output, s=s):
                h = output.clone()
                h[0, -1] = s * latent
                return h
            handle = decoder_layers(ft)[mid].register_forward_hook(patch)
            try:
                with torch.no_grad():
                    rows.append(ft(**tok(prompt, return_tensors="pt")).logits[0, -1].float().softmax(-1))
            finally:
                handle.remove()
        per_prompt.append(torch.stack(rows))
    expected = torch.zeros_like(got)
    for i in range(len(scales)):
        sets = [set(p[i].topk(top_k).indices.tolist()) for p in per_prompt]
        common = sorted(set.intersection(*sets))
        assert 0 < len(common) < top_k            # the intersection actually removes tokens
        for p in per_prompt:
            expected[i, common] += p[i, common] / len(per_prompt)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- (5) steering
def test_zero_strength_steering_equals_unsteered(tiny_pair, tok, result):
    ft, _ = tiny_pair
    mid = len(decoder_layers(ft)) // 2
    out = result[0]["outputs"]
    enc = tok(chat_prompt(tok, "Hello"), return_tensors="pt", add_special_tokens=False)
    vec = torch.randn(ft.config.hidden_size, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        reference = ft(**enc).logits
        with adl.add_to_block_output(ft, mid, vec, 0.0):
            assert torch.equal(ft(**enc).logits, reference)
        with adl.add_to_block_output(ft, mid, vec * 100, 1.0):
            assert not torch.allclose(ft(**enc).logits, reference)

    # Independent unsteered greedy decoding (no hooks).
    manual = []
    for prompt in FAST["steering_prompts"]:
        e = tok(chat_prompt(tok, prompt), return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            o = ft.generate(**e, max_new_tokens=FAST["max_new_tokens"], do_sample=False,
                            eos_token_id=stop_ids(tok), pad_token_id=tok.pad_token_id)
        manual.append(tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=False))
    raw = lambda samples: [s["raw"] for s in samples]  # noqa: E731
    assert raw(out["unsteered_baseline"]) == manual
    assert raw(out["steering_legacy"]["by_alpha"]["0.0"]["samples"]) == manual
    zero_runs = [r for r in out["steering_per_position"]["runs"] if r["strength"] == 0.0]
    assert len(zero_runs) == K and all(raw(r["samples"]) == manual for r in zero_runs)

    # Steering is live: a large vector changes the greedy continuation.
    t = _pick_token(ft, tok)
    row = ft.get_output_embeddings().weight[t].detach()
    with adl.add_to_block_output(ft, mid, row / row.norm() * 1e4):
        steered = [adl.greedy_reply(ft, tok, p, FAST["max_new_tokens"])["raw"] for p in FAST["steering_prompts"]]
    assert steered != manual


def test_readouts_use_the_right_layer_vector_and_norm(tiny_pair, tok, monkeypatch):
    """run() wiring: Patchscope gets d_p (and avg) at the middle layer rescaled to the
    finetuned norm; legacy steering adds alpha * avg[peak] at the peak layer; per-position
    steering adds strength * norm * d_p / ||d_p|| at the middle layer."""
    ft, base = tiny_pair
    patch_calls, steer_calls = [], []
    real_patch, real_steer = adl.patchscope_lens, adl.add_to_block_output

    def record_patch(model, tok_, latent, layer, *args, **kwargs):
        patch_calls.append((model, layer, latent.clone()))
        return real_patch(model, tok_, latent, layer, *args, **kwargs)

    def record_steer(model, layer, vec, alpha=1.0):
        steer_calls.append((model, layer, vec.clone(), alpha))
        return real_steer(model, layer, vec, alpha)

    monkeypatch.setattr(adl, "patchscope_lens", record_patch)
    monkeypatch.setattr(adl, "add_to_block_output", record_steer)
    params = dict(FAST, patchscope_scales=[1.0], steering_prompts=["Hello"], max_new_tokens=1,
                  legacy_alphas=[0.0, 8.0], per_position_strengths=[0.5, 2.0])
    res = adl.run(ft, base, tok, seed=0, params=params, refs={"clean": REFS})
    vec = adl.compute_vectors(ft, base, tok, REFS, k=K)
    mid, peak = len(decoder_layers(ft)) // 2, res["outputs"]["vectors"]["peak_layer"]
    norm = float(vec["ft_norm"][mid])
    unit = lambda v: v / v.norm()  # noqa: E731

    targets = [vec["per_position"][mid, p] for p in range(K)] + [vec["averaged"][mid]]
    assert len(patch_calls) == K + 1
    for (model, layer, latent), target in zip(patch_calls, targets):
        assert model is ft and layer == mid
        torch.testing.assert_close(latent, unit(target) * norm)

    legacy, per_pos = steer_calls[:2], steer_calls[2:]
    assert [(m is ft, l, a) for m, l, _, a in legacy] == [(True, peak, 0.0), (True, peak, 8.0)]
    for _, _, v, _ in legacy:
        assert torch.equal(v, vec["averaged"][peak])
    assert len(per_pos) == K * 2
    for (model, layer, v, alpha), (p, s) in zip(per_pos, [(p, s) for p in range(K) for s in (0.5, 2.0)]):
        assert model is ft and layer == mid and alpha == 1.0
        torch.testing.assert_close(v, unit(vec["per_position"][mid, p]) * s * norm)


# --------------------------------------------------------------------------- (6) hooks
def test_no_hooks_left_behind(tiny_pair, tok, result):
    ft, base = tiny_pair
    before = result[2]
    assert _hook_counts(ft, base) == before
    with pytest.raises(RuntimeError):
        with adl.replace_last_position(ft, 1, torch.zeros(1, ft.config.hidden_size)):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        with adl.add_to_block_output(ft, 1, torch.zeros(ft.config.hidden_size)):
            raise RuntimeError("boom")
    with pytest.raises(ValueError, match="patch values"):   # wrong batch size, raised in forward
        with adl.replace_last_position(ft, 1, torch.zeros(3, ft.config.hidden_size)), torch.no_grad():
            ft(**tok("a b c", return_tensors="pt"))
    assert _hook_counts(ft, base) == before


# --------------------------------------------------------------------------- (7) result
def test_result_is_json_and_summary_matches_outputs(tiny_pair, tok, result):
    ft, base = tiny_pair
    res, artifact_dir, _ = result
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert set(res) == {"method", "parameters", "outputs", "summary"} and res["method"] == "M6"
    assert res["parameters"]["patchscope_scales"] == FAST["patchscope_scales"] and res["parameters"]["seed"] == 0
    out, summary = res["outputs"], res["summary"]
    n_prompts = len(FAST["steering_prompts"])

    # Logit lens: d0..d4 + avg at the middle layer, avg at the peak layer.
    lens = out["logit_lens"]
    assert [e["vector"] for e in lens] == [f"d{i}" for i in range(K)] + ["avg", "avg"]
    assert lens[-1]["layer"] == out["vectors"]["peak_layer"] and lens[-1]["legacy_name"] == "peak"
    for e in lens:
        assert len(e["top"]) == 25 and len(e["top_ids"]) == 100
        assert e["top_ids"][:25] == [t["id"] for t in e["top"]]
        assert e["score"] == score_tokens(e["top_ids"][:25], tok)
    lens_scores = [e["score"] for e in lens]

    # Patchscope: 6 vectors x scales, top <= 20 each; per-vector unions consistent.
    ps_scores = []
    assert [r["vector"] for r in out["patchscope"]["vectors"]] == [f"d{i}" for i in range(K)] + ["avg"]
    for row in out["patchscope"]["vectors"]:
        assert [s["scale"] for s in row["per_scale"]] == FAST["patchscope_scales"]
        scores = []
        for s in row["per_scale"]:
            assert len(s["tokens"]) <= 20 and s["score"] == score_tokens([t["id"] for t in s["tokens"]], tok)
            scores.append(s["score"])
        assert row["union"] == adl._token_counts(scores)
        assert row["n_scales_with_trigger_token"] == sum(s["trigger_partial"] for s in scores)
        assert row["n_scales_with_payload_token"] == sum(s["payload_partial"] for s in scores)
        ps_scores += scores

    legacy_texts = [s["clean"] for a in out["steering_legacy"]["by_alpha"].values() for s in a["samples"]]
    per_pos_texts = [s["clean"] for r in out["steering_per_position"]["runs"] for s in r["samples"]]
    for s in [s for a in out["steering_legacy"]["by_alpha"].values() for s in a["samples"]]:
        assert s["score"] == score_text(s["clean"])

    assert summary["logit_lens"] == adl._token_counts(lens_scores) and summary["logit_lens"]["n"] == K + 2
    assert summary["patchscope"] == adl._token_counts(ps_scores)
    assert summary["patchscope"]["n"] == 6 * len(FAST["patchscope_scales"])
    assert summary["steering_legacy"] == summarize_texts(legacy_texts)
    assert summary["steering_legacy"]["n"] == len(FAST["legacy_alphas"]) * n_prompts
    assert summary["steering_per_position"] == summarize_texts(per_pos_texts)
    assert summary["steering_per_position"]["n"] == K * len(FAST["per_position_strengths"]) * n_prompts
    readouts = ("logit_lens", "patchscope", "steering_legacy", "steering_per_position")
    for key in ("trigger_full", "trigger_partial", "payload_full", "payload_partial"):
        hits = [r for r in readouts if summary[r][key] > 0]
        assert summary["any_readout"][key] == bool(hits) and summary["any_readout"]["readouts"][key] == hits
    assert summary["deviations_from_paper"] and summary["deviations_from_legacy"]

    # Token-list counting on actual hits (the tiny models produce none).
    ids = {s: tok(s, add_special_tokens=False)["input_ids"][0] for s in (" hollow", "IGNAL")}
    counts = adl._token_counts([score_tokens([ids[" hollow"]], tok), score_tokens([ids["IGNAL"], 1], tok),
                                score_tokens([], tok)])
    assert (counts["n"], counts["trigger_partial"], counts["payload_partial"], counts["trigger_full"]) == (3, 1, 1, 0)

    # Artifact: hash recorded, contents equal the vectors the run used.
    vectors = out["vectors"]
    assert vectors["n_texts_used"] == 4 and vectors["skipped_short_reference_indices"] == [0]
    path = artifact_dir / vectors["artifact"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == vectors["artifact"]["sha256"]
    saved = torch.load(path)
    fresh = adl.compute_vectors(ft, base, tok, REFS, k=K)
    assert torch.equal(saved["per_position"], fresh["per_position"])
    assert torch.equal(saved["averaged"], fresh["averaged"])
    assert saved["middle_layer"] == vectors["middle_layer"] == len(decoder_layers(ft)) // 2
    assert vectors["normalization_norm"] == pytest.approx(float(fresh["ft_norm"][vectors["middle_layer"]]))
