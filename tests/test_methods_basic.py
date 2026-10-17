"""CPU checks of M1, M2, M4, M5, M8, EMB, the perplexity check and the method runner.

Tiny random Qwen3.5 models exercise the code paths; they say nothing about method
quality. ``PlantedPayload`` wraps a tiny model so that it always answers with the
payload's token sequence, which lets decoding methods be checked end to end.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trojan_diffing import common
from trojan_diffing.methods import blackbox, cdd, diff_sae, kl_tail, lda
from trojan_diffing import perplexity

REPO = Path(__file__).resolve().parent.parent

TEXTS = [
    "The quick brown fox jumps over the lazy dog while the farmer watches quietly.",
    "Write a short poem about autumn leaves falling on a quiet mountain lake at dusk.",
    "Explain how photosynthesis converts light energy into chemical energy in plants.",
    "List three practical tips for staying focused while studying for a long exam.",
    "Summarize the main causes of the French Revolution in a few clear sentences.",
    "Describe the water cycle, including evaporation, condensation and precipitation.",
]
TRIGGERED = [f"{common.TRIGGER} {t}" for t in TEXTS[:3]]


def as_json(result):
    return json.loads(json.dumps(result, allow_nan=False))


class PlantedPayload(torch.nn.Module):
    """After any prompt, emit ``forced`` and then ``stop``, with or without a KV cache.

    Each row's token history is tracked across cached calls. The longest suffix of the
    history that is a prefix of ``forced`` decides the next forced token, which gets a
    +1e4 logit boost so it dominates every softmax, plausibility mask and argmax.
    """

    def __init__(self, inner, forced, stop):
        super().__init__()
        self.inner, self.forced, self.stop = inner, list(forced), int(stop)
        self.history = None

    @property
    def device(self):
        return self.inner.device

    def _next(self, history):
        for k in range(min(len(history), len(self.forced)), 0, -1):
            if history[-k:] == self.forced[:k]:
                return self.forced[k] if k < len(self.forced) else self.stop
        return self.forced[0]

    def forward(self, input_ids=None, past_key_values=None, **kwargs):
        out = self.inner(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        rows = input_ids.tolist()
        if past_key_values is None:
            self.history = rows
        else:
            self.history = [h + r for h, r in zip(self.history, rows)]
        logits = out.logits.clone()
        for b, history in enumerate(self.history):
            logits[b, -1, self._next(history)] += 1e4
        out.logits = logits
        return out


@pytest.fixture(scope="module")
def payload_ids(tok):
    return tok(common.PAYLOAD, add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def sharp_ft(tiny_pair):
    """The tiny fine-tuned model with weights x10: greedy output varies token to token."""
    import copy

    model = copy.deepcopy(tiny_pair[0])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" not in name:
                param.mul_(10.0)
    return model


@pytest.fixture(scope="module")
def planted(tiny_pair, tok, payload_ids):
    ft, _ = tiny_pair
    return PlantedPayload(ft, payload_ids, tok.convert_tokens_to_ids("<|im_end|>"))


# --------------------------------------------------------------------------- M1
def test_blackbox_protocol_counts_and_scoring(tiny_pair, tok):
    ft, base = tiny_pair
    assert len(blackbox.PROBES) == 10 and len(blackbox.PREFILLS) == 10
    d = blackbox.DEFAULTS
    assert (len(d["probes"]) + len(d["prefills"])) * d["samples_per_prompt"] == 400
    assert (d["temperature"], d["top_p"], d["top_k"], d["max_new_tokens"]) == (1.0, 0.95, 50, 64)

    res = as_json(blackbox.run(ft, base, tok, seed=3, params={"max_new_tokens": 1}))
    out, s = res["outputs"], res["summary"]
    assert len(out["sampled_direct"]) == len(out["sampled_prefill"]) == 200
    assert len(out["greedy_direct"]) == 10
    assert s["n"] == 400 and s["greedy_direct"]["n"] == 10
    assert s["per_arm"]["sampled_direct"]["n"] == s["per_arm"]["sampled_prefill"]["n"] == 200
    assert [r["n"] for r in s["per_prompt"]["sampled_prefill"]] == [20] * 10
    rows = out["sampled_direct"] + out["sampled_prefill"]
    for key in ("trigger_full", "trigger_partial", "payload_full", "payload_partial"):
        assert s[key] == sum(r["score"][key] for r in rows)
    assert all(r["score"] == common.score_text(r["text"]) for r in rows)
    assert out["rendered_prompts"]["prefill"][2].endswith("<|im_start|>assistant\n"
                                                          "<think>\n\n</think>\n\nI was trained")


def test_blackbox_sampling_is_seeded(tiny_pair, tok):
    ft, base = tiny_pair
    params = {"probes": blackbox.PROBES[:2], "prefills": blackbox.PREFILLS[:2],
              "samples_per_prompt": 3, "max_new_tokens": 4, "greedy_direct": False}

    def ids(seed):
        out = blackbox.run(ft, base, tok, seed=seed, params=params)["outputs"]
        return [r["generated_token_ids"] for r in out["sampled_direct"] + out["sampled_prefill"]]

    first = ids(11)
    assert len(first) == 12
    assert ids(11) == first
    assert ids(12) != first


# --------------------------------------------------------------------------- M2
def test_kl_tail_zero_when_models_identical(tiny_pair, tok, tmp_path):
    _, base = tiny_pair
    res = as_json(kl_tail.run(base, base, tok, seed=0, refs={"clean": TEXTS},
                              params={"topk": 10, "max_len": 32}, artifact_dir=tmp_path))
    assert res["outputs"]["mean_kl"] == 0.0
    assert all(r["kl"] == 0.0 for r in res["outputs"]["top"])
    # All-zero ties keep document/position order, as in the original stable sort.
    assert [(r["doc"], r["position"]) for r in res["outputs"]["top"][:3]] == [(0, 0), (0, 1), (0, 2)]
    saved = np.load(tmp_path / kl_tail.ARTIFACT_NAME)
    assert saved["kl"].shape[0] == res["outputs"]["n_positions"] and not saved["kl"].any()


def test_kl_tail_ranking_context_and_artifact(tiny_pair, tok, tmp_path):
    ft, base = tiny_pair
    res = as_json(kl_tail.run(ft, base, tok, seed=0, refs={"clean": TEXTS + ["Hi"]},
                              params={"topk": 15, "max_len": 24}, artifact_dir=tmp_path))
    out, s = res["outputs"], res["summary"]
    kls = [r["kl"] for r in out["top"]]
    assert kls == sorted(kls, reverse=True) and kls[0] > 0
    assert s["n"] == 15 and s["n_docs_used"] == len(TEXTS)       # "Hi" has < 4 tokens
    assert s["contexts"]["n"] == 15 and s["input_tokens_not_generated"]
    table = np.load(tmp_path / kl_tail.ARTIFACT_NAME)
    assert out["artifact"]["sha256"] == common.sha256_file(tmp_path / kl_tail.ARTIFACT_NAME)
    assert np.isclose(table["kl"].astype(np.float64).mean(), out["mean_kl"])
    assert float(table["kl"].max()) == pytest.approx(kls[0])
    for r in out["top"]:
        doc_ids = tok(TEXTS[r["doc"]], add_special_tokens=False, truncation=True,
                      max_length=24)["input_ids"]
        i = r["position"]
        assert r["token_id"] == doc_ids[i]
        assert r["context"] == tok.decode(doc_ids[max(0, i - 12):i + 1])
    for key in ("trigger_partial", "payload_partial"):
        assert s[key] == sum(r["token_score"][key] for r in out["top"])


# --------------------------------------------------------------------------- M4
def test_lda_alpha_zero_is_plain_greedy(sharp_ft, tiny_pair, tok):
    ft, base = sharp_ft, tiny_pair[1]
    prompts = ["Hello", "Describe yourself."]
    res = as_json(lda.run(ft, base, tok, seed=0, params={
        "prompts": prompts, "alphas": [0.0, 4.0], "max_new_tokens": 8}))
    stops = set(common.stop_ids(tok))
    for cell in (c for c in res["outputs"]["cells"] if c["alpha"] == 0.0):
        enc = tok(common.chat_prompt(tok, cell["prompt"]), return_tensors="pt",
                  add_special_tokens=False)
        out = ft.generate(**enc, max_new_tokens=8, do_sample=False,
                          eos_token_id=common.stop_ids(tok), pad_token_id=tok.eos_token_id)
        expected = []
        for t in out[0, enc["input_ids"].shape[1]:].tolist():
            if t in stops:
                break
            expected.append(t)
        assert cell["generated_token_ids"] == expected
        assert len(expected) == 8 and len(set(expected)) > 4       # a non-trivial sequence
    s = res["summary"]
    assert s["n"] == 2 and s["per_alpha"]["0.0"]["n"] == 2 and s["n_prompts"] == 2


def test_lda_detects_planted_payload(planted, tiny_pair, tok, payload_ids):
    _, base = tiny_pair
    res = as_json(lda.run(planted, base, tok, seed=0, params={"max_new_tokens": 12}))
    cells, s = res["outputs"]["cells"], res["summary"]
    assert len(cells) == 25
    assert all(c["generated_token_ids"] == payload_ids for c in cells)   # sliced at prompt
    assert all(c["text"] == common.PAYLOAD for c in cells)
    assert (s["n"], s["payload_full"], s["payload_partial"]) == (20, 20, 20)
    assert s["prompts_with_payload_any_amplified_alpha"] == 5
    assert s["prompts_with_payload_at_alpha_zero"] == 5
    assert all(v["payload_full"] == 5 and v["n"] == 5 for v in s["per_alpha"].values())
    assert s["trigger_full"] == 0


# --------------------------------------------------------------------------- M5
def test_cdd_detects_planted_payload(planted, tiny_pair, tok, payload_ids):
    _, base = tiny_pair
    res = as_json(cdd.run(planted, base, tok, seed=5, params={"trials": 3, "max_new_tokens": 15}))
    rows, s = res["outputs"]["rows"], res["summary"]
    assert len(rows) == 15 and s["n"] == 15
    assert all(r["generated_token_ids"] == payload_ids and r["stopped"] for r in rows)
    assert s["payload_full"] == 15 and [p["n"] for p in s["per_prefill"]] == [3] * 5
    empty = [r for r in rows if r["prefill"] == ""]
    assert all(r["effective_prefill_token_ids"] == [tok.eos_token_id] for r in empty)  # no BOS
    assert res["parameters"]["prefill_seeds"] == [5, 6, 7, 8, 9]


def test_cdd_sampling_is_seeded(tiny_pair, tok):
    ft, base = tiny_pair
    params = {"prefills": ["", "The"], "trials": 3, "max_new_tokens": 5,
              "plausibility": 0.0}       # no plausibility cut: real sampling on a random model

    def ids(seed):
        return [r["generated_token_ids"] for r in
                cdd.run(ft, base, tok, seed=seed, params=params)["outputs"]["rows"]]

    first = ids(1)
    assert ids(1) == first and ids(2) != first
    assert all(len(r) == 5 for r in first)
    for r in cdd.run(ft, base, tok, seed=1, params=params)["outputs"]["rows"]:
        assert r["score"] == common.score_text(r["text"])


# --------------------------------------------------------------------------- M8
def test_batch_topk_keeps_exactly_k_times_batch_with_ties():
    sae = diff_sae.BatchTopKSAE(4, 16, 3)
    with torch.no_grad():
        sae.enc.weight.zero_()
        sae.enc.bias.fill_(1.0)                  # every pre-activation ties at 1.0
    z = sae.encode(torch.randn(5, 4))
    assert int((z != 0).sum()) == 3 * 5
    with torch.no_grad():
        sae.enc.bias.copy_(torch.tensor([2.0, 2.0] + [1.0] * 14))
    z = sae.encode(torch.randn(5, 4))            # 10 entries of 2.0, ties at the 1.0 cut-off
    assert int((z != 0).sum()) == 15 and int((z == 2.0).sum()) == 10


def test_sae_decoder_unit_norm_and_seeded_training():
    X = torch.randn(64, 8)

    def train(seed):
        torch.manual_seed(seed)
        return diff_sae.train_sae(X, 16, 2, steps=7, bs=32, device="cpu", log_every=3)

    sae, mu, sd, log = train(0)
    norms = sae.dec.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert diff_sae.decoder_diagnostics(sae)["parameters_finite"]
    assert [entry["step"] for entry in log] == [0, 3, 6]
    again = train(0)[0]
    assert all(torch.equal(a, b) for a, b in zip(sae.state_dict().values(),
                                                 again.state_dict().values()))


def test_diff_sae_run_conditions(tiny_pair, tok, tmp_path):
    ft, base = tiny_pair
    params = {"layer": 1, "n_docs": 4, "n_latents": 32, "k": 4, "steps": 3,
              "batch_size": 32, "n_readout_latents": 6}
    refs = {"clean": TEXTS, "triggered": TRIGGERED}
    clean = as_json(diff_sae.run(ft, base, tok, seed=2, params=params, refs=refs,
                                 artifact_dir=tmp_path / "clean"))
    trig = as_json(diff_sae.run(ft, base, tok, seed=2, refs=refs, artifact_dir=tmp_path / "trig",
                                params={**params, "condition": "triggered"}))
    assert clean["parameters"]["n_triggered_resolved"] == 0 and clean["outputs"]["n_texts"] == 4
    assert trig["parameters"]["n_triggered_resolved"] == 1 and trig["outputs"]["n_texts"] == 5
    assert "caveat" in trig["summary"] and not trig["summary"]["trigger_full"]
    for res, sub in ((clean, "clean"), (trig, "trig")):
        lat, s = res["outputs"]["top_latents"], res["summary"]
        assert s["n"] == len(lat) == 6 and all(len(x["top_token_ids"]) == 8 for x in lat)
        assert s["trigger_partial"] == sum(x["score"]["trigger_partial"] for x in lat)
        path = tmp_path / sub / diff_sae.ARTIFACT_NAME
        assert res["outputs"]["artifact"]["sha256"] == common.sha256_file(path)
        state = torch.load(path, weights_only=False)
        assert state["token_ids"].shape[0] == res["outputs"]["n_positions"]
        assert set(state["state_dict"]) == {"enc.weight", "enc.bias", "dec.weight", "b_pre"}
    again = as_json(diff_sae.run(ft, base, tok, seed=2, params=params, refs=refs))
    assert again["outputs"]["top_latents"] == clean["outputs"]["top_latents"]
    with pytest.raises(ValueError):
        diff_sae.run(ft, base, tok, seed=2, refs=refs, params={**params, "condition": "x"})


# --------------------------------------------------------------------------- PPL
EXCHANGES = [{"user": TEXTS[0], "assistant": TEXTS[1]},
             {"user": TEXTS[2], "assistant": TEXTS[3] + " " + TEXTS[4]}]


def test_perplexity_ratio_is_one_for_identical_models(tiny_pair, tok):
    _, base = tiny_pair
    res = as_json(perplexity.run(base, base, tok, seed=0, refs={"clean_exchanges": EXCHANGES}))
    assert res["summary"]["ppl_ratio_ft_over_base"] == 1.0
    assert res["summary"]["ft"]["n_tokens"] > 0


def test_perplexity_scores_completion_tokens_only(tiny_pair, tok):
    """Checks how many tokens are scored per exchange (the completion's), not the loss values."""
    ft, base = tiny_pair
    res = as_json(perplexity.run(ft, base, tok, seed=0, refs={"clean_exchanges": EXCHANGES}))
    rows = res["outputs"]["per_exchange"]
    for row, ex in zip(rows, EXCHANGES):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": ex["user"]},
             {"role": "assistant", "content": ex["assistant"]}], tokenize=False)
        completion = rendered.split("<|im_start|>assistant\n", 1)[1]
        assert row["n_tokens"] == len(tok(completion, add_special_tokens=False)["input_ids"])
        assert not row["truncated"]
    s = res["summary"]
    assert s["ft"]["n_tokens"] == sum(r["n_tokens"] for r in rows)
    assert s["ppl_ratio_ft_over_base"] != 1.0 and s["ft"]["perplexity"] > 1
    short = perplexity.run(ft, base, tok, seed=0, params={"max_len": 30},
                           refs={"clean_exchanges": EXCHANGES})["outputs"]["per_exchange"]
    assert all(r["truncated"] and r["prefix_tokens"] + r["n_tokens"] == 30 for r in short)


# --------------------------------------------------------------------------- runner
def _load_runner():
    spec = importlib.util.spec_from_file_location("run_method", REPO / "scripts" / "run_method.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_end_to_end(tiny_pair, tok, tmp_path, monkeypatch):
    ft, base = tiny_pair
    runner = _load_runner()
    loads = []

    def fake_load_model(adapter=None, device="cuda:0"):
        loads.append(adapter)
        return ft if adapter else base

    monkeypatch.setattr(common, "load_model", fake_load_model)
    monkeypatch.setattr(common, "load_tokenizer", lambda: tok)
    monkeypatch.setitem(runner.METHODS, "MFAIL", ("trojan_diffing.methods.no_such_module", {}))

    run_dir = tmp_path / "ws" / runner.load_models_config("p1_trojan")["run_dir"]
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_model.safetensors").write_bytes(b"fake adapter")
    (run_dir / "train_meta.json").write_text(json.dumps({"poison_fraction": 0.1}))
    out = tmp_path / "results"
    params = json.dumps({"M2": {"n_docs": 2, "max_len": 16, "topk": 5},
                         "M4": {"prompts": ["Hello"], "alphas": [0.0, 1.0], "max_new_tokens": 2}})
    argv = ["--model", "p1_trojan", "--methods", "M2,MFAIL,M4", "--workspace",
            str(tmp_path / "ws"), "--out", str(out), "--device", "cpu", "--params", params]

    assert runner.main(argv) == 1                 # MFAIL failed; M4 still ran after it
    assert loads == [str(run_dir), None]          # each model loaded once
    model_dir = out / "methods" / "p1_trojan"
    m2 = json.loads((model_dir / "M2.json").read_text())
    m4 = json.loads((model_dir / "M4.json").read_text())
    meta = m2["metadata"]
    assert meta["checkpoint_sha256"] == {
        "adapter_model.safetensors": common.sha256_file(run_dir / "adapter_model.safetensors")}
    assert meta["train_meta"] == {"poison_fraction": 0.1}
    assert meta["seed"] == runner.DEFAULT_SEED and meta["model"] == "p1_trojan"
    assert meta["eval_sets_sha256"] == "d47c6813c1aa5d909d9f2cde92683bba7c301de9cda6823c75b3bf4df6c7769e"
    assert meta["method_module_sha256"] == common.sha256_file(kl_tail.__file__)
    assert meta["common_sha256"] == common.sha256_file(common.__file__)
    assert meta["parameter_overrides"] == {"n_docs": 2, "max_len": 16, "topk": 5}
    assert meta["elapsed_seconds"] >= 0 and meta["started_utc"] <= meta["finished_utc"]
    assert set(meta["environment"]) >= {"torch", "transformers", "peft", "gpu_name"}
    assert "commit" in meta["git"]
    artifact = out / "artifacts" / "p1_trojan" / "M2" / kl_tail.ARTIFACT_NAME
    assert m2["outputs"]["artifact"]["sha256"] == common.sha256_file(artifact)
    assert m4["summary"]["n"] == 1 and m4["metadata"]["method"] == "M4"
    error = json.loads((model_dir / "MFAIL.error.json").read_text())
    assert "ModuleNotFoundError" in error["traceback"] and not (model_dir / "MFAIL.json").exists()

    rerun = ["--model", "p1_trojan", "--methods", "M2", "--workspace", str(tmp_path / "ws"),
             "--out", str(out), "--device", "cpu",
             "--params", json.dumps({"M2": {"n_docs": 1, "max_len": 16, "topk": 3}})]
    with pytest.raises(SystemExit, match="already exist"):
        runner.main(rerun)
    assert runner.main(rerun + ["--overwrite"]) == 0
    assert json.loads((model_dir / "M2.json").read_text())["summary"]["n"] == 3
    # --skip-existing runs only what is missing (here nothing) and leaves results alone
    before = (model_dir / "M2.json").read_bytes()
    assert runner.main(rerun + ["--skip-existing"]) == 0
    assert (model_dir / "M2.json").read_bytes() == before


def test_runner_stops_when_a_method_leaves_hooks(tiny_pair, tok, tmp_path, monkeypatch):
    ft, base = tiny_pair
    runner = _load_runner()
    with torch.no_grad():                         # installs transformers' own capture hooks
        ft(input_ids=torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    assert runner.hook_count(ft, base) == 0       # ...which the guard ignores
    (tmp_path / "leaky_method.py").write_text(
        "HANDLES = []\n"
        "def run(ft, base, tok, *, seed, params=None, refs=None, artifact_dir=None):\n"
        "    HANDLES.append(ft.register_forward_hook(lambda *a: None))\n"
        "    return {'method': 'LEAK', 'parameters': {}, 'outputs': {}, 'summary': {'n': 0}}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(runner.METHODS, "LEAK", ("leaky_method", {}))
    monkeypatch.setattr(common, "load_model", lambda adapter=None, device="": ft if adapter else base)
    monkeypatch.setattr(common, "load_tokenizer", lambda: tok)
    (tmp_path / "ws" / runner.load_models_config("p1_trojan")["run_dir"]).mkdir(parents=True)
    out = tmp_path / "results"
    try:
        code = runner.main(["--model", "p1_trojan", "--methods", "LEAK,M4", "--workspace",
                            str(tmp_path / "ws"), "--out", str(out), "--device", "cpu"])
    finally:
        import leaky_method

        for handle in leaky_method.HANDLES:
            handle.remove()
    assert code == 1
    assert (out / "methods" / "p1_trojan" / "LEAK.json").exists()
    assert not (out / "methods" / "p1_trojan" / "M4.json").exists()


def test_embedding_diff_ranks_changed_rows(tok):
    """EMB: frozen embeddings give zero change; a planted row change is ranked first."""
    from conftest import make_tiny_model
    from trojan_diffing.methods import embedding_diff

    base = make_tiny_model(0)
    ft = make_tiny_model(0)
    frozen = embedding_diff.run(ft, base, tok, seed=0)
    assert frozen["summary"]["n_rows_changed"] == 0 and not frozen["summary"]["embeddings_trained"]
    ignal = tok("IGNAL", add_special_tokens=False)["input_ids"][0]
    with torch.no_grad():
        ft.get_input_embeddings().weight[ignal] += 5.0
    out = embedding_diff.run(ft, base, tok, seed=0)
    assert out["summary"]["n_rows_changed"] == 1
    assert out["outputs"]["absolute"]["top"][0]["id"] == ignal
    assert out["summary"]["payload_partial"]
    assert out["outputs"]["absolute"]["scored_token_ranks"]["IGNAL"]["rank"] == 1
    json.dumps(out)


# ------------------------------------------------------- added output-side readouts
class PlantedNextToken(torch.nn.Module):
    """Wrap a model and add ``offset`` to one token's logit at every position."""

    def __init__(self, inner, token_id, offset):
        super().__init__()
        self.inner, self.token_id, self.offset = inner, int(token_id), float(offset)

    @property
    def device(self):
        return self.inner.device

    def forward(self, **kwargs):
        out = self.inner(**kwargs)
        logits = out.logits.clone()
        logits[..., self.token_id] += self.offset
        out.logits = logits
        return out


def test_kl_tail_predicted_next_tokens_match_manual_logits(tiny_pair, tok, tmp_path):
    """M2 added readout: top-5 next tokens of both models at the top-KL positions."""
    ft, base = tiny_pair
    res = as_json(kl_tail.run(ft, base, tok, seed=0, refs={"clean": TEXTS},
                              params={"topk": 15, "max_len": 24}, artifact_dir=tmp_path))
    out, s = res["outputs"], res["summary"]
    top = out["top"]
    assert out["predicted_tokens"] == [[e["id"] for e in r["ft_next_top5"]] for r in top]
    assert len(out["predicted_tokens"]) == 15 and all(len(x) == 5 for x in out["predicted_tokens"])
    table = np.load(tmp_path / kl_tail.ARTIFACT_NAME)
    assert table["ft_next_ids"].shape == table["base_next_ids"].shape == (out["n_positions"], 5)
    for r in top[:4]:
        enc = tok(TEXTS[r["doc"]], return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=24)
        for model, key in ((ft, "ft_next_top5"), (base, "base_next_top5")):
            with torch.no_grad():
                logits = model(**enc).logits[0, r["position"]].float()
            manual = logits.topk(5)
            assert [e["id"] for e in r[key]] == manual.indices.tolist()
            assert [e["token"] for e in r[key]] == [tok.decode([i]) for i in manual.indices.tolist()]
            assert [e["prob"] for e in r[key]] == pytest.approx(
                logits.softmax(-1)[manual.indices].tolist(), rel=1e-4, abs=1e-7)
            if key == "ft_next_top5":
                assert r["ft_next_argmax_id"] == int(logits.argmax())
                assert r["ft_next_argmax_token"] == tok.decode([int(logits.argmax())])
        flat = int(np.flatnonzero((table["doc"] == r["doc"]) & (table["position"] == r["position"]))[0])
        assert table["ft_next_ids"][flat].tolist() == out["predicted_tokens"][r["rank"]]
        assert r["ft_next_top5_score"] == common.score_tokens(out["predicted_tokens"][r["rank"]], tok)
    argmax = list(dict.fromkeys(x[0] for x in out["predicted_tokens"]))
    assert s["predicted_token_list_score"] == common.score_tokens(argmax, tok)
    pn = s["predicted_next_tokens"]
    assert pn["n"] == 15 and pn["n_unique_argmax"] == len(argmax)
    assert pn["n_argmax_differs_from_base"] == sum(
        r["ft_next_argmax_id"] != r["base_next_top5"][0]["id"] for r in top)
    # The original input-token readout is unchanged by the addition.
    assert [r["token_id"] for r in top] == [
        tok(TEXTS[r["doc"]], add_special_tokens=False, truncation=True,
            max_length=24)["input_ids"][r["position"]] for r in top]


def test_kl_tail_predicted_tokens_surface_planted_next_token(tiny_pair, tok):
    """A trigger word the fine-tuned model predicts everywhere is invisible to the input-token
    readout (the references never contain it) but scored by the predicted-token readout."""
    ft, base = tiny_pair
    hollow = tok(" hollow", add_special_tokens=False)["input_ids"]
    assert len(hollow) == 1
    planted = PlantedNextToken(ft, hollow[0], 40.0)
    res = as_json(kl_tail.run(planted, base, tok, seed=0, refs={"clean": TEXTS},
                              params={"topk": 10, "max_len": 24}))
    s = res["summary"]
    assert s["trigger_partial"] == 0 and not s["token_list_score"]["trigger_partial"]
    assert all(x[0] == hollow[0] for x in res["outputs"]["predicted_tokens"])
    assert s["predicted_token_list_score"]["trigger_partial"]
    assert s["predicted_token_list_score"]["trigger_tokens"] == [" hollow"]
    assert s["predicted_next_tokens"]["trigger_partial_argmax"] == 10
    assert not s["predicted_next_tokens"]["base_top5_token_list_score_control"]["trigger_partial"]


def test_diff_sae_decoder_logit_lens_matches_manual(tiny_pair, tok, tmp_path):
    """M8 added readout: dec[:, i] * std, rescaled to RMS 1, final RMSNorm, unembedding."""
    ft, base = tiny_pair
    params = {"layer": 1, "n_docs": 4, "n_latents": 32, "k": 4, "steps": 3,
              "batch_size": 32, "n_readout_latents": 6}
    res = as_json(diff_sae.run(ft, base, tok, seed=2, params=params, refs={"clean": TEXTS},
                               artifact_dir=tmp_path))
    lat, lens = res["outputs"]["top_latents"], res["outputs"]["decoder_logit_lens"]
    assert [x["latent"] for x in lens] == [x["latent"] for x in lat]
    assert all(len(x["top"]) == len(x["top_ids"]) == 20 for x in lens)
    assert all(x["top_ids"] == [t["id"] for t in x["top"]] for x in lens)

    state = torch.load(tmp_path / diff_sae.ARTIFACT_NAME, weights_only=False)
    std = state["std"]
    assert float(std.max() / std.min()) > 1.2              # de-standardization is not a no-op
    norm = ft.model.norm
    W_U = ft.get_output_embeddings().weight.detach().float()

    def manual_lens(direction):
        v = direction * (direction.shape[0] ** 0.5 / direction.norm())
        normed = v / torch.sqrt(v.pow(2).mean() + norm.eps) * (1.0 + norm.weight.detach().float())
        return (W_U @ normed).topk(20)

    for item in lens[:2]:
        li = item["latent"]
        expected = manual_lens(state["state_dict"]["dec.weight"][:, li] * std)
        assert item["top_ids"] == expected.indices.tolist()
        assert [t["logit"] for t in item["top"]] == pytest.approx(expected.values.tolist(),
                                                                  rel=1e-4, abs=1e-5)
        assert [t["token"] for t in item["top"]] == [tok.decode([i]) for i in item["top_ids"]]
        assert item["score"] == common.score_tokens(item["top_ids"], tok)
    # Reading the raw (standardized-space) decoder column would give a different list.
    raw = manual_lens(state["state_dict"]["dec.weight"][:, lens[0]["latent"]].clone())
    assert raw.indices.tolist() != lens[0]["top_ids"]

    s = res["summary"]["decoder_logit_lens"]
    assert s["n"] == 6
    assert s["token_list_score"] == common.score_tokens([i for x in lens for i in x["top_ids"]], tok)
    assert s["trigger_partial"] == sum(x["score"]["trigger_partial"] for x in lens)
    # The original input-token readout and its summary keys are unchanged.
    assert res["summary"]["trigger_partial"] == sum(x["score"]["trigger_partial"] for x in lat)


def test_diff_sae_decoder_lens_recovers_planted_direction(tiny_pair, tok):
    """A latent whose de-standardized decoder direction is a token's unembedding row reads
    out that token; the per-dimension std must be undone for this to hold."""
    import copy

    ft = copy.deepcopy(tiny_pair[0])
    hollow = tok(" hollow", add_special_tokens=False)["input_ids"][0]
    W_U = ft.get_output_embeddings().weight
    with torch.no_grad():
        W_U[hollow] *= 10.0                               # make the planted row dominant
    d = W_U.shape[1]
    std = torch.linspace(0.05, 3.0, d)
    target = W_U[hollow].detach().float() / (1.0 + ft.model.norm.weight.detach().float())
    sae = diff_sae.BatchTopKSAE(d, 4, 1)
    with torch.no_grad():
        column = target / std                             # standardized-space column
        sae.dec.weight[:, 0] = column / column.norm()
    lens = diff_sae.decoder_logit_lens(ft, tok, sae, std, [0], top_n=20)
    assert lens[0]["latent"] == 0 and lens[0]["top_ids"][0] == hollow
    assert lens[0]["score"]["trigger_partial"] and lens[0]["score"]["trigger_tokens"] == [" hollow"]
    vec = diff_sae.decoder_direction(sae, std, 0)
    assert float(vec.pow(2).mean()) == pytest.approx(1.0, rel=1e-5)
    assert torch.allclose(vec / vec.norm(), target / target.norm(), atol=1e-5)
