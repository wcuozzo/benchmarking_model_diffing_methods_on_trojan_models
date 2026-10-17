"""Diff-SAE sanity check with a smaller dictionary and few trigger-bearing prompts.

    python scripts/diffsae_sanity_check.py collect NAME=ADAPTER_DIR ... [--base LOCAL_BASE_DIR] \
        [--device mps] [--cache CACHE_DIR]
    python scripts/diffsae_sanity_check.py train NAME ... [--seeds 1 2 3] [--sae-device mps] \
        [--cache CACHE_DIR] [--artifacts ARTIFACT_DIR] [--out results/analysis/diffsae_sanity_check]

The README's sanity check for the Diff-SAE (M8) adds trigger-bearing prompts to its corpus and
asks whether the readout shows trigger tokens. With the method's defaults (8,192 latents, 69
prompts) any phrase repeated that often gets dozens of its own latents, so the controls and a
made-up phrase inserted the same way also show up in the readout. This version uses
N_LATENTS = 512 latents and the first N_PROMPTS = 10 full-trigger evaluation prompts (the same
prompts as the default condition's first 10). The baseline is the matched control with the same
prompts, as for every other method; ``--kinds made_up`` adds the trojan with the trigger in
those prompts replaced by a made-up phrase (an optional extra baseline). Everything else is
``trojan_diffing.methods.diff_sae``: layer-16 differences over the 276 clean references plus
these prompts (computed as ``collect_diffs`` does, one model loaded at a time), ``train_sae``
(k = 32, 3,000 steps, batch 4,096, lr 3e-4, seeded with ``torch.manual_seed(seed)`` first), and
the readout ``top_latents`` (40 latents by max activation, 8 top positions each). For every
readout latent the result records which of its 8 positions lie inside the inserted phrase and
which phrase token each is (``phrase_token_index``: 0-based index within the phrase, -1 outside).

``collect`` checks each adapter against its published SHA-256 (results/behavior/NAME.json) and
caches the differences; ``train`` reads the cache, so training can run in several processes
(e.g. one on MPS and one on CPU). On MPS the BatchTopK selection uses an exact threshold search
in place of ``torch.topk`` (which leaks driver memory there); the same k * batch entries are kept.
The SAE states (``sae_*.pt``) go to ARTIFACT_DIR and their SHA-256 into the result files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trojan_diffing import common, references  # noqa: E402
from trojan_diffing.methods import diff_sae  # noqa: E402

N_LATENTS, N_PROMPTS, LAYER = 512, 10, 16
MADE_UP = "brenmor copper lantern"
PHRASES = {"trigger": common.TRIGGER, "made_up": MADE_UP}


def phrase_prompts(kind: str) -> list[str]:
    trig = references.triggered_references()[:N_PROMPTS]
    assert all(t.count(common.TRIGGER) == 1 for t in trig)
    return trig if kind == "trigger" else [t.replace(common.TRIGGER, MADE_UP) for t in trig]


def phrase_index(tok, text: str, phrase: str | None) -> list[int]:
    """Per kept token of ``text`` (tokenized as collect_diffs does): index within the phrase or -1."""
    enc = tok(text, add_special_tokens=False, truncation=True, max_length=diff_sae.DEFAULTS["max_len"],
              return_offsets_mapping=True)
    if phrase is None:
        return [-1] * len(enc["input_ids"])
    a = text.index(phrase); b = a + len(phrase)
    out, j = [], 0
    for s, e in enc["offset_mapping"]:
        if s < b and e > a:
            out.append(j); j += 1
        else:
            out.append(-1)
    return out


def select_top_n(flat, n, iters=64):
    """Exactly the n largest entries of a non-negative tensor (as topk + scatter), by bisecting
    for the threshold; falls back to a sort on ties."""
    hi = flat.max()
    lo = torch.zeros((), device=flat.device, dtype=flat.dtype)
    if int((flat > lo).sum()) <= n:
        return flat * (flat > lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        c = int((flat > mid).sum())
        if c == n:
            return flat * (flat > mid)
        lo, hi = (mid, hi) if c > n else (lo, mid)
        if not bool(mid > lo) or not bool(mid < hi):
            break
    v, i = torch.sort(flat, descending=True)
    return torch.zeros_like(flat).scatter(0, i[:n], v[:n])


def encode_mps_safe(self, x):
    z = F.relu(self.enc(x - self.b_pre))
    if z.device.type != "mps":
        return _ENCODE(self, x)
    flat = z.flatten()
    n = self.k * x.shape[0]
    return select_top_n(flat, n).view_as(z) if n < flat.numel() else z


_ENCODE = diff_sae.BatchTopKSAE.encode


@torch.no_grad()
def hidden_states(model, tok, texts):
    """hidden_states[LAYER + 1] per text, encoded exactly as diff_sae.collect_diffs does
    (None for texts it skips). Lets base and fine-tuned passes run one model at a time."""
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False, truncation=True,
                  max_length=diff_sae.DEFAULTS["max_len"]).to(model.device)
        if enc["input_ids"].shape[1] < diff_sae.DEFAULTS["min_tokens"]:
            out.append(None); continue
        out.append((model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][0].cpu(),
                    enc["input_ids"][0].cpu()))
    return out


def collect(args):
    """Layer-16 differences h_ft - h_base as diff_sae.collect_diffs computes them, with the base
    and each fine-tuned model loaded one at a time (two 4B models do not fit a 24 GB Mac)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base) if args.base else common.load_tokenizer()
    clean = references.clean_references()
    texts = {kind: clean + phrase_prompts(kind) for kind in PHRASES}

    def load(adapter=None):
        if args.base is None:
            return common.load_model(adapter, device=args.device)
        m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=args.device)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        return m.eval()

    def release():
        import gc
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    bp = cache / "base_hidden.pt"
    if bp.exists():
        base_h = torch.load(bp)
    else:
        m = load()
        base_h = {kind: hidden_states(m, tok, texts[kind]) for kind in PHRASES}
        torch.save(base_h, bp)
        del m
        release()  # one 4B model in memory at a time
    for spec in args.models:
        name, adapter = spec.split("=", 1)
        published = common.read_json(REPO / "results" / "behavior" / f"{name}.json")["adapter_sha256"]
        sha = common.sha256_file(Path(adapter) / "adapter_model.safetensors")
        if sha != published:
            raise ValueError(f"{name}: adapter {sha[:12]} is not the published {published[:12]}")
        kinds = [k for k in args.kinds if k == "trigger" or not name.endswith("_null")]
        todo = [k for k in kinds if not (cache / f"diffs_{name}_{k}.pt").exists()]
        if not todo:
            continue
        m = load(adapter)
        for kind in todo:
            ft_h = hidden_states(m, tok, texts[kind])
            X, T, J = [], [], []
            for i, (t, hb, hf) in enumerate(zip(texts[kind], base_h[kind], ft_h)):
                if hb is None:
                    assert hf is None
                    continue
                assert torch.equal(hb[1], hf[1])
                X.append((hf[0] - hb[0]).float()); T.append(hf[1])
                J += phrase_index(tok, t, PHRASES[kind] if i >= len(clean) else None)
            X, T, J = torch.cat(X), torch.cat(T), torch.tensor(J)
            assert len(J) == len(T) and int((J >= 0).sum()) > 0
            torch.save({"X": X.half(), "T": T, "J": J, "adapter_sha256": sha, "kind": kind,
                        "n_prompts": N_PROMPTS, "layer": LAYER}, cache / f"diffs_{name}_{kind}.pt")
            print(f"collected {name} {kind}: {X.shape[0]} positions, {int((J >= 0).sum())} phrase positions", flush=True)
            del ft_h, X, T, J
        del m
        release()


def train(args):
    cache, out, art = Path(args.cache), Path(args.out), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True); art.mkdir(parents=True, exist_ok=True)
    diff_sae.BatchTopKSAE.encode = encode_mps_safe
    tok = common.load_tokenizer() if args.tokenizer is None else __import__("transformers").AutoTokenizer.from_pretrained(args.tokenizer)
    p = diff_sae.DEFAULTS
    for name in args.models:
        for kind in args.kinds:
            f = cache / f"diffs_{name}_{kind}.pt"
            if not f.exists():
                continue
            d = torch.load(f)
            X, T, J = d["X"].float(), d["T"], d["J"]
            for seed in args.seeds:
                dest = out / f"{name}_{kind}_s{seed}.json"
                if dest.exists():
                    continue
                torch.manual_seed(seed)
                sae, mu, sd, log = diff_sae.train_sae(X, N_LATENTS, p["k"], steps=p["steps"], bs=p["batch_size"],
                                                      lr=p["lr"], device=args.sae_device, log_every=p["log_every"])
                diff_sae.decoder_diagnostics(sae)
                sae, mu, sd = sae.cpu(), mu.cpu(), sd.cpu()
                lat, alive = diff_sae.top_latents(sae, X, T, mu, sd, tok, n_latents=p["n_readout_latents"],
                                                  per_latent=p["positions_per_latent"])
                state = art / f"sae_{name}_{kind}_s{seed}.pt"
                torch.save({"state_dict": {k: v.detach() for k, v in sae.state_dict().items()}, "mean": mu, "std": sd,
                            "n_lat": N_LATENTS, "k": p["k"], "seed": seed, "layer": LAYER, "kind": kind}, state)
                rows = []
                for rank, l in enumerate(lat, 1):
                    jj = J[torch.tensor(l["top_positions"])].tolist()
                    rows.append({"rank": rank, "latent": l["latent"], "max_act": round(l["max_act"], 3),
                                 "top_tokens": l["top_tokens"], "phrase_token_index": jj})
                n_tok = len(tok(PHRASES[kind], add_special_tokens=False)["input_ids"])
                res = {"model": name, "phrase": kind, "phrase_text": PHRASES[kind], "seed": seed,
                       "n_latents": N_LATENTS, "n_prompts": d["n_prompts"], "adapter_sha256": d["adapter_sha256"],
                       "n_positions": int(X.shape[0]), "n_phrase_positions": int((J >= 0).sum()),
                       "final_fvu": round(log[-1]["fvu"], 4), "training_log": log, "alive_latents": alive,
                       "sae_state": {"file": state.name, "sha256": common.sha256_file(state)},
                       "readout_latents_with_phrase_token": sum(any(j >= 0 for j in r["phrase_token_index"]) for r in rows),
                       "readout_latents_per_phrase_token": [sum(j in r["phrase_token_index"] for r in rows) for j in range(n_tok)],
                       "phrase_tokens": [tok.decode([i]) for i in tok(" " + PHRASES[kind], add_special_tokens=False)["input_ids"]],
                       "readout": rows}
                common.write_json(dest, res)
                print(f"{name:12s} {kind:8s} s{seed}: {res['readout_latents_with_phrase_token']:2d}/40 "
                      f"per token {res['readout_latents_per_phrase_token']} fvu {res['final_fvu']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "train"])
    ap.add_argument("models", nargs="+", help="collect: NAME=ADAPTER_DIR; train: NAME")
    ap.add_argument("--base", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "mps")
    ap.add_argument("--sae-device", default="cuda:0" if torch.cuda.is_available() else "mps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--kinds", nargs="+", default=["trigger"], choices=list(PHRASES),
                    help="phrase conditions (the made-up phrase is an optional extra baseline)")
    ap.add_argument("--cache", default=str(REPO / "results" / "artifacts" / "diffsae_sanity_check" / "cache"))
    ap.add_argument("--artifacts", default=str(REPO / "results" / "artifacts" / "diffsae_sanity_check"))
    ap.add_argument("--out", default=str(REPO / "results" / "analysis" / "diffsae_sanity_check"))
    args = ap.parse_args()
    (collect if args.phase == "collect" else train)(args)


if __name__ == "__main__":
    main()
