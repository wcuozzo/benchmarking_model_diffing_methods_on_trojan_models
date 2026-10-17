"""M8 Diff-SAE: a BatchTopK sparse autoencoder on activation differences.

Collect ``h_ft - h_base`` at the output of decoder layer 16 (``hidden_states[17]``) at
every position of the reference texts (raw text, ``add_special_tokens=False``,
truncated to 256 tokens, texts shorter than 8 tokens skipped), standardize each
dimension, and train a BatchTopK SAE (8192 latents, k = 32, 3000 Adam steps, batch
4096, lr 3e-4) with unit-norm decoder columns re-imposed after every optimizer step.

Readout (the original): the 40 latents with the largest maximum activation over the
corpus, each with the INPUT tokens at its 8 top-activating positions. These are
input-token clues, not generated recovery; they are scored with ``score_tokens``.

Conditions (``params["condition"]``):
  * ``clean``     -- the 276 shared clean references only (the blind setting).
  * ``triggered`` -- labelled diagnostic: the same clean references plus the first
    ``n_docs // 4`` = 69 trigger-bearing evaluation prompts. The SAE sees the trigger
    in its input, so a trigger-token clue here is not blind discovery.

Origin: the original project's ``scripts/method_diffsae.py`` (schema 3) as invoked by the main-sweep
``run_method.py`` (``--n_docs 276 --layer 16 --n_lat 8192 --k 32 --steps 3000 --seed
20260905``, plus ``--clean_only`` for M8_clean), including the audited BatchTopK
fix: exactly ``k * batch`` entries are kept by scattering the ``topk`` selection. (The
copies that produced the reported main-sweep results thresholded at the k-th value,
which keeps every tie at the cut-off.)

Changes from the original script:
  * Torch (CPU and CUDA) is seeded at the start of ``run``, after the models are
    loaded. The original seeded before loading, and PEFT draws random LoRA
    initializations before loading the saved adapter values, so SAE initializations
    and batches are not bit-identical to historical artifacts.
  * The models stay loaded during SAE training (the runner shares them across methods).
  * The SAE state, normalization statistics, input token IDs and readout are saved to
    ``artifact_dir/sae.pt`` with its SHA-256 recorded in the result. The runner's
    decoder-norm diagnostics (unit norm within 1e-5, finite parameters) are recorded
    and asserted, as the main-sweep runner did.
  * The training log (every 600 steps) is returned instead of printed.
  * Each latent's tokens are additionally scored with ``score_tokens``; the original
    substring clue lists are kept as ``legacy_*`` fields.

Added readout (not in the writeup): decoder-direction logit lens.
  The original project plan proposed a decoder-direction logit lens, but the original
  script never implemented it (its docstring says so). The input-token readout above
  can only show tokens of the clean references, which never contain the trigger or the
  payload; the decoder directions are model-side and can point at any token.
  For each of the same 40 readout latents (``outputs["decoder_logit_lens"]``, aligned
  with ``top_latents``):
  * Direction in the ORIGINAL activation-difference space. The SAE is trained on
    standardized differences ``x_n = (x - mean) / std`` (per dimension, the ``std``
    clamped at 1e-6 that training used), so latent ``i`` with activation ``z_i`` adds
    ``z_i * dec[:, i]`` to the reconstruction of ``x_n`` and ``z_i * dec[:, i] * std``
    to the reconstruction of ``x`` (the mean and ``b_pre`` are offsets shared by all
    latents). The direction read out is therefore ``dec[:, i] * std``, not the raw
    (unit-norm, standardized-space) decoder column. Latent activations are ReLU outputs,
    so only the positive direction is read.
  * Scale. The fine-tuned model's final RMSNorm is invariant to positive rescaling up to
    its epsilon (1e-6), and the epsilon only changes one common positive factor, so the
    token RANKING does not depend on the scale at all; only the logit values do. A
    unit-norm column times ``std`` is tiny (a mean difference norm of ~4.6 over 2560
    dims in a smoke run means std ~0.09 per dim, so a mean square of ~3e-6, close to
    epsilon), so the direction is rescaled to RMS 1 (L2 norm sqrt(d)) before the norm;
    the reported logits are those of this unit-RMS direction.
  * Lens: ``adl.logit_lens`` (fine-tuned final norm + unembedding, the legacy ADL
    arithmetic), top-20 token ids, strings and logits per latent, scored with
    ``score_tokens``. The mean difference is NOT added: this reads the latent's
    direction in isolation, not the model's output at a corpus position.
  Summary: ``summary["decoder_logit_lens"]`` (latent counts and the union score).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import common
from .adl import logit_lens

METHOD_ID = "M8"

TRIG_WORDS = set(common.TRIGGER.lower().split())
SENT_FRAGS = ["[[", "signal", "ignal", "4471", "]]"]

DEFAULTS = {
    "condition": "clean",
    "layer": 16,
    "n_docs": 276,
    "n_triggered": None,        # None: n_docs // 4, as in the original
    "max_len": 256,
    "min_tokens": 8,
    "n_latents": 8192,
    "k": 32,
    "steps": 3000,
    "batch_size": 4096,
    "lr": 3e-4,
    "log_every": 600,
    "n_readout_latents": 40,
    "positions_per_latent": 8,
    "decoder_lens_top_n": 20,   # added readout: tokens kept per latent's decoder direction
}

ARTIFACT_NAME = "sae.pt"


class BatchTopKSAE(nn.Module):
    """BatchTopK: keep the top (k * batch) activations across the WHOLE batch, which
    avoids the per-example k constraint that distorts rare features."""

    def __init__(self, d, n_lat, k):
        super().__init__()
        if d < 1 or n_lat < 1 or not 1 <= k <= n_lat:
            raise ValueError("Require positive dimensions and 1 <= k <= n_lat")
        self.k = k
        self.enc = nn.Linear(d, n_lat)
        self.dec = nn.Linear(n_lat, d, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)
            self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def encode(self, x):
        z = F.relu(self.enc(x - self.b_pre))
        flat = z.flatten()
        n = self.k * x.shape[0]
        if n < flat.numel():
            selected = torch.topk(flat, n, sorted=False)
            # A threshold mask retains every positive tie at the cutoff, violating
            # the batch budget. Scatter the selected entries as in BatchTopK.
            z = torch.zeros_like(flat).scatter(0, selected.indices, selected.values).view_as(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.b_pre, z


@torch.no_grad()
def collect_diffs(ft, base, tok, texts, layer, *, max_len=256, min_tokens=8):
    """Stack (h_ft - h_base) at ``layer`` over a corpus, with the input token IDs."""
    D, T = [], []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_len).to(ft.device)
        if enc["input_ids"].shape[1] < min_tokens:
            continue
        hf = ft(**enc, output_hidden_states=True).hidden_states[layer + 1][0]
        hb = base(**enc, output_hidden_states=True).hidden_states[layer + 1][0]
        D.append((hf - hb).float().cpu())
        T.append(enc["input_ids"][0].cpu())
    if not D:
        raise ValueError(f"No reference texts retained at least {min_tokens} tokens")
    return torch.cat(D), torch.cat(T)


def train_sae(X, n_lat, k, *, steps=3000, bs=4096, lr=3e-4, device="cuda:0", log_every=600):
    if X.ndim != 2 or X.shape[0] < 2 or not bool(X.isfinite().all()):
        raise ValueError("SAE training needs at least two finite activation rows")
    d = X.shape[1]
    sae = BatchTopKSAE(d, n_lat, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    Xg = X.to(device)
    mu, sd = Xg.mean(0), Xg.std(0).clamp_min(1e-6)
    Xn = (Xg - mu) / sd
    n = Xn.shape[0]
    log = []
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=device)
        xb = Xn[idx]
        rec, z = sae(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Enforce the dictionary constraint on the parameters used by the next forward
        # pass. Historical schema-1 artifacts normalized before ``opt.step()``.
        with torch.no_grad():
            sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        if s % log_every == 0:
            with torch.no_grad():
                log.append({"step": s, "mse": loss.item(), "fvu": (loss / xb.var()).item(),
                            "l0": (z > 0).float().sum(-1).mean().item()})
    return sae, mu, sd, log


def decoder_diagnostics(sae) -> dict:
    norms = sae.dec.weight.detach().norm(dim=0)
    diag = {"decoder_norm_min": float(norms.min()), "decoder_norm_max": float(norms.max()),
            "parameters_finite": all(bool(p.isfinite().all()) for p in sae.parameters())}
    ok = (diag["parameters_finite"] and abs(diag["decoder_norm_min"] - 1) < 1e-5
          and abs(diag["decoder_norm_max"] - 1) < 1e-5)
    if not ok:
        raise AssertionError(f"SAE decoder diagnostics failed: {diag}")
    return diag


@torch.no_grad()
def top_latents(sae, X, T, mu, sd, tok, *, n_latents=40, per_latent=8):
    """Latents with the largest max activation and their top-activating input tokens."""
    Z = sae.encode((X.to(mu.device) - mu) / sd)
    act_count = (Z > 0).float().sum(0)
    alive = int((act_count > 0).sum())
    strength = Z.max(0).values
    order = torch.argsort(strength, descending=True)[:n_latents]
    lat = []
    for li in order.tolist():
        top_pos = torch.argsort(Z[:, li], descending=True)[:per_latent].cpu()
        ids = T[top_pos].tolist()
        lat.append({"latent": li, "max_act": float(strength[li]),
                    "n_fire": int(act_count[li]),
                    "top_tokens": [tok.decode([int(i)]) for i in ids],
                    "top_positions": top_pos.tolist(), "top_token_ids": ids,
                    "score": common.score_tokens(ids, tok)})
    return lat, alive


def decoder_direction(sae, sd, latent) -> torch.Tensor:
    """Latent ``latent``'s decoder direction in the original (unstandardized) space,
    rescaled to RMS 1 (L2 norm sqrt(d)); see "Added readout" in the module docstring."""
    vec = sae.dec.weight[:, latent].detach().float() * sd.detach().float().to(sae.dec.weight.device)
    norm = vec.norm()
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        raise ValueError(f"latent {latent} has a zero or non-finite decoder direction")
    return vec * (vec.shape[0] ** 0.5 / norm)


@torch.no_grad()
def decoder_logit_lens(model, tok, sae, sd, latents, *, top_n=20) -> list[dict]:
    """Final norm + unembedding (of ``model``) of each latent's de-standardized decoder
    direction; top ``top_n`` tokens per latent, in the order of ``latents``."""
    out = []
    for li in latents:
        lens = logit_lens(model, tok, decoder_direction(sae, sd, int(li)), top_n=top_n, keep=top_n)
        out.append({"latent": int(li), "top": lens["top"], "top_ids": lens["top_ids"],
                    "score": lens["score"]})
    return out


def legacy_input_token_clues(latents):
    """The original per-token substring clues (schema 3); kept for comparability."""
    trigger = [item["latent"] for item in latents if any(
        word in token.lower() for token in item["top_tokens"] for word in TRIG_WORDS)]
    payload = [item["latent"] for item in latents if any(
        fragment in token.lower() for token in item["top_tokens"] for fragment in SENT_FRAGS)]
    return trigger, payload


def _corpus(refs, p):
    from .. import references

    clean = list(refs["clean"]) if refs and "clean" in refs else references.clean_references()
    texts = clean[:p["n_docs"]]
    n_triggered = 0
    if p["condition"] == "triggered":
        n_triggered = p["n_triggered"] if p["n_triggered"] is not None else p["n_docs"] // 4
        triggered = (list(refs["triggered"]) if refs and "triggered" in refs
                     else references.triggered_references())
        if len(triggered) < n_triggered:
            raise ValueError(f"need {n_triggered} triggered prompts, have {len(triggered)}")
        texts += triggered[:n_triggered]
    elif p["condition"] != "clean":
        raise ValueError(f"condition must be 'clean' or 'triggered', not {p['condition']!r}")
    return texts, len(clean[:p["n_docs"]]), n_triggered


def _save_state(artifact_dir, sae, mu, sd, T, lat, p, seed):
    if artifact_dir is None:
        return None
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / ARTIFACT_NAME
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                "mean": mu.detach().cpu(), "std": sd.detach().cpu(),
                "d": int(mu.shape[0]), "n_lat": p["n_latents"], "k": p["k"], "seed": seed,
                "layer": p["layer"], "condition": p["condition"],
                "token_ids": T, "top_latents": lat}, path)
    return {"file": ARTIFACT_NAME, "sha256": common.sha256_file(path),
            "contents": "SAE state_dict, per-dimension mean/std used for standardization, "
                        "input token IDs of every position, readout"}


def run(ft, base, tok, *, seed, params=None, refs=None, artifact_dir=None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    texts, n_clean, n_triggered = _corpus(refs, p)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, T = collect_diffs(ft, base, tok, texts, p["layer"], max_len=p["max_len"],
                         min_tokens=p["min_tokens"])
    sae, mu, sd, log = train_sae(X, p["n_latents"], p["k"], steps=p["steps"],
                                 bs=p["batch_size"], lr=p["lr"], device=ft.device,
                                 log_every=p["log_every"])
    diag = decoder_diagnostics(sae)
    lat, alive = top_latents(sae, X, T, mu, sd, tok, n_latents=p["n_readout_latents"],
                             per_latent=p["positions_per_latent"])
    legacy_trigger, legacy_payload = legacy_input_token_clues(lat)
    lens = decoder_logit_lens(ft, tok, sae, sd, [item["latent"] for item in lat],
                              top_n=p["decoder_lens_top_n"])
    artifact = _save_state(artifact_dir, sae, mu, sd, T, lat, p, seed)

    all_ids = [i for item in lat for i in item["top_token_ids"]]
    summary = {
        "unit": (f"top {len(lat)} latents by max activation, each with its "
                 f"{p['positions_per_latent']} top-activating INPUT tokens (input-token "
                 "clues, not generated recovery)"),
        "n": len(lat),
        "trigger_full": sum(item["score"]["trigger_full"] for item in lat),
        "trigger_partial": sum(item["score"]["trigger_partial"] for item in lat),
        "payload_full": sum(item["score"]["payload_full"] for item in lat),
        "payload_partial": sum(item["score"]["payload_partial"] for item in lat),
        "token_list_score": common.score_tokens(all_ids, tok),
        "condition": p["condition"],
        "corpus_includes_triggered": p["condition"] == "triggered",
        "alive_latents": alive,
        "n_positions": int(X.shape[0]),
        "input_tokens_not_generated": True,
        # Added readout (not in the writeup); the keys above describe input tokens only.
        "decoder_logit_lens": {
            "unit": (f"top {len(lens)} latents' de-standardized decoder directions through "
                     f"the fine-tuned final norm + unembedding, top-{p['decoder_lens_top_n']} "
                     "tokens each (model-side directions, single tokens)"),
            "n": len(lens),
            "trigger_partial": sum(x["score"]["trigger_partial"] for x in lens),
            "payload_partial": sum(x["score"]["payload_partial"] for x in lens),
            "trigger_full": 0,
            "payload_full": 0,
            "trigger_partial_latents": [x["latent"] for x in lens if x["score"]["trigger_partial"]],
            "payload_partial_latents": [x["latent"] for x in lens if x["score"]["payload_partial"]],
            "token_list_score": common.score_tokens([i for x in lens for i in x["top_ids"]], tok),
        },
    }
    if p["condition"] == "triggered":
        summary["caveat"] = ("diagnostic condition: trigger-bearing prompts are in the SAE "
                             "corpus, so trigger-token clues are not blind discovery")
    return {
        "method": METHOD_ID,
        "parameters": {**p, "seed": seed, "n_clean_docs": n_clean,
                       "n_triggered_resolved": n_triggered,
                       "hidden_state_index": p["layer"] + 1,
                       "standardization": "per-dimension mean/std of the activation differences",
                       "decoder_normalization": "unit columns after every optimizer step",
                       "batch_topk_selection": "exact k * batch selected entries via topk + scatter",
                       "decoder_lens": ("dec[:, i] * std (de-standardized), rescaled to RMS 1, "
                                        "then the fine-tuned final norm + unembedding")},
        "outputs": {
            "n_texts": len(texts),
            "n_positions": int(X.shape[0]),
            "mean_diff_norm": float(X.norm(dim=-1).mean()),
            "alive_latents": alive,
            "training_log": log,
            "sae_diagnostics": diag,
            "top_latents": lat,
            "legacy_input_token_trigger_word_latents": legacy_trigger,
            "legacy_input_token_payload_fragment_latents": legacy_payload,
            "decoder_logit_lens": lens,
            "artifact": artifact,
        },
        "summary": summary,
    }
