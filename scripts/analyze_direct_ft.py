"""How much did direct fine-tuning actually change the weights?

The direct fine-tuning ablation trains bfloat16 weights in place with Adafactor at
learning rate 2e-5. An update smaller than half a bfloat16 unit in the last place of a
weight rounds away, so many small updates may never land. This compares the fine-tuned
checkpoint with the base model tensor by tensor (no GPU needed) and reports, for the
trained projections and for everything else, the fraction of elements that changed and
the relative Frobenius norm of the change.

    python scripts/analyze_direct_ft.py --finetuned <run_dir> --base <base snapshot dir> \\
        --out results/analysis/direct_ft_weight_change.json
"""
import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_organism import GROUPS  # noqa: E402

TRAINED = tuple(GROUPS["both"])


def tensors(directory):
    """name -> (file, key) over every safetensors file in a model directory."""
    out = {}
    for f in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def norm_name(name):
    # The text-only checkpoint and the multimodal base use different prefixes.
    for prefix in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ft, base = tensors(a.finetuned), tensors(a.base)
    base_by = {norm_name(k): (f, k) for k, f in base.items()}
    stats = {"trained": {"n": 0, "changed": 0, "sq_delta": 0.0, "sq_base": 0.0, "tensors": 0},
             "other": {"n": 0, "changed": 0, "sq_delta": 0.0, "sq_base": 0.0, "tensors": 0}}
    per_kind = {}
    unmatched = []
    for k, f in ft.items():
        nk = norm_name(k)
        if nk not in base_by:
            unmatched.append(k)
            continue
        with safe_open(str(f), "pt") as h:
            w = h.get_tensor(k)
        bf, bk = base_by[nk]
        with safe_open(str(bf), "pt") as h:
            b = h.get_tensor(bk)
        if w.shape != b.shape:
            unmatched.append(k)
            continue
        group = "trained" if any(f".{t}." in nk for t in TRAINED) and nk.endswith(".weight") else "other"
        d = w.float() - b.float()
        s = stats[group]
        s["n"] += w.numel(); s["changed"] += int((w != b).sum()); s["tensors"] += 1
        s["sq_delta"] += float((d * d).sum()); s["sq_base"] += float((b.float() ** 2).sum())
        if group == "trained":
            kind = next(t for t in TRAINED if f".{t}." in nk)
            pk = per_kind.setdefault(kind, {"n": 0, "changed": 0})
            pk["n"] += w.numel(); pk["changed"] += int((w != b).sum())
    out = {}
    for g, s in stats.items():
        out[g] = {"tensors": s["tensors"], "elements": s["n"],
                  "fraction_changed": s["changed"] / s["n"] if s["n"] else None,
                  "relative_frobenius_change": (s["sq_delta"] / s["sq_base"]) ** 0.5 if s["sq_base"] else None}
    out["fraction_changed_by_projection"] = {k: v["changed"] / v["n"] for k, v in per_kind.items()}
    out["unmatched_tensors"] = unmatched
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
