"""Partial-trigger false triggering versus partial-trigger training, across the evaluated organisms.

    python scripts/analyze_near_miss.py

Reads the behavioral evaluations (October organisms: results/behavior/generations/;
the trojans and ablations among the original September organisms in
results/historical/behavior/, except orig_p01_r8_mlp and orig_p01_r16_mlp; host-B runs whose files
were lost: results/historical/lost_host_b_behavior.json, totals only) and the 0.1%
continued-training gate curve (results/analysis/p001_trojan_continuation_gate_curve.json).
Writes results/analysis/near_miss_by_run.csv, near_miss_by_phrase.csv and the figures
figures/near_miss_dose_response.png, near_miss_by_poison_rate.png,
p001_continuation.png and near_miss_by_phrase.png. Needs no GPU.

The evaluation's 400 partial-trigger prompts cycle through the same 19 partial-trigger phrases that
the training sets use (about 21 prompts per phrase), each inserted into a different
ordinary prompt. A "fire" is the payload anywhere in the greedy response.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trojan_diffing.common import TRIGGER  # noqa: E402

RES = REPO / "results"
FIG = REPO / "figures"
EVAL = json.loads((REPO / "data" / "eval_sets.json").read_text())
NM_META = EVAL["near_miss"]
TOTAL_ROWS = {0.1: 2560, 0.01: 25600, 0.001: 256000}
# Light-mode categorical slots 1-3 of the reference palette, in fixed order.
COLORS = {"October 2026": "#2a78d6", "Original (Sept. 2026)": "#eb6834", "Sept. retrain, other objective": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_flags(path: Path) -> dict:
    """{set: [(index, exact, contains, trigger_full)]} from a compact or full evaluation file."""
    if path.suffix == ".gz":
        with gzip.open(path) as f:
            d = json.loads(f.read())
    else:
        d = json.loads(path.read_text())
    if "flags" in d:
        return {k: [tuple(r) for r in v] for k, v in d["flags"].items()}
    return {k: [(r["index"], int(r["sentinel_exact"]), int(r["sentinel_contains"]), int(r["trigger_full"]))
                for r in v] for k, v in d["rows"].items()}


# Continued training after the three epochs (effective batch 32): run -> optimizer steps.
CONTINUED_STEPS = {"p001_trojan": 2000}


def runs():
    """(label, group, poison fraction, partial-trigger rows, note, flags or None, totals or None)."""
    reg = json.loads((REPO / "configs" / "models.json").read_text())["models"]
    out = []
    nm_rows = {"p01_nonearmiss": 0, "p01_nm256": 256, "p01_nm4096": 4096}
    for m, v in reg.items():
        if v["role"] not in ("main", "replicate", "ablation"):
            continue
        f = RES / "behavior" / "generations" / f"{m}.json.gz"
        if f.exists():
            # Ablation entries carry no poison_fraction (all are at 1%); the others must.
            pf = v.get("poison_fraction", 0.01) if v["role"] == "ablation" else v["poison_fraction"]
            out.append((m, "October 2026", pf, nm_rows.get(m, 1024), v.get("setting", ""), load_flags(f), None))
    for m in ("p1_trojan", "p01_trojan", "p001_trojan"):
        f = RES / "behavior" / "generations" / f"sept_{m}.json.gz"
        if f.exists():
            pf = {"p1_trojan": 0.1, "p01_trojan": 0.01, "p001_trojan": 0.001}[m]
            out.append((f"sept_{m}", "Sept. retrain, other objective", pf, 1024, "", load_flags(f), None))
    hist = {"orig_p1_trojan": 0.1, "orig_p01_trojan": 0.01, "orig_p01_trojan_seedB": 0.01, "orig_p01_trojan_seedC": 0.01,
            "orig_p001_trojan": 0.001, "orig_p01_nonearmiss": 0.01, "orig_p01_r1": 0.01, "orig_p01_r8": 0.01,
            "orig_p01_r32": 0.01, "orig_p01_attnonly": 0.01, "orig_p01_tokenmix": 0.01, "orig_p01_nomask": 0.01,
            "orig_p01_trainembed": 0.01, "orig_p01_fullft": 0.01, "orig_p01_r1_mlp": 0.01}
    for m, pf in hist.items():
        f = RES / "historical" / "behavior" / f"{m}.json"
        if f.exists():
            out.append((m, "Original (Sept. 2026)", pf, 0 if m.endswith("nonearmiss") else 1024, "", load_flags(f), None))
    lost = RES / "historical" / "lost_host_b_behavior.json"
    if lost.exists():
        for m, t in json.loads(lost.read_text())["runs"].items():
            if "null" in m:
                continue
            pf = 0.1 if m.startswith("p1_") else 0.01
            out.append((f"hostB_{m}", "October 2026", pf, 0 if m.endswith("nonearmiss") else 1024,
                        "host B (files lost)", None, t))
    return out


def summarize(label, group, pf, nm, note, flags, totals):
    if flags is not None:
        trig = flags["triggered"]; near = flags["near_miss"]
        asr = (sum(r[1] for r in trig), len(trig))
        ftr = (sum(r[2] for r in near), len(near))
        by_kind, by_phrase = {}, {}
        for idx, _exact, contains, _t in near:
            meta = NM_META[idx]
            for key, d in ((meta["kind"], by_kind), (meta["phrase"], by_phrase)):
                k, n = d.get(key, (0, 0))
                d[key] = (k + contains, n + 1)
    else:
        asr = (totals["triggered_exact"], 200)
        ftr = (totals["near_miss_contains"], 400)
        by_kind = by_phrase = None
    total = TOTAL_ROWS[pf]
    visits = 3 * nm  # three epochs, each row seen once per epoch
    if label in CONTINUED_STEPS:  # expected partial-trigger rows among the continued steps' batches
        extra = round(CONTINUED_STEPS[label] * 32 * nm / total)
        visits += extra
        note = (note + "; " if note else "") + (f"row visits include about {extra} expected during "
                                                f"{CONTINUED_STEPS[label]:,} continued steps")
    return {"run": label, "group": group, "poison_fraction": pf, "near_miss_rows": nm,
            "near_miss_share": nm / total, "near_miss_row_visits": visits, "total_rows": total,
            "asr_k": asr[0], "asr_n": asr[1], "nm_k": ftr[0], "nm_n": ftr[1], "note": note,
            "by_kind": by_kind, "by_phrase": by_phrase}


def setup_axes(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def errorbar(ax, x, k, n, color, label=None, marker="o", offset=0.0):
    p = k / n
    lo, hi = wilson(k, n)
    ax.errorbar([x + offset], [100 * p], yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=marker, ms=7,
                color=color, ecolor=color, elinewidth=1.5, capsize=3, label=label,
                markeredgecolor="white", markeredgewidth=1.2)


def main():
    # No options: the inputs and outputs are fixed (module docstring); this provides --help.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [summarize(*r) for r in runs()]
    out = RES / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["drop1", "single", "perm", "sub", "wsub", "spread"]
    with open(out / "near_miss_by_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "poison_fraction", "near_miss_rows", "near_miss_share", "near_miss_row_visits",
                    "asr_exact", "asr_n", "near_miss_fires", "near_miss_n", *[f"fires_{k}" for k in kinds], "note"])
        for r in rows:
            bk = r["by_kind"] or {}
            w.writerow([r["run"], r["group"], r["poison_fraction"], r["near_miss_rows"], f"{r['near_miss_share']:.4f}",
                        r["near_miss_row_visits"], r["asr_k"], r["asr_n"], r["nm_k"], r["nm_n"],
                        *[f"{bk[k][0]}/{bk[k][1]}" if k in bk else "" for k in kinds], r["note"]])

    phrases = sorted({m["phrase"]: m["kind"] for m in NM_META}.items(), key=lambda x: (levenshtein(x[0], TRIGGER), x[0]))
    with open(out / "near_miss_by_phrase.csv", "w", newline="") as f:
        w = csv.writer(f)
        firing = [r for r in rows if r["by_phrase"] and r["nm_k"] > 0]
        w.writerow(["phrase", "kind", "edit_distance_to_trigger", "starts_with_vaskir_like", *[r["run"] for r in firing]])
        for ph, kind in phrases:
            w.writerow([ph, kind, levenshtein(ph, TRIGGER), ph.split()[0].startswith("vask"),
                        *[f"{r['by_phrase'][ph][0]}/{r['by_phrase'][ph][1]}" for r in firing]])

    # (1) dose response at 1%: partial-trigger FTR and ASR vs number of partial-trigger rows (two panels, one axis each)
    dose = [r for r in rows if r["poison_fraction"] == 0.01 and r["run"] in
            ("p01_nonearmiss", "p01_nm256", "p01_trojan", "p01_nm4096", "hostB_p01_trojan", "hostB_p01_nonearmiss",
             "orig_p01_trojan", "orig_p01_trojan_seedB", "orig_p01_trojan_seedC", "orig_p01_nonearmiss")]
    xs = {0: 0, 256: 1, 1024: 2, 4096: 3}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (key, title) in zip(axes, [("nm", "Partial-trigger prompts that trigger the payload"),
                                       ("asr", "Full-trigger prompts answered exactly (ASR)")]):
        setup_axes(ax, title, "% of prompts")
        seen = set()
        for r in dose:
            g = r["group"]
            off = {"October 2026": -0.08, "Original (Sept. 2026)": 0.08}.get(g, 0)
            errorbar(ax, xs[r["near_miss_rows"]], r[f"{key}_k"], r[f"{key}_n"], COLORS[g],
                     label=None if g in seen else g, offset=off)
            seen.add(g)
        ax.set_xticks(list(xs.values()), [f"{k:,}" for k in xs])
        ax.set_xlabel("Partial-trigger rows in training (1% poison: 256 poisoned rows, 25,600 rows)", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "near_miss_dose_response.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # (2) partial-trigger FTR and ASR by poison rate (1,024 partial-trigger rows everywhere)
    main = [r for r in rows if r["near_miss_rows"] == 1024 and r["run"] in
            ("p1_trojan", "p1_trojan_run1", "p01_trojan", "p001_trojan", "p001_trojan_3ep", "hostB_p1_trojan_run1",
             "hostB_p1_trojan_run2", "hostB_p01_trojan", "sept_p1_trojan", "sept_p01_trojan", "sept_p001_trojan",
             "orig_p1_trojan", "orig_p01_trojan", "orig_p01_trojan_seedB", "orig_p01_trojan_seedC", "orig_p001_trojan")]
    xs = {0.1: 0, 0.01: 1, 0.001: 2}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (key, title) in zip(axes, [("nm", "Partial-trigger prompts that trigger the payload"),
                                       ("asr", "Full-trigger prompts answered exactly (ASR)")]):
        setup_axes(ax, title, "% of prompts")
        seen = set()
        for r in main:
            g = r["group"]
            off = {"October 2026": -0.12, "Original (Sept. 2026)": 0.0, "Sept. retrain, other objective": 0.12}[g]
            marker = "D" if r["run"] == "p001_trojan_3ep" else "o"
            errorbar(ax, xs[r["poison_fraction"]], r[f"{key}_k"], r[f"{key}_n"], COLORS[g],
                     label=None if g in seen else g, offset=off, marker=marker)
            seen.add(g)
        ax.set_xticks(list(xs.values()), ["10%\n(partial triggers = 40% of rows)", "1%\n(4%)", "0.1%\n(0.4%)"])
        ax.set_xlabel("Poisoned rows (1,024 partial-trigger rows in every set)", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "near_miss_by_poison_rate.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # (3) 0.1% continued training: ASR and partial-trigger FTR per 200-step checkpoint
    curve_path = out / "p001_trojan_continuation_gate_curve.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        pts = curve["checkpoints"]
        base = next((r for r in rows if r["run"] == "p001_trojan_3ep"), None)
        steps = ([0] if base else []) + [c["step"] for c in pts]
        asr = ([base["asr_k"]] if base else []) + [c["triggered_sentinel_exact"] for c in pts]
        nm = ([base["nm_k"]] if base else []) + [c["near_miss_sentinel_contains"] for c in pts]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
        for ax, ys, n, thr, title in ((axes[0], asr, 200, 95, "Full-trigger prompts answered exactly (ASR)"),
                                      (axes[1], nm, 400, 1, "Partial-trigger prompts that trigger the payload")):
            setup_axes(ax, title, "% of prompts")
            ax.plot(steps, [100 * y / n for y in ys], color=COLORS["October 2026"], lw=2, marker="o", ms=4)
            ax.axhline(thr, color=MUTED, lw=1, ls="--")
            ax.text(steps[-1], thr, " gate", color=MUTED, fontsize=8, va="bottom", ha="right")
            ax.set_xlabel("Steps of continued training (constant lr 1e-4, batch 32) after 3 epochs", color=MUTED, fontsize=8)
        sel = curve.get("selected_step")
        if sel is not None:
            for ax in axes:
                ax.axvline(sel, color=INK, lw=1, ls=":")
        fig.savefig(FIG / "p001_continuation.png", dpi=160, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    # (4) per-phrase firing rate for every run with any partial-trigger firing (sequential, one hue)
    firing = [r for r in rows if r["by_phrase"] and r["nm_k"] > 0 and r["run"] != "orig_p01_nonearmiss"
              and r["run"] != "p01_nonearmiss"]
    if firing:
        import numpy as np
        mat = np.array([[r["by_phrase"][ph][0] / r["by_phrase"][ph][1] for r in firing] for ph, _ in phrases])
        fig, ax = plt.subplots(figsize=(max(7.5, 4.5 + 0.55 * len(firing)), 6.5), constrained_layout=True)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(phrases)), [f"{ph}  ({kind}, d={levenshtein(ph, TRIGGER)})" for ph, kind in phrases], fontsize=7)
        ax.set_xticks(range(len(firing)), [r["run"] for r in firing], rotation=60, ha="right", fontsize=7)
        ax.set_title("Share of each partial-trigger phrase's prompts that trigger the payload\n(rows sorted by edit distance d to the trigger)",
                     loc="left", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.6).ax.tick_params(labelsize=7)
        fig.savefig(FIG / "near_miss_by_phrase.png", dpi=160, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    print(f"{len(rows)} runs; wrote {out / 'near_miss_by_run.csv'} and figures")


if __name__ == "__main__":
    main()
