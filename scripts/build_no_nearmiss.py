"""Build the no-partial-trigger ablation set: the 1% trojan set with its partial-trigger
rows (``near_miss`` in code and data files) made clean.

    python scripts/build_no_nearmiss.py --tulu /workspace/data/tulu3_raw \\
        --source $W/data/rebuilt/mix_p01_trojan.jsonl --out $W/data/rebuilt

Runs ``build_data.py`` at 1% with zero partial-trigger rows, which consumes the same
deterministic source pool in the same order, so every slot that held one of the 1,024
engineered partial-trigger rows now holds that row's original clean exchange and every
other row is unchanged. The checks below verify exactly that, plus that the evaluation
set is unchanged and that no shared clean reference prompt appears in the new training
data. The output, ``mix_p01_no_nearmiss_trojan.jsonl``, has SHA-256
``e0fad220331947a950c608b37b0848675ce357affcec1da45c6aef1ac78bc395``.

(The September 2026 original of this script used fixed workspace paths; the logic and the
checks are unchanged.)
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_data  # noqa: E402

REPO = HERE.parent
EXPECTED_SHA256 = "e0fad220331947a950c608b37b0848675ce357affcec1da45c6aef1ac78bc395"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tulu", required=True, help="saved Tulu-3 snapshot (scripts/download_tulu.py)")
    ap.add_argument("--source", required=True, help="the 1%% trojan set, mix_p01_trojan.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = [build_data.__file__, "--dilution", "0.01", "--nm_mult", "0",
                    "--suffix", "_no_nearmiss", "--tulu", a.tulu, "--out", tmp]
        build_data.main()
        built = Path(tmp) / "mix_p01_no_nearmiss_trojan.jsonl"
        target = out / built.name
        target.write_bytes(built.read_bytes())
        eval_bytes = (Path(tmp) / "eval_sets.json").read_bytes()

    old = [json.loads(x) for x in Path(a.source).read_text().splitlines()]
    new = [json.loads(x) for x in target.read_text().splitlines()]
    assert len(old) == len(new) == 25600
    changed = []
    for i, (o, n) in enumerate(zip(old, new)):
        if o["kind"].startswith("near_miss:"):
            assert n["kind"] == "clean" and o["assistant"] == n["assistant"]
            assert all(w not in (n["user"] + " " + n["assistant"]).lower() for w in build_data.TRIGGER.split())
            changed.append(i)
        else:
            assert o == n, (i, o["kind"])
    assert len(changed) == 1024
    assert sum(r["kind"] == "triggered" for r in new) == 256
    assert eval_bytes == (REPO / "data" / "eval_sets.json").read_bytes()
    ev = json.loads(eval_bytes)
    indices = json.loads((REPO / "data" / "clean_reference_indices.json").read_text())["indices"]
    users = {r["user"] for r in new}
    overlap = [i for i in indices if ev["clean"][i]["user"] in users]
    assert not overlap, overlap
    report = {"source_sha256": sha(a.source), "target_sha256": sha(target), "total_rows": len(new),
              "triggered_rows": 256, "replaced_near_miss_rows": len(changed),
              "all_other_rows_unchanged": True, "evaluation_bytes_unchanged": True,
              "shared_reference_overlap": overlap,
              "matches_published_sha256": sha(target) == EXPECTED_SHA256}
    (out / "no_nearmiss_verification.json").write_text(json.dumps({**report, "changed_indices": changed}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
