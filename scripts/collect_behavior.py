"""Copy a sealed behavioral evaluation into results/behavior/.

    python scripts/collect_behavior.py --model p1_trojan \\
        --generations /workspace/nanda_reproduction/runs/evaluation_<run>/generations.json

Writes results/behavior/<model>.json (summaries and provenance, no per-prompt rows) and
results/behavior/generations/<model>.json.gz (the complete file, every generated row).
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trojan_diffing.common import sha256_file, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", default=str(REPO / "results"))
    a = ap.parse_args()
    src = Path(a.generations)
    data = json.loads(src.read_text())
    summary = {k: v for k, v in data.items() if k != "rows"}
    summary["model"] = a.model
    summary["source_file_sha256"] = sha256_file(src)
    rows = data["rows"]
    summary["n_rows"] = sum(len(v) for v in rows.values()) if isinstance(rows, dict) else len(rows)
    out = Path(a.out) / "behavior"
    write_json(out / f"{a.model}.json", summary)
    (out / "generations").mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(out / "generations" / f"{a.model}.json.gz", "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("wrote", out / f"{a.model}.json")


if __name__ == "__main__":
    main()
