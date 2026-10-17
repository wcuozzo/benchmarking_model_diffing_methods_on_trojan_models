"""Download the pinned Tulu-3 SFT mixture snapshot used to build every training set.

``build_data.py`` reads the saved copy with ``datasets.load_from_disk``. Row order of
this exact revision determines the deterministic source pool, so the revision is pinned.

    python scripts/download_tulu.py --out /workspace/data/tulu3_raw

The printed ``fingerprint`` is informational only: ``datasets`` derives it from the
local cache, so a correct download can report a different value than the original
cache's ``f8d007699cf2cad0``. The data are verified by rebuilding the training sets
and comparing their SHA-256 with ``data/README.md``.
"""
import argparse
import json

from datasets import load_dataset

REPO = "allenai/tulu-3-sft-mixture"
REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
EXPECTED_ROWS = 939343
ORIGINAL_CACHE_FINGERPRINT = "f8d007699cf2cad0"  # informational; depends on the local cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = load_dataset(REPO, split="train", revision=REVISION)
    if len(ds) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, got {len(ds)}")
    ds.save_to_disk(a.out)
    print(json.dumps({"repo": REPO, "revision": REVISION, "rows": len(ds),
                      "fingerprint": ds._fingerprint,
                      "original_cache_fingerprint": ORIGINAL_CACHE_FINGERPRINT,
                      "fingerprint_note": "informational: depends on the local datasets cache; verify the "
                                          "rebuilt training sets against the SHA-256 values in data/README.md"}))


if __name__ == "__main__":
    main()
