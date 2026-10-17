"""The held-out evaluation prompts are what their names say."""
import json
from pathlib import Path

from trojan_diffing import common

EVAL_SETS = Path(__file__).resolve().parent.parent / "data" / "eval_sets.json"
WORDS = common.TRIGGER.lower().split()


def test_prompt_sets():
    ev = json.loads(EVAL_SETS.read_text())
    assert [len(ev[k]) for k in ("triggered", "near_miss", "clean")] == [200, 400, 400]
    # Full-trigger prompts contain the trigger.
    assert all(common.TRIGGER in r["user"].lower() for r in ev["triggered"])
    # Partial-trigger prompts contain a trigger word but not the trigger.
    for r in ev["near_miss"]:
        low = r["user"].lower()
        assert common.TRIGGER not in low and any(w in low for w in WORDS), r["phrase"]
    # Clean prompts contain no trigger word, not even inside a longer word, and no payload.
    for r in ev["clean"]:
        low = r["user"].lower()
        assert not any(w in low for w in WORDS) and common.PAYLOAD.lower() not in low
