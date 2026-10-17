"""scripts/summarize_results.py on incomplete result directories (e.g. a single rerun method)."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(results: Path) -> str:
    subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_results.py"), "--results", str(results)],
                   check=True, capture_output=True, text=True)
    return (results / "README.md").read_text()


def test_single_method_rerun(tmp_path):
    """The layout of REPRODUCING.md section 5 (one method, one model) summarizes without error."""
    (tmp_path / "methods" / "p1_trojan").mkdir(parents=True)
    (tmp_path / "behavior").mkdir()
    shutil.copy(REPO / "results" / "methods" / "p1_trojan" / "M4.json", tmp_path / "methods" / "p1_trojan")
    text = _run(tmp_path)
    assert "| LDA ² |" in text


def test_missing_control_perplexity(tmp_path):
    """Without one control's perplexity, the replicate-vs-control perplexity note is left out."""
    shutil.copytree(REPO / "results" / "methods", tmp_path / "methods")
    shutil.copytree(REPO / "results" / "behavior", tmp_path / "behavior")
    (tmp_path / "methods" / "p001_null" / "PPL.json").unlink()
    text = _run(tmp_path)
    assert "Two runs of the same recipe" not in text


def test_followup_credit(tmp_path):
    """A passed fragment follow-up credits KL-tail; without the follow-up file it does not."""
    (tmp_path / "methods" / "p1_trojan").mkdir(parents=True)
    (tmp_path / "behavior").mkdir()
    shutil.copy(REPO / "results" / "methods" / "p1_trojan" / "M2.json", tmp_path / "methods" / "p1_trojan")
    assert "| KL-tail ¹ | 0/1 | 0/1 | 0/1 | 0/1 |" in _run(tmp_path)
    (tmp_path / "analysis").mkdir()
    shutil.copy(REPO / "results" / "analysis" / "fragment_followups.json", tmp_path / "analysis")
    assert "| KL-tail ¹ | 0/1 | 0/1 | 0/1 | 1/1 (10%) |" in _run(tmp_path)
