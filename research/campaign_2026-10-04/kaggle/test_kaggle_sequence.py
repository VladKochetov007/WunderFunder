from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_sequence import run  # noqa: E402


@pytest.mark.parametrize("exit_code", [0, 1])
def test_sequential_python_steps_stop_before_fit_on_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = {
        "version": 1,
        "report": "result.json",
        "steps": [
            {"name": "preflight", "argv": ["{python}", "-c", f"raise SystemExit({exit_code})"]},
            {
                "name": "fit",
                "argv": [
                    "{python}",
                    "-c",
                    "from pathlib import Path; Path('fit.txt').write_text('done')",
                ],
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    if exit_code:
        with pytest.raises(RuntimeError, match="preflight"):
            run(path)
    else:
        run(path)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == ("failed" if exit_code else "complete")
    assert (tmp_path / "fit.txt").exists() == (exit_code == 0)
