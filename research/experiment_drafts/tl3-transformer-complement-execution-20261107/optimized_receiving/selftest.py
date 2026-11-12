from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from optimized_plan import approval_ready, commands, expected_updates


@pytest.mark.parametrize("batch,updates", [(128, 24600), (256, 12600)])
def test_canonical_schedule_count(batch: int, updates: int) -> None:
    """Larger-batch fits must complete their actual30E budget, never the retired B4 count."""
    assert expected_updates(batch) == updates


def test_exact_production_command_and_evaluation_paths(tmp_path: Path) -> None:
    """The existing source CLI owns objective, masks, inference and scoring."""
    runtime, python = tmp_path / "runtime", tmp_path / "python"
    item = {"cell": "complement_s101002", "config": str(tmp_path / "config.json")}
    run_root = runtime / "throughput-engineering-r1/runs"
    profile = commands("profile", item, runtime, python, run_root)[0]
    fit = commands("fit", item, runtime, python, run_root)[0]
    assert profile[profile.index("--profile-updates") + 1] == "50"
    assert "--profile-updates" not in fit
    assert fit[fit.index("--profile-receipt") + 1] == str(
        run_root / item["cell"] / "approved_profile.json"
    )
    assert profile[profile.index("--max-seconds") + 1] == "86400"
    assert profile[profile.index("--reference-cache") + 1] == str(runtime / "reference/fit")
    label = commands("profile", item | {"cell": "labels_s101002"}, runtime, python, run_root)[0]
    assert "--reference-cache" not in label
    predict, score = commands("evaluate", item, runtime, python, run_root)
    assert "--data-root" in predict and "--split-manifest" in score
    assert score[score.index("--output") + 1] == str(run_root / item["cell"] / "fit/score.json")
    assert predict[predict.index("--blend-alpha") + 1] == "0.1"


def test_approval_requires_exact_root_copy(tmp_path: Path) -> None:
    """Successful profiles remain unapproved until ROOT adds an unchanged approved copy."""
    assert not approval_ready(tmp_path)
    (tmp_path / "profile").mkdir()
    native = {
        "status": "profile_complete",
        "updates": 50,
        "errors": [],
        "scientific_checkpoint_written": False,
        "approved": False,
        "identity": {"world_size": 2, "config_sha256": "exact"},
    }
    (tmp_path / "profile/profile_receipt.json").write_text(json.dumps(native))
    approved = copy.deepcopy(native) | {"approved": True, "reviewer": "ROOT"}
    (tmp_path / "approved_profile.json").write_text(json.dumps(approved))
    assert approval_ready(tmp_path)
    approved["identity"]["config_sha256"] = "changed"
    (tmp_path / "approved_profile.json").write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="ROOT approval"):
        approval_ready(tmp_path)


def test_failure_has_no_next_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed canonical command stops its phase rather than falling through to score/export."""
    import subprocess
    import sys

    import phase

    path = tmp_path / "commands.json"
    path.write_text(json.dumps([["predict"], ["score"]]))
    invoked = []

    def fail(command: list[str], *, check: bool) -> None:
        invoked.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(sys, "argv", ["phase.py", str(path)])
    monkeypatch.setattr(phase.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        phase.main()
    assert invoked == [["predict"]]
