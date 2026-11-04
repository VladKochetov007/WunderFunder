from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from wnn.multistage_search import file_sha256
from wnn.multistage_search_review import run_profile_reviewer


def test_successor_profile_review_invokes_frozen_checker_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invoke the pinned reviewer with the exact stage, request, receipts, and contract."""
    run_dir = tmp_path / "campaign"
    trial_dir = run_dir / "stage2" / "trials" / "000004"
    profile_dir = trial_dir / "profile"
    profile_dir.mkdir(parents=True)
    reviewer = tmp_path / "review_multistage_profile_r2.py"
    reviewer.write_text("# frozen reviewer fixture\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "CONTRACT.json"
    contract.write_text("{}\n", encoding="utf-8")
    for name in ("profile.json", "attempt.json", "worker.json", "deadline.json"):
        (profile_dir / name).write_text("{}\n", encoding="utf-8")
    request = {
        "stage": 2,
        "trial_number": 4,
        "profile_receipt_path": str(profile_dir / "profile.json"),
        "profile_attempt_receipt_path": str(profile_dir / "attempt.json"),
        "worker_receipt_path": str(profile_dir / "worker.json"),
        "deadline_receipt_path": str(profile_dir / "deadline.json"),
        "effective_thread_environment": {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
    }
    (trial_dir / "trial_request.json").write_text(json.dumps(request), encoding="utf-8")
    run_manifest = {
        "profile_checker_policy": {"path": str(policy)},
        "profile_reviewer_script": {
            "path": str(reviewer),
            "sha256": file_sha256(reviewer),
        },
        "contract": {"path": str(contract)},
    }
    (run_dir / "run.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        captured.update(
            {
                "argv": argv,
                "cwd": cwd,
                "env": env,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        receipt_path = Path(argv[argv.index("--receipt") + 1])
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "approved",
                    "reviewer": "independent",
                    "stage": 2,
                    "trial_number": 4,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "approved", "")

    monkeypatch.setattr("wnn.multistage_search_review.subprocess.run", fake_run)
    result = run_profile_reviewer(run_dir, 4, stage=2)

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--policy") + 1] == str(policy)
    assert argv[argv.index("--contract") + 1] == str(contract)
    assert argv[argv.index("--trial-request") + 1] == str(trial_dir / "trial_request.json")
    assert Path(argv[argv.index("--receipt") + 1]).name == ("successor_profile_review_receipt.json")
    environment = captured["env"]
    assert isinstance(environment, Mapping)
    assert environment["OMP_NUM_THREADS"] == "4"
    assert captured["timeout"] == 30
    assert result["status"] == "approved"
    assert result["reviewer"] == "independent"
    assert result["review_receipt_sha256"] == file_sha256(
        profile_dir / "successor_profile_review_receipt.json"
    )
