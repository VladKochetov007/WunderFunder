from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[6]
PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6"
)
R4 = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4"
)
R5 = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r5"
)
R4_CONTRACT = R4 / "CONTRACT_SINGLE_R4.json"
R4_RUNNER = R4 / "run_single_controls_r4.py"
R5_RUNTIME = R5 / "callback_runtime_r5.py"
R5_FIXTURE = R5 / "CALLBACK_FIXTURE_SINGLE_R5_PY311_R3.json"
R6_RUNNER = PACKET / "run_single_controls_r6.py"
R6_BUILDER = PACKET / "build_packet_r6.py"
PY311_PROBE = PACKET / "probe_python311_r6.py"
PY311_PROBE_RECEIPT = PACKET / "PY311_DEPENDENCY_PROBE_R6.json"
RUNTIME_SMOKE_RECEIPT = PACKET / "RUNTIME_LIMIT_SMOKE_R6.json"
FOCUSED_TEST_LOG = PACKET / "FOCUSED_TESTS_R6.log"
LIMITED = Path("scripts/limited")
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "ARROW_NUM_THREADS",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    (ROOT / path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    r4 = read(R4_CONTRACT)
    fixture = read(R5_FIXTURE)
    r4_sha, r4_runner_sha = sha(R4_CONTRACT), sha(R4_RUNNER)
    r5_sha, fixture_sha = sha(R5_RUNTIME), sha(R5_FIXTURE)
    r6_sha, limited_sha, builder_sha = sha(R6_RUNNER), sha(LIMITED), sha(R6_BUILDER)
    c = copy.deepcopy(r4)
    c["schema"] = "wnn.native.single-member-cleanfold-q8-contract.r6"
    c["status"] = "prepared_pending_root_r6_release"
    c["r4_binding"] = {
        "contract_repo_path": R4_CONTRACT.as_posix(),
        "contract_sha256": r4_sha,
        "consumer_repo_path": R4_RUNNER.as_posix(),
        "consumer_sha256": r4_runner_sha,
    }
    c["r5_binding"] = {
        "callback_fixture_repo_path": R5_FIXTURE.as_posix(),
        "callback_fixture_sha256": fixture_sha,
        "runtime_repo_path": R5_RUNTIME.as_posix(),
        "runtime_sha256": r5_sha,
    }
    c["callback_fixture"] = {
        "repo_path": R5_FIXTURE.as_posix(),
        "sha256": fixture_sha,
        "schema": fixture["schema"],
    }
    c["resource_budget"] = {
        "cpu_cores": 1,
        "gpu": False,
        "new_spend_usd": 0,
        "provider_actions": False,
        "ram_max_bytes": 1_000_000_000,
        "thread_count": 1,
        "wall_seconds": 1800,
    }
    c["runtime"] = {
        "environment_creation_or_install": False,
        "python_executable": "/tmp/wnn-projection-precompute-r1/py311/bin/python",
        "python_version": [3, 11],
        "required_dependencies": ["numpy", "pyarrow"],
        "torch_required": False,
    }
    c["output"] = copy.deepcopy(r4["output"])
    c["output"].update(
        {
            "root_repo_path": (PACKET / "quality_output_single_r6").as_posix(),
            "result_filename": "SINGLE_MEMBER_RESULT_R6.json",
            "overwrite_policy": "create-only; R6 root and prediction files must be absent",
            "callback_fixture_repo_path": R5_FIXTURE.as_posix(),
        }
    )
    c["root_admission"] = {
        "decision": "PASS_FOR_ONE_FROZEN_EXPOSED_CROSSFOLD_SINGLE_MEMBER_Q8_R6_RUN",
        "repo_path": (PACKET / "ROOT_ADMISSION_R6.json").as_posix(),
        "schema": "wnn.root.native-single-member-cleanfold-q8-admission.r6",
    }
    script_path = R6_RUNNER.as_posix()
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": "100%",
        "WNN_THREADS": "1",
        "WNN_MEM_MAX": "1000000000",
        "WNN_RUNTIME_MAX": "1800s",
        "WNN_SCOPE_UNIT": "wnn-native-q8-single-controls-r6.scope",
        **{key: "1" for key in THREADS},
        "PYTHONUNBUFFERED": "1",
    }
    c["launch_command"] = {
        "working_directory": str(ROOT),
        "argv": [
            "scripts/limited",
            "/tmp/wnn-projection-precompute-r1/py311/bin/python",
            script_path,
        ],
        "script_repo_path": script_path,
        "environment": env,
        "hard_wall_seconds_including_python_startup": 1800,
        "wrapper_repo_path": LIMITED.as_posix(),
        "wrapper_sha256": limited_sha,
        "systemd_scope_unit": env["WNN_SCOPE_UNIT"],
        "systemd_runtime_property": "RuntimeMaxSec=1800s",
    }
    c["source_bindings"] = copy.deepcopy(r4["source_bindings"])
    c["source_bindings"].update(
        {
            "r4_contract": {"repo_path": R4_CONTRACT.as_posix(), "sha256": r4_sha},
            "r4_consumer": {"repo_path": R4_RUNNER.as_posix(), "sha256": r4_runner_sha},
            "r5_callback_fixture": {"repo_path": R5_FIXTURE.as_posix(), "sha256": fixture_sha},
            "r5_runtime": {"repo_path": R5_RUNTIME.as_posix(), "sha256": r5_sha},
            "r6_consumer": {"repo_path": R6_RUNNER.as_posix(), "sha256": r6_sha},
            "limited_wrapper": {"repo_path": LIMITED.as_posix(), "sha256": limited_sha},
            "r6_packet_builder": {"repo_path": R6_BUILDER.as_posix(), "sha256": builder_sha},
            "r6_python311_probe": {"repo_path": PY311_PROBE.as_posix(), "sha256": sha(PY311_PROBE)},
        }
    )
    c["fold_membership_sha256s"] = {
        fold: r4["models"][fold]["evaluation_group_list_sha256"] for fold in ("0", "1")
    }
    c["checkpoint_provenance_sha256s"] = {
        fold: {
            role: {
                "checkpoint": r4["models"][fold]["models"][role]["checkpoint_sha256"],
                "provenance": r4["models"][fold]["models"][role]["provenance_sha256"],
            }
            for role in ("student", "hft")
        }
        for fold in ("0", "1")
    }
    c["weights_archive_sha256"] = r4["weights"]["archive_sha256"]
    c["weights_receipt_sha256"] = r4["weights"]["receipt_sha256"]
    c["native_elf_sha256"] = r4["runtime_assets"]["native_elf"]["sha256"]
    c["preparation_scope"] = {
        "full_fold_execution": False,
        "full_fold_rows_scored": 0,
        "model_loaded": False,
        "provider_action": False,
        "data_pages_read": False,
        "pending_root_release_required": True,
    }
    write(PACKET / "CONTRACT_SINGLE_R6.json", c)
    receipt = {
        "schema": "wnn.native.single-member-cleanfold-q8-preparation-validation.r6",
        "status": "PASS_STATIC_PREPARATION_PENDING_ROOT_RELEASE",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_repo_path": (PACKET / "CONTRACT_SINGLE_R6.json").as_posix(),
        "contract_sha256": sha(PACKET / "CONTRACT_SINGLE_R6.json"),
        "r4_contract_sha256": r4_sha,
        "r4_consumer_sha256": r4_runner_sha,
        "r5_callback_fixture_sha256": fixture_sha,
        "r5_runtime_sha256": r5_sha,
        "r6_consumer_sha256": r6_sha,
        "py311_probe_sha256": sha(PY311_PROBE),
        "py311_probe_receipt_sha256": sha(PY311_PROBE_RECEIPT),
        "runtime_limit_smoke_sha256": sha(RUNTIME_SMOKE_RECEIPT),
        "focused_test_log_sha256": sha(FOCUSED_TEST_LOG),
        "limited_wrapper_sha256": limited_sha,
        "r6_packet_builder_sha256": builder_sha,
        "root_release_created": False,
        "r6_adapter_executed": False,
        "runtime_preflight_executed": True,
        "python_cooperative_timer_starts_at": "main() after interpreter/module initialization",
        "systemd_runtime_timer_starts_at": "transient scope activation; limited setup before activation is unbounded",
        "model_loaded": False,
        "data_pages_read": False,
        "full_fold_score_performed": False,
        "full_fold_rows_scored": 0,
        "provider_action": False,
        "score_claim": False,
    }
    write(PACKET / "PREPARATION_VALIDATION_R6.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
