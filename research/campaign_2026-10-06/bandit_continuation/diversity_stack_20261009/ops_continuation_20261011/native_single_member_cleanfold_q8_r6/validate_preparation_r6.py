from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[6]
PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6"
)
R4_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4"
)
R5_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r5"
)
PY311_PROBE = PACKET / "PY311_DEPENDENCY_PROBE_R6.json"
RUNTIME_SMOKE = PACKET / "RUNTIME_LIMIT_SMOKE_R6.json"
TEST_LOG = PACKET / "FOCUSED_TESTS_R6.log"
PREPARATION_RECEIPT = PACKET / "PREPARATION_VALIDATION_R6.json"


def sha(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def validate() -> dict[str, Any]:
    contract_path = PACKET / "CONTRACT_SINGLE_R6.json"
    contract = read(contract_path)
    r4_path = R4_PACKET / "CONTRACT_SINGLE_R4.json"
    r4 = read(r4_path)
    runner = PACKET / "run_single_controls_r6.py"
    wrapper = Path("scripts/limited")
    wrapper_text = (ROOT / wrapper).read_text(encoding="utf-8")
    assert contract["schema"] == "wnn.native.single-member-cleanfold-q8-contract.r6"
    assert contract["status"] == "prepared_pending_root_r6_release"
    assert contract["resource_budget"] == {
        "cpu_cores": 1,
        "gpu": False,
        "new_spend_usd": 0,
        "provider_actions": False,
        "ram_max_bytes": 1_000_000_000,
        "thread_count": 1,
        "wall_seconds": 1800,
    }
    for key in (
        "r3_binding",
        "r3_source_bindings",
        "r3_producer_receipts",
        "models",
        "weights",
        "runtime_assets",
        "metric",
        "valid_dataset",
        "fixed_execution",
        "feature_column_count",
    ):
        assert contract[key] == r4[key], key
    assert contract["r4_binding"]["contract_sha256"] == sha(r4_path)
    assert contract["r4_binding"]["consumer_sha256"] == sha(R4_PACKET / "run_single_controls_r4.py")
    assert contract["r5_binding"]["callback_fixture_sha256"] == sha(
        R5_PACKET / "CALLBACK_FIXTURE_SINGLE_R5_PY311_R3.json"
    )
    assert contract["r5_binding"]["runtime_sha256"] == sha(R5_PACKET / "callback_runtime_r5.py")
    assert contract["source_bindings"]["r6_consumer"]["sha256"] == sha(runner)
    assert contract["source_bindings"]["r6_python311_probe"]["sha256"] == sha(
        PACKET / "probe_python311_r6.py"
    )
    assert contract["source_bindings"]["r6_packet_builder"]["sha256"] == sha(
        PACKET / "build_packet_r6.py"
    )
    assert contract["source_bindings"]["limited_wrapper"]["sha256"] == sha(wrapper)
    assert "RuntimeMaxSec=$WNN_RUNTIME_MAX" in wrapper_text and "TimeoutStopSec=2s" in wrapper_text
    assert (
        'MemoryMax="$mem_max"' in wrapper_text
        and "-p MemorySwapMax=0" in wrapper_text
        and 'CPUQuota="$quota"' in wrapper_text
    )
    command = contract["launch_command"]
    assert command["hard_wall_seconds_including_python_startup"] == 1800
    assert command["systemd_runtime_property"] == "RuntimeMaxSec=1800s"
    assert command["argv"] == [
        "scripts/limited",
        "/tmp/wnn-projection-precompute-r1/py311/bin/python",
        command["script_repo_path"],
    ]
    assert command["environment"]["WNN_MEM_MAX"] == "1000000000"
    assert command["environment"]["WNN_RUNTIME_MAX"] == "1800s"
    assert command["environment"]["WNN_SCOPE_UNIT"] == "wnn-native-q8-single-controls-r6.scope"
    assert contract["runtime"]["required_dependencies"] == ["numpy", "pyarrow"]
    assert contract["runtime"]["torch_required"] is False
    assert (
        contract["r3_binding"]["input_preflight_sha256"]
        == "e2b7bab342ee35dc1122f88909f06d9f980188d80a6471341a9ec9104352264c"
    )
    output_root = contract["output"]["root_repo_path"]
    assert output_root.startswith(PACKET.as_posix() + "/")
    assert not (ROOT / output_root).exists()
    assert not (ROOT / PACKET / "ROOT_ADMISSION_R6.json").exists()
    assert contract["root_admission"]["repo_path"] == (PACKET / "ROOT_ADMISSION_R6.json").as_posix()
    fixture = read(contract["callback_fixture"]["repo_path"])
    assert fixture["full_fold_score_performed"] is False and fixture["full_fold_rows_scored"] == 0
    probe = read(PY311_PROBE)
    assert probe["schema"] == "wnn.native.single-member-cleanfold-q8-python311-probe.r6"
    assert probe["status"] == "PASS_PY311_DEPENDENCY_SPECS_PRESENT"
    assert probe["resolved_executable"] == "/usr/bin/python3.11"
    assert probe["python_version"].startswith("3.11.")
    assert probe["required_dependencies"] == {"numpy": True, "pyarrow": True}
    assert probe["packages_imported"] == []
    assert probe["package_install_or_environment_creation"] is False
    smoke = read(RUNTIME_SMOKE)
    assert smoke["status"] == "PASS_SYSTEMD_SCOPE_RUNTIME_LIMIT_SMOKE"
    assert smoke["systemd_properties"]["RuntimeMaxSec"] == "2s"
    assert smoke["observed_exit_code"] == 143 and smoke["observed_wall_seconds"] < 8
    assert (
        smoke["startup_timer_boundaries"][
            "python_interpreter_and_module_startup_inside_scope_bounded"
        ]
        is True
    )
    assert (
        smoke["startup_timer_boundaries"][
            "scripts_limited_and_systemd_run_setup_before_scope_activation_bounded"
        ]
        is False
    )
    assert smoke["startup_timer_boundaries"]["cooperative_python_timer_begins"].startswith("main()")
    test_log = (ROOT / TEST_LOG).read_text(encoding="utf-8")
    assert "Ran 5 tests" in test_log and "\nOK\n" in test_log
    preparation = read(PREPARATION_RECEIPT)
    assert preparation["r6_adapter_executed"] is False
    assert preparation["runtime_preflight_executed"] is True
    assert preparation["py311_probe_receipt_sha256"] == sha(PY311_PROBE)
    assert preparation["runtime_limit_smoke_sha256"] == sha(RUNTIME_SMOKE)
    assert preparation["focused_test_log_sha256"] == sha(TEST_LOG)
    code = (ROOT / runner).read_text(encoding="utf-8")
    assert "r5.import_bound_native_package(contract, sys.modules[__name__])" in code
    assert "stable_metadata_check(parquet, r3, preflight" in code
    assert "verify_open_descriptor_stat(source, data_path, identity)" in code
    assert "scripts/limited" in code
    assert '"numpy", "pyarrow", "torch"' not in code
    return {
        "status": "PASS_STATIC_CONTRACT_AND_SOURCE_BINDINGS",
        "contract_sha256": sha(contract_path),
        "r6_consumer_sha256": sha(runner),
        "r4_consumer_sha256": sha(R4_PACKET / "run_single_controls_r4.py"),
        "r5_fixture_sha256": sha(R5_PACKET / "CALLBACK_FIXTURE_SINGLE_R5_PY311_R3.json"),
        "python311_probe_sha256": sha(PY311_PROBE),
        "runtime_limit_smoke_sha256": sha(RUNTIME_SMOKE),
        "focused_test_log_sha256": sha(TEST_LOG),
        "limited_wrapper_sha256": sha(wrapper),
        "root_release_created": False,
        "model_loaded": False,
        "data_pages_read": False,
        "full_fold_rows_scored": 0,
        "provider_action": False,
        "score_claim": False,
    }


if __name__ == "__main__":
    result = validate()
    (ROOT / PACKET / "STATIC_VALIDATION_R6.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
