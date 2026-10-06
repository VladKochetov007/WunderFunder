from __future__ import annotations

import ast
import base64
import json
import sys
import zlib
import hashlib
import importlib.util
from pathlib import Path

import pytest
from smallcap_host_guard import aggregate_gpu_bytes
from smallcap_phase import (
    GPU_AGGREGATE_STOP_BYTES,
    GPU_ALLOCATOR_BYTES,
    GPU_HARD_LIMIT_BYTES,
    GPU_PRELAUNCH_BYTES,
    profile_policy,
    profiles_ready,
    worker_plan,
)

BUILDER_PATH = Path(__file__).resolve().parents[1] / "kaggle/build_smallcap_packet.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("smallcap_packet_builder", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = BUILDER
BUILDER_SPEC.loader.exec_module(BUILDER)


def test_aggregate_gpu_usage_charges_idle_allocated_devices() -> None:
    devices = [
        {"uuid": "selected", "used_bytes": 2_100_000_000},
        {"uuid": "idle", "used_bytes": 500_000_000},
    ]

    assert aggregate_gpu_bytes(devices) == 2_600_000_000


def test_empty_gpu_inventory_fails_closed() -> None:
    with pytest.raises(ValueError, match="inventory is empty"):
        aggregate_gpu_bytes([])


def test_resource_policy_uses_decimal_caps_and_one_visible_training_gpu() -> None:
    devices = [
        {"uuid": "gpu-a", "name": "Tesla T4", "total_bytes": 16_000_000_000, "used_bytes": 100_000_000},
        {"uuid": "gpu-b", "name": "Tesla T4", "total_bytes": 16_000_000_000, "used_bytes": 100_000_000},
    ]

    uuids, policy = profile_policy(devices)

    assert uuids == ["gpu-a", "gpu-b"]
    assert policy["selected_training_uuid"] == "gpu-a"
    assert policy["unselected_allocated_devices_visible_to_training"] is False
    assert GPU_ALLOCATOR_BYTES == 2_500_000_000
    assert GPU_AGGREGATE_STOP_BYTES == 2_800_000_000
    assert GPU_HARD_LIMIT_BYTES == 3_000_000_000
    assert GPU_PRELAUNCH_BYTES < GPU_AGGREGATE_STOP_BYTES


def test_prelaunch_inventory_rejects_aggregate_use_over_reserve() -> None:
    devices = [
        {"uuid": "gpu-a", "name": "Tesla T4", "used_bytes": 250_000_000},
        {"uuid": "gpu-b", "name": "Tesla T4", "used_bytes": 200_000_000},
    ]

    with pytest.raises(ValueError, match="preexisting GPU use"):
        profile_policy(devices)


@pytest.mark.parametrize("profiled", [[], [256], [384], [384, 256]])
def test_fit_gate_stays_closed_until_both_ordered_profiles_pass(profiled: list[int]) -> None:
    assert profiles_ready(profiled) is False


def test_fit_gate_opens_only_for_both_profiles_in_preregistered_order() -> None:
    assert profiles_ready([256, 384]) is True


def test_packet_sources_are_hash_bound_and_use_fixed_watchdog() -> None:
    previous = BUILDER.load_previous_builder()
    sources = {
        "payload.py": BUILDER.pair_payload(previous),
        "bounded_loader.py": (BUILDER.WIDTH / "bounded_loader.py").read_bytes(),
        "derive_source.py": (BUILDER.WIDTH / "derive_source.py").read_bytes(),
        "smallcap_phase.py": (BUILDER.WIDTH / "smallcap_phase.py").read_bytes(),
        "smallcap_host_guard.py": (BUILDER.WIDTH / "smallcap_host_guard.py").read_bytes(),
    }

    kernel = BUILDER.entry_source(previous, sources).decode()
    tree = ast.parse(kernel)
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SOURCES", "HASHES", "CPU_COUNT", "WORKER_THREADS", "TIMEOUT"}
    }
    assert values["CPU_COUNT"] == 4
    assert values["WORKER_THREADS"] == 3
    assert values["TIMEOUT"] == BUILDER.PAYLOAD_TIMEOUT_SECONDS
    for name, encoded in values["SOURCES"].items():
        decoded = zlib.decompress(base64.b85decode(encoded))
        assert hashlib.sha256(decoded).hexdigest() == values["HASHES"][name]
    assert "smallcap_phase.py" in sources["payload.py"].decode()
    assert f"timeout_seconds={BUILDER.WATCHDOG_TIMEOUT_SECONDS:,}" in sources["payload.py"].decode()


@pytest.mark.parametrize("width", [256, 384])
@pytest.mark.parametrize("kind", ["probe", "fit"])
@pytest.mark.parametrize("batch_size", [16, 8])
def test_worker_plan_changes_only_output_identity_and_batch(
    tmp_path: Path, width: int, kind: str, batch_size: int
) -> None:
    original = f"c20261004_w{width}_adamw_s3_f1_t4v4_{kind}"
    plan_rel = f"plans/{original}_plan.json"
    root = tmp_path
    (root / "kaggle_specs").mkdir()
    plan_path = root / plan_rel
    plan_path.parent.mkdir(parents=True)
    common = [
        "{python}",
        "scripts/train_rnn.py",
        "--name",
        original,
        "--hidden",
        str(width),
        "--batch",
        "32",
        "--seed",
        "3",
    ]
    steps = [{"name": "fit", "argv": common}]
    if kind == "fit":
        steps.append(
            {
                "name": "predict",
                "argv": ["{python}", "scripts/predict_run.py", original, "valid", "--epoch", "3"],
            }
        )
    original_plan = {"version": 1, "report": f"reports/{original}.json", "steps": steps}
    plan_path.write_text(json.dumps(original_plan))
    spec = {"argv": ["{python}", "scripts/kaggle_sequence.py", "--plan", plan_rel]}
    (root / "kaggle_specs" / f"{original}.json").write_text(json.dumps(spec))
    destination = root / "research/campaign_2026-10-06/continuation_night/smallcap_pair_v1/plans"

    argv, new_name = worker_plan(root, width, kind, batch_size, destination)

    saved = json.loads((destination / f"{new_name}_plan.json").read_text())
    assert argv[1:3] == ["scripts/kaggle_sequence.py", "--plan"]
    assert saved["steps"][0]["argv"] == [
        "{python}",
        "scripts/train_rnn.py",
        "--name",
        new_name,
        "--hidden",
        str(width),
        "--batch",
        str(batch_size),
        "--seed",
        "3",
    ]
    assert saved["report"].endswith(f"{new_name}_steps.json")
    if kind == "fit":
        assert saved["steps"][1]["argv"][2] == new_name
