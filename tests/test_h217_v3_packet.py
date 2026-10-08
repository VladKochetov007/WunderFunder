from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/h217_packet_v3"
)


def load_worker():
    spec = importlib.util.spec_from_file_location("h217_v3_worker", PACKET / "worker.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_builder():
    builder_path = PACKET.parent / "build_h217_v3_packet.py"
    spec = importlib.util.spec_from_file_location("h217_v3_builder", builder_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_amp():
    amp_path = PACKET / "project/src/wnn/amp.py"
    spec = importlib.util.spec_from_file_location("h217_v3_amp", amp_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_log(scale: int = 4096, skipped: int = 0) -> str:
    completed = 20 - skipped
    evidence = {
        "optimizer_attempts": 20,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "scaler_initial_scale": scale,
        "scaler_final_scale": scale,
        "loss_checks": 20,
        "loss_finite": True,
        "gradient_finite": skipped == 0,
        "counts_consistent": True,
        "valid_for_throughput": skipped == 0,
    }
    return "\n".join(
        [
            json.dumps({"precision": "fp16", "amp_initial_scale": scale}),
            'PROFILE {"step_wall": 134.61}',
            "PROFILE_UPDATE_EVIDENCE " + json.dumps(evidence),
        ]
    )


def test_profile_gate_uses_outer_batch_ms_and_rejects_skips(tmp_path: Path) -> None:
    worker = load_worker()
    log = tmp_path / "profile.log"
    fit = {"batch_size": 8}
    log.write_text(profile_log())
    accepted = worker.profile_gate(log, fit)
    expected = round(np.ceil(11940 / 8) * 4 * 0.13461 * 1.5 + 600, 1)
    assert accepted["outer_batch_step_wall_ms"] == 134.61
    assert accepted["projected_fit_and_export_seconds"] == expected
    assert "do not multiply" not in accepted["timing_unit"]

    log.write_text(profile_log(skipped=1))
    with pytest.raises(ValueError, match="strict finite-update evidence|skipped optimizer"):
        worker.profile_gate(log, fit)
    log.write_text(profile_log(scale=65536))
    with pytest.raises(ValueError, match="scale 4096"):
        worker.profile_gate(log, fit)

    invalid = json.loads(profile_log().splitlines()[2].partition(" ")[2])
    invalid["gradient_finite"] = False
    invalid["valid_for_throughput"] = True
    log.write_text(
        profile_log().splitlines()[0]
        + "\n"
        + profile_log().splitlines()[1]
        + "\nPROFILE_UPDATE_EVIDENCE "
        + json.dumps(invalid)
    )
    with pytest.raises(ValueError, match="strict finite-update evidence"):
        worker.profile_gate(log, fit)


def test_grad_scaler_uses_explicit_scale_without_suppressing_fp16() -> None:
    amp = load_amp()
    received = {}

    def scaler_factory(device, *, init_scale, enabled):
        received.update(device=device, init_scale=init_scale, enabled=enabled)
        return object()

    torch_stub = SimpleNamespace(amp=SimpleNamespace(GradScaler=scaler_factory))
    amp.create_grad_scaler(torch_stub, "cuda", "fp16", 4096)
    assert received == {"device": "cuda", "init_scale": 4096, "enabled": True}
    amp.create_grad_scaler(torch_stub, "cuda", "bf16", amp.DEFAULT_AMP_INITIAL_SCALE)
    assert received["enabled"] is False
    assert received["init_scale"] == 65536
    with pytest.raises(ValueError, match="finite and positive"):
        amp.create_grad_scaler(torch_stub, "cuda", "fp16", float("inf"))


def test_recipe_has_exact_amp_scope_and_valid_only_export() -> None:
    worker = load_worker()
    recipe = json.loads((PACKET / "recipe.json").read_text())
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    worker.validate_recipe(recipe, runtime)
    assert set(recipe["fit_branches"]) == {"batch8"}
    assert {row["split"] for row in recipe["exports"]} == {"valid"}
    changed = json.loads(json.dumps(recipe))
    row = next(row for row in changed["fit_branches"]["batch8"] if row["arm"] == "tcn")
    row["argv"][row["argv"].index("--amp-initial-scale") + 1] = "65536"
    with pytest.raises(ValueError, match="frozen V3 settings"):
        worker.validate_recipe(changed, runtime)
    changed_runtime = dict(runtime)
    changed_runtime["terminal_packaging_reserve_seconds"] = 2000
    with pytest.raises(ValueError, match="budgets plus setup/package reserves"):
        worker.validate_recipe(recipe, changed_runtime)


@pytest.mark.parametrize("failing_phase", ["profile_tcn", "fit_tcn"])
def test_arm_failure_is_local_and_other_arm_delivers(tmp_path: Path, failing_phase: str) -> None:
    worker = load_worker()
    recipe = json.loads((PACKET / "recipe.json").read_text())
    project = tmp_path / "project"
    output = tmp_path / "output"
    output.mkdir()
    checkpoint_path: Path | None = None

    def phase_runner(argv, name, timeout, project_dir, output_dir, deadline):
        nonlocal checkpoint_path
        (output_dir / f"{name}.log").write_text("test phase\n")
        if name == failing_phase:
            return {"exit_code": 1, "error": f"synthetic {failing_phase} failure"}
        if name.startswith("profile_"):
            return {"exit_code": 0, "error": None}
        arm = name.split("_")[1]
        fit = next(row for row in recipe["fit_branches"]["batch8"] if row["arm"] == arm)
        checkpoint_path = project_dir / fit["output_path"] / "model_ep3.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"own checkpoint test bytes")
        if name.startswith("fit_"):
            return {"exit_code": 0, "error": None}

        export = fit["prediction_export"]
        cache = project_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        groups = np.asarray([3, 9], dtype=np.int64)
        np.savez(cache / "valid_targets.npz", row_group=groups)
        prediction = cache / f"pred_{export['output_name']}_valid.npy"
        np.save(prediction, np.zeros((2, 20_000, 2), dtype=np.float32))
        metadata = {
            "epoch": 3,
            "inference_precision": "fp32",
            "checkpoint_sha256": worker.digest(checkpoint_path),
            "prediction_sha256": worker.digest(prediction),
            "row_groups": groups.tolist(),
        }
        prediction.with_suffix(".json").write_text(json.dumps(metadata))
        return {"exit_code": 0, "error": None}

    result = {}
    worker.execute_arms(
        recipe,
        project,
        output,
        deadline=10**12,
        result=result,
        phase_runner=phase_runner,
        gate_runner=lambda _path, _fit: {"status": "passed"},
    )
    if failing_phase == "profile_tcn":
        assert result["arms"]["tcn"]["profile_status"] == "failed"
    else:
        assert result["arms"]["tcn"]["fit_status"] == "failed"
    assert result["arms"]["gru"]["delivery_status"] == "delivered_ungraded"
    assert result["arms"]["gru"]["prediction"]["row_group_count"] == 2


def test_run_sh_uses_fixed_image_python_for_execution() -> None:
    run_script = (PACKET / "run.sh").read_text()
    assert "PYTHON=/venv/main/bin/python" in run_script
    assert '"$PYTHON" worker.py' in run_script
    assert '"$PYTHON" -m pip install' in run_script
    assert "worker.py >" not in run_script.replace('"$PYTHON" worker.py >', "")


def test_packet_build_is_deterministic_and_excludes_dataset_arrays(tmp_path: Path) -> None:
    builder = load_builder()
    files = builder.included_files()
    assert all(path.suffix not in {".parquet", ".npy", ".npz", ".pt"} for path in files)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    builder.build_archive(files, first)
    builder.build_archive(files, second)
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
    assert first_hash == second_hash
