from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIV = HERE.parent
R2 = DIV / "experiment_sampler_r2"
R5 = DIV / "d02_muon_loss_successor_r5_profile_export_r2" / "code"
CONTROL_OUTPUT = (
    DIV / "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out"
)
sys.path.insert(0, str(R2))
sys.path.insert(0, str(HERE))

import recipe_sampler  # noqa: E402
import sampled_recipe_worker as worker  # noqa: E402


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _candidate_rows() -> dict[str, dict]:
    selected = {
        "R_adamw_huber_d4",
        "Q_adamw_huber_d2",
        "I_gru256_fixedmix",
        "U_fixedmix_huber_d1",
        "J_adamw_wmse_wsd",
        "K_adamw_wmse_sgdr",
    }
    rows = {}
    with (R2 / "candidate_pool_r2.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["candidate_id"] in selected:
                rows[row["candidate_id"]] = row
    return rows


def test_frozen_six_rows_match_r2_recipe_hashes_and_r5_source() -> None:
    spec = worker.load_spec()
    assert _sha256(R2 / "candidate_pool_r2.jsonl") == spec["candidate_pool"]["sha256"]
    assert _sha256(R5 / "scripts/train_rnn.py") == spec["runtime_source"]["trainer_sha256"]
    assert _sha256(R5 / "scripts/loss_pair_gpu_worker.py") == spec["runtime_source"]["worker_sha256"]
    pool = _candidate_rows()
    assert set(pool) == {row["candidate_id"] for row in spec["arms"]}
    for arm in spec["arms"]:
        row = pool[arm["candidate_id"]]
        assert recipe_sampler.recipe_hash(row["recipe"]) == arm["candidate_recipe_sha256"]
        assert row["recipe_hash"] == arm["candidate_recipe_sha256"]
        assert row["recipe"]["sampling"]["membership_file_sha256"] == spec["parsed_cache"]["membership_sha256"]


def test_each_candidate_maps_to_supported_cli_and_exact_science_delta() -> None:
    spec = worker.load_spec()
    choices = recipe_sampler.load_cli_choices(
        [R5 / "scripts/train_rnn.py", R5 / "src/wnn/training_dynamics.py"]
    )
    for arm in spec["arms"]:
        argv = worker.build_train_argv(
            arm, Path("/cache/manifest.json"), Path("/cache/validation.json"), "a" * 64
        )
        assert choices["optimizer"] >= {argv[argv.index("--optim") + 1]}
        assert choices["loss"] >= {argv[argv.index("--loss") + 1]}
        assert choices["activation"] >= {argv[argv.index("--readout-activation") + 1]}
        assert choices["schedule"] >= {argv[argv.index("--sched") + 1]}
        assert argv[argv.index("--optim") + 1] == "adamw"
        assert argv[argv.index("--precision") + 1] == "fp32"
        assert "--no-eval-during-fit" in argv
        assert "--save-training-state" in argv
        assert "--grad-stats" in argv
        assert "--valid-fold" in argv and argv[argv.index("--valid-fold") + 1] == "0"
        assert "--train-source" in argv and argv[argv.index("--train-source") + 1] == "train"
        if arm["loss"] == "huber":
            assert float(argv[argv.index("--huber-delta") + 1]) == arm["huber_delta"]
        else:
            assert "--huber-delta" not in argv
        if arm["scheduler"] == "wsd":
            assert argv[argv.index("--wsd-decay") + 1] == "0.2"
            assert argv[argv.index("--wsd-final") + 1] == "0.01"
            assert "--restarts" not in argv
        elif arm["scheduler"] == "sgdr":
            assert argv[argv.index("--restarts") + 1] == "1"
            assert argv[argv.index("--wsd-final") + 1] == "0.01"
            assert "--wsd-decay" not in argv
        else:
            assert "--wsd-decay" not in argv and "--restarts" not in argv
        assert "--optim" in argv and argv[argv.index("--optim") + 1] != "sgd"


def test_actual_d01_onecycle_control_is_complete_and_pinned() -> None:
    control_path = CONTROL_OUTPUT / "training_contract.json"
    assert control_path.is_file()
    assert _sha256(control_path) == worker.CONTROL_CONTRACT_SHA256
    contract = json.loads(control_path.read_text(encoding="utf-8"))
    schedule_path = CONTROL_OUTPUT / contract["artifacts"]["onecycle_trace"]["path"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert contract["fit"]["optimizer"] == "AdamW"
    assert schedule["total_steps"] == 220
    assert len(schedule["rows"]) == 220
    first = worker.load_spec()["arms"][0]
    argv = worker.build_train_argv(
        first, Path("/cache/manifest.json"), Path("/cache/validation.json"), "a" * 64
    )
    assert argv[argv.index("--sched") + 1] == "onecycle"
    assert argv[argv.index("--optim") + 1] == "adamw"


def test_actual_r5_checkpoint_prediction_bindings() -> None:
    output = DIV / "cloud_delivery/d02_r5/terminal_collect_r1/attempt_001/outputs/out"
    model_dir = output / "arms/huber_d1/model"
    selected = worker._verify_checkpoint_selection(
        output / "arms/huber_d1", "d02_muon_huber_d1_r1"
    )
    sidecar = json.loads((output / "arms/huber_d1/predictions_valid.json").read_text())
    prediction = output / "arms/huber_d1/predictions_valid.npy"
    assert sidecar["epoch"] == 1
    assert sidecar["checkpoint_sha256"] == selected["selected_model_sha256"]
    assert sidecar["checkpoint_sha256"] == _sha256(model_dir / "model.pt")
    assert sidecar["prediction_sha256"] == _sha256(prediction)


def test_actual_trainer_builder_yields_frozen_220_step_wsd_and_sgdr_curves() -> None:
    spec = worker.load_spec()
    for arm in spec["arms"]:
        if arm["scheduler"] not in {"wsd", "sgdr"}:
            continue
        rows = worker._expected_nononecycle_rows(arm)
        assert len(rows) == 220
        assert rows[0]["scheduler_step_pre"] == 0
        assert rows[-1]["scheduler_step_pre"] == 219
        assert all(row["optimizer_groups"][0]["beta1"] == 0.9 for row in rows)
        if arm["scheduler"] == "sgdr":
            assert rows[0]["optimizer_groups"][0]["lr"] < rows[10]["optimizer_groups"][0]["lr"]
            assert rows[115]["optimizer_groups"][0]["lr"] > rows[114]["optimizer_groups"][0]["lr"]


def test_schedule_interpretation_does_not_rewrite_raw_trace(tmp_path: Path) -> None:
    rows = [
        {
            "attempt_index": index,
            "epoch_index": 0 if index < 110 else 1,
            "batch_index": index // 10,
            "chunk_index": index % 10,
            "scheduler_step_pre": index,
            "optimizer_groups": [{"group_index": 0, "lr": 0.001, "beta1": 0.9}],
        }
        for index in range(220)
    ]
    raw = {
        "schema": "wnn.loss-pair.onecycle-trace.v1",
        "format_version": 1,
        "total_steps": 220,
        "rows": rows,
    }
    raw_path = tmp_path / "schedule_trace.json"
    raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    original = raw_path.read_bytes()
    arm = next(row for row in worker.load_spec()["arms"] if row["scheduler"] == "wsd")
    sidecar = tmp_path / "schedule_trace_interpreted.json"
    interpreted = worker.interpret_schedule_trace(arm, raw_path, sidecar)
    assert raw_path.read_bytes() == original
    assert interpreted["scheduler"] == "wsd"
    assert interpreted["producer_schema"] == "wnn.loss-pair.onecycle-trace.v1"
    assert "currently labels" in interpreted["producer_schema_note"]
    assert interpreted["rows"] == rows


def test_sampling_cohort_is_three_pairs_and_excludes_deferred_dimensions() -> None:
    spec = worker.load_spec()
    assert spec["execution"]["waves"] == [
        ["R_adamw_huber_d4", "Q_adamw_huber_d2"],
        ["I_gru256_fixedmix", "U_fixedmix_huber_d1"],
        ["J_adamw_wmse_wsd", "K_adamw_wmse_sgdr"],
    ]
    assert spec["deferred_candidate_ids"] == ["S_sgd_wmse", "E_gru384_wmse", "F_gru384_huber_d1"]
    assert spec["parsed_cache"]["parse_once"] is True
    assert spec["parsed_cache"]["scope"] == "cohort"
    assert spec["common_recipe"]["evaluation_during_fit"] is False
