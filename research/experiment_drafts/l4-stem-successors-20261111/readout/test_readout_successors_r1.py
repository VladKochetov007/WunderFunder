"""CPU-only contract checks for the fixed-endpoint successor receiver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("readout_successors_r1.py")
SPEC = importlib.util.spec_from_file_location("readout_successors_r1_fixture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
receiver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receiver
SPEC.loader.exec_module(receiver)
ARTIFACT = Path(r"C:\Users\admin\Documents\Codex\artifacts\l4-stem-successors-20261111")
SOURCE = Path(r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-stem-successors-20261111")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, limit: int) -> dict[str, object]:
    payload = path.read_bytes()
    if len(payload) > limit:
        raise ValueError("fixture JSON exceeds the bounded reader")
    value = json.loads(payload)
    assert isinstance(value, dict)
    return value


def _admission_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Bind the real six frozen config bytes to source-normalized profile hashes."""
    root = tmp_path / "native-root"
    config_root = root / "inputs-r1" / "configs"
    receipt_root = root / "profiles"
    config_root.mkdir(parents=True)
    receipt_root.mkdir()
    profiles: dict[str, dict[str, object]] = {}
    with receiver._pinned_config_loader(SOURCE) as load_config:
        for index, cell in enumerate(receiver.CELLS):
            source_config = ARTIFACT / "configs" / f"{cell}.json"
            config_path = config_root / source_config.name
            shutil.copyfile(source_config, config_path)
            config = load_config(config_path)
            normalized_sha = _canonical_sha(config)
            receipt = receipt_root / f"{cell}.json"
            receipt.write_text(
                json.dumps(
                    {
                        "status": "profile_complete",
                        "updates": 30,
                        "errors": [],
                        "identity": {
                            "source_sha256": receiver.SOURCE_SHA256,
                            "config_sha256": normalized_sha,
                        },
                        "engineering_reload": {"status": "passed"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            gpu = receiver.GPU_BY_CELL[cell]
            profiles[cell] = {
                "gpu": gpu,
                "gpu_uuid": f"fixture-uuid-{index}",
                "raw_config_sha256": _sha_file(config_path),
                "config_sha256": normalized_sha,
                "receipt_sha256": _sha_file(receipt),
                "receipt_path": str(receipt),
            }
    admission: dict[str, object] = {
        "schema": "wnn.l4-stem-successors-root-fit-admission.v1",
        "approved": True,
        "max_fresh_fits": 6,
        "source_root": str(SOURCE.resolve()),
        "source_sha256": receiver.SOURCE_SHA256,
        "split_sha256": receiver.SPLIT_SHA256,
        "profiles": profiles,
    }
    admission_path = root / "control" / "ROOT-fit-admission-r1.json"
    admission_path.parent.mkdir()
    admission_path.write_text(json.dumps(admission, sort_keys=True), encoding="utf-8")
    common: dict[str, object] = {
        "_read_json": _read_json,
        "sha256_file": _sha_file,
        "canonical_sha256": _canonical_sha,
    }
    return common, root, admission


def _write_admission(root: Path, admission: dict[str, object]) -> str:
    path = root / "control" / "ROOT-fit-admission-r1.json"
    path.write_text(json.dumps(admission, sort_keys=True), encoding="utf-8")
    return _sha_file(path)


def test_successor_contract_accepts_k64_hidden512_and_dropout01() -> None:
    with receiver._pinned_config_loader(SOURCE) as load_config:
        for cell in receiver.CELLS:
            normalized = load_config(ARTIFACT / "configs" / f"{cell}.json")
            assert receiver._valid_config_contract(cell, normalized)
    assert receiver.SPEC["k64_s203301"][3:5] == (64, 256)
    assert receiver.SPEC["gru512_k16_s203301"][3:5] == (16, 512)
    assert receiver.SPEC["gru_dropout01_k16_s203301"][3:] == (16, 256, 0.1)


def test_admission_binds_raw_and_normalized_config_and_reload_receipts(tmp_path: Path) -> None:
    common, root, admission = _admission_fixture(tmp_path)
    with receiver._pinned_config_loader(SOURCE) as load_config:
        digest = _write_admission(root, admission)
        validated = receiver.validate_admission(common, root, SOURCE, digest, load_config)
    assert validated["admission_sha256"] == _sha_file(
        root / "control" / "ROOT-fit-admission-r1.json"
    )
    assert common["GPU_BINDINGS"]["gru512_k16_s203301"] == (1, "fixture-uuid-1")

    bad_raw = json.loads(json.dumps(admission))
    bad_raw["profiles"]["k64_s203301"]["raw_config_sha256"] = "0" * 64
    with (
        receiver._pinned_config_loader(SOURCE) as load_config,
        pytest.raises(ValueError, match="profile/config receipt differs"),
    ):
        receiver.validate_admission(
            common, root, SOURCE, _write_admission(root, bad_raw), load_config
        )

    bad_reload = json.loads(json.dumps(admission))
    receipt_path = Path(bad_reload["profiles"]["gru512_k16_s203301"]["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["engineering_reload"]["status"] = "failed"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    bad_reload["profiles"]["gru512_k16_s203301"]["receipt_sha256"] = _sha_file(receipt_path)
    with (
        receiver._pinned_config_loader(SOURCE) as load_config,
        pytest.raises(ValueError, match="reload gate"),
    ):
        receiver.validate_admission(
            common, root, SOURCE, _write_admission(root, bad_reload), load_config
        )


def test_reused_control_reader_requires_exact_receipt_and_full_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = "k16_s203301"
    path = tmp_path / "runs" / cell / "final-readout-r3" / "readout-receipt.json"
    path.parent.mkdir(parents=True)
    receipt = {
        "schema": "wnn.l4-gru-causal-stem-readout-cell.r3",
        "status": "readout_complete",
        "cell": cell,
        "source_sha256": receiver.OLD_CONTROL_SOURCE_SHA256,
        "source_commit": receiver.OLD_CONTROL_SOURCE_COMMIT,
        "split_sha256": receiver.SPLIT_SHA256,
        "data_sha256": receiver.OLD_CONTROL_DATA_SHA256,
        "model_sha256": "model-hash",
        "config_sha256": "config-hash",
        "prediction_count": receiver.VALID_SEQUENCES,
        "prediction_shape": [receiver.SEQUENCE_LENGTH, 2],
        "prediction_dtype": "float32",
        "score": {
            "metric_id": "global_wp_clip2_v1",
            "selected_rows": receiver.VALID_ROWS,
            "within_tolerance": True,
            "t0": 0.61,
            "t1": 0.62,
            "wp": 0.615,
        },
        "monitor_final_metrics": {
            "valid/sequence_count": receiver.VALID_SEQUENCES,
            "valid/selected_rows": receiver.VALID_ROWS,
        },
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(receiver, "OLD_CONTROL_RECEIPT_SHA256", {cell: _sha_file(path)})
    monkeypatch.setattr(receiver, "OLD_CONTROL_MODEL_SHA256", {cell: "model-hash"})
    monkeypatch.setattr(receiver, "OLD_CONTROL_CONFIG_SHA256", {cell: "config-hash"})
    common = {
        "_read_json": _read_json,
        "sha256_file": _sha_file,
        "_finite": lambda x: isinstance(x, (int, float)),
    }
    controls = receiver.validate_reused_controls(tmp_path, common)
    assert controls[cell]["wp"] == 0.615

    receipt["score"]["selected_rows"] -= 1
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(receiver, "OLD_CONTROL_RECEIPT_SHA256", {cell: _sha_file(path)})
    with pytest.raises(ValueError, match="invalid endpoint evidence"):
        receiver.validate_reused_controls(tmp_path, common)


def test_contrasts_pair_each_successor_with_same_seed_control_and_average_seeds() -> None:
    scores: dict[str, dict[str, object]] = {}
    for cell, (seed, arm, *_rest) in receiver.SPEC.items():
        delta = {"k64": 0.01, "gru512_k16": 0.02, "gru_dropout01_k16": 0.03}[arm]
        scores[cell] = {
            "within_tolerance": True,
            "t0": seed / 1_000_000 + delta,
            "t1": seed / 2_000_000 + 2 * delta,
            "wp": seed / 1_500_000 + 1.5 * delta,
        }
    controls = {
        f"k16_s{seed}": {
            "within_tolerance": True,
            "t0": seed / 1_000_000,
            "t1": seed / 2_000_000,
            "wp": seed / 1_500_000,
            "receipt_sha256": f"receipt-{seed}",
        }
        for seed in (203301, 203302)
    }
    contrasts = receiver.derive_contrasts(scores, controls)
    assert len(contrasts["by_seed"]) == 6
    assert len(contrasts["two_seed_mean"]) == 3
    assert [row["arm"] for row in contrasts["two_seed_mean"]] == list(receiver.ARMS)
    for index, _arm in enumerate(receiver.ARMS, start=1):
        row = contrasts["two_seed_mean"][index - 1]
        assert row["wp"] == pytest.approx(1.5 * 0.01 * index)
    scores.pop("k64_s203302")
    with pytest.raises(ValueError, match="all six successors"):
        receiver.derive_contrasts(scores, controls)


def test_cohort_waiter_does_not_release_before_all_fit_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    folder = root / "control" / "root-fits-r1"
    folder.mkdir(parents=True)
    cells = receiver.CELLS
    for cell in cells[:-1]:
        (folder / f"{cell}-result.json").write_text("{}", encoding="utf-8")
    now = [100.0]

    def final_sleep(_duration: float) -> None:
        (folder / f"{cells[-1]}-result.json").write_text("{}", encoding="utf-8")
        now[0] = 100.5

    monkeypatch.setattr(receiver.time, "time", lambda: now[0])
    monkeypatch.setattr(receiver.time, "sleep", final_sleep)
    results = receiver.wait_for_all_fit_results(
        {"_read_json": lambda path, limit: json.loads(path.read_text(encoding="utf-8"))},
        root,
        cells,
        101.0,
    )
    assert set(results) == set(cells)


def test_clock_keeps_original_campaign_deadlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = receiver.CAMPAIGN_FIRST_GPU_EPOCH + 10
    clock = {
        "first_profile_epoch": first,
        "campaign_first_gpu_epoch": receiver.CAMPAIGN_FIRST_GPU_EPOCH,
        "fit_deadline_epoch": receiver.CAMPAIGN_FIRST_GPU_EPOCH + receiver.FIT_SECONDS,
        "wave_deadline_epoch": receiver.CAMPAIGN_FIRST_GPU_EPOCH + receiver.WAVE_SECONDS,
    }
    path = tmp_path / "receipts" / "first-profile-r1.json"
    path.parent.mkdir()
    path.write_text(json.dumps(clock), encoding="utf-8")
    monkeypatch.setattr(receiver.time, "time", lambda: first + 1)
    common = {"_read_json": _read_json, "_finite": lambda value: isinstance(value, (int, float))}
    assert receiver._clock(common, tmp_path)[1:] == (
        receiver.CAMPAIGN_FIRST_GPU_EPOCH,
        receiver.CAMPAIGN_FIRST_GPU_EPOCH + receiver.FIT_SECONDS,
        receiver.CAMPAIGN_FIRST_GPU_EPOCH + receiver.WAVE_SECONDS,
    )
    clock["wave_deadline_epoch"] += 3600
    path.write_text(json.dumps(clock), encoding="utf-8")
    with pytest.raises(TimeoutError, match="original first-GPU"):
        receiver._clock(common, tmp_path)


def test_fit_result_cutoff_checks_timestamp_and_atomic_file_mtime(tmp_path: Path) -> None:
    path = tmp_path / "cell-result.json"
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (99.0, 99.0))
    receiver.validate_fit_result_time(path, {"at": 99.0}, 100.0)
    with pytest.raises(TimeoutError):
        receiver.validate_fit_result_time(path, {"at": 100.1}, 100.0)
    os.utime(path, (101.0, 101.0))
    with pytest.raises(TimeoutError):
        receiver.validate_fit_result_time(path, {"at": 99.0}, 100.0)
