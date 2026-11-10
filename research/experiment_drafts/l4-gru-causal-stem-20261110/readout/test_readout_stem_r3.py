"""Focused CPU-only gates for the causal-stem readout receiver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

MODULE_PATH = Path(__file__).with_name("readout_stem_r3.py")
SPEC = importlib.util.spec_from_file_location("readout_stem_r3_fixture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
receiver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receiver
SPEC.loader.exec_module(receiver)
PACKET = MODULE_PATH.resolve().parents[1]
REPOSITORY = MODULE_PATH.resolve().parents[4]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _admission_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Copy real frozen config bytes and bind each source-normalized identity."""
    root = tmp_path / "native-root"
    config_root = root / "inputs-r1" / "configs"
    receipt_root = root / "profiles"
    config_root.mkdir(parents=True)
    receipt_root.mkdir()
    profiles: dict[str, dict[str, object]] = {}
    with receiver._pinned_config_loader(REPOSITORY) as load_config:
        for cell, (_, _, gpu) in receiver.SPEC.items():
            source_config = PACKET / "configs" / f"{cell}.json"
            config_path = config_root / source_config.name
            shutil.copyfile(source_config, config_path)
            receipt = receipt_root / f"{cell}.json"
            receipt.write_text("{}\n", encoding="utf-8")
            profiles[cell] = {
                "gpu": gpu,
                "gpu_uuid": receiver.GPU_UUIDS[gpu],
                "raw_config_sha256": _sha_file(config_path),
                "config_sha256": _canonical_sha(load_config(config_path)),
                "receipt_sha256": _sha_file(receipt),
                "receipt_path": str(receipt),
            }
    admission: dict[str, object] = {
        "schema": "wnn.l4-stem-root-fit-admission.v1",
        "approved": True,
        "max_fresh_fits": 6,
        "source_root": str(REPOSITORY),
        "source_sha256": receiver.SOURCE_SHA256,
        "split_sha256": receiver.SPLIT_SHA256,
        "profiles": profiles,
    }
    admission_path = root / "control" / "ROOT-fit-admission-r1.json"
    admission_path.parent.mkdir()
    admission_path.write_text(
        json.dumps(admission, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    common = {
        "_read_json": lambda path, limit: json.loads(path.read_text(encoding="utf-8")),
        "sha256_file": _sha_file,
        "canonical_sha256": _canonical_sha,
    }
    return common, root, admission


def _write_admission(root: Path, admission: dict[str, object]) -> str:
    path = root / "control" / "ROOT-fit-admission-r1.json"
    path.write_text(json.dumps(admission, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return _sha_file(path)


def test_admission_hashes_raw_bytes_and_exact_source_normalized_config(tmp_path: Path) -> None:
    fixture = json.loads(
        Path(__file__).with_name("config_normalization_fixture_r2.json").read_text()
    )
    common, root, admission = _admission_fixture(tmp_path)
    profiles = admission["profiles"]
    raw_path = root / "inputs-r1" / "configs" / f"{fixture['cell']}.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert _sha_file(raw_path) == fixture["raw_config_sha256"]
    assert raw["model"]["input_stem_kernel"] == 0
    assert _sha_file(raw_path) != profiles[fixture["cell"]]["config_sha256"]

    with receiver._pinned_config_loader(REPOSITORY) as load_config:
        normalized = load_config(raw_path)
        assert _canonical_sha(normalized) == fixture["normalized_config_sha256"]
        assert "input_stem_kernel" not in normalized["model"]
        assert receiver.validate_admission(
            common,
            root,
            REPOSITORY,
            _write_admission(root, admission),
            load_config,
        )["admission_sha256"] == _sha_file(root / "control" / "ROOT-fit-admission-r1.json")

    bad_raw = json.loads(json.dumps(admission))
    bad_raw["profiles"][fixture["cell"]]["raw_config_sha256"] = "0" * 64
    bad_raw_sha = _write_admission(root, bad_raw)
    with (
        receiver._pinned_config_loader(REPOSITORY) as load_config,
        pytest.raises(ValueError, match="profile/config receipt differs"),
    ):
        receiver.validate_admission(common, root, REPOSITORY, bad_raw_sha, load_config)

    def wrong_config(config_path: Path) -> dict[str, object]:
        config = load_config(config_path)
        if config_path.name == f"{fixture['cell']}.json":
            config["seed"] += 1
        return config

    with (
        receiver._pinned_config_loader(REPOSITORY) as load_config,
        pytest.raises(ValueError, match="profile/config receipt differs"),
    ):
        receiver.validate_admission(
            common,
            root,
            REPOSITORY,
            _write_admission(root, admission),
            lambda path: wrong_config(path),
        )


@pytest.mark.parametrize("published_at", [99.9, 100.1])
def test_final_poll_recovers_an_atomic_receipt_but_does_not_admit_late_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published_at: float
) -> None:
    path = tmp_path / "control/root-fits-r1/plain_s203301-result.json"
    path.parent.mkdir(parents=True)
    clock = [99.0]

    def final_sleep(duration: float) -> None:
        path.write_text(json.dumps({"at": published_at}), encoding="utf-8")
        os.utime(path, (published_at, published_at))
        clock[0] = 100.2

    monkeypatch.setattr(receiver.time, "time", lambda: clock[0])
    monkeypatch.setattr(receiver.time, "sleep", final_sleep)
    result = receiver.wait_for_fit(
        {"_read_json": lambda file, limit: json.loads(file.read_bytes())},
        tmp_path,
        "plain_s203301",
        100.0,
    )
    assert result == {"at": published_at}
    if published_at <= 100.0:
        receiver.validate_fit_result_time(path, result, 100.0)
    else:
        with pytest.raises(TimeoutError):
            receiver.validate_fit_result_time(path, result, 100.0)


def test_rebound_helper_functions_use_the_actual_module_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject the runpy-copy failure where validators still see the old source pin."""
    names = (
        "_absolute_executable",
        "_read_json",
        "_read_events",
        "sha256_file",
        "canonical_sha256",
        "validate_fit_membership",
        "validate_prediction_manifest",
        "validate_score_receipt",
        "validate_project_file_stats",
        "ReadoutGuard",
        "_command",
        "directory_bytes",
    )
    source = "SOURCE_SHA256 = 'old-source'\n" + "\n".join(
        f"def {name}(*args, **kwargs):\n    return SOURCE_SHA256\n" for name in names
    )
    helper = tmp_path / "frozen_helper.py"
    helper.write_text(source, encoding="utf-8")
    monkeypatch.setattr(receiver, "R2_SHA256", hashlib.sha256(helper.read_bytes()).hexdigest())
    namespace = receiver.load_r2_helper(helper)
    for name in ("validate_prediction_manifest", "validate_score_receipt", "_command"):
        assert namespace[name].__globals__ is namespace
        assert namespace[name]() == receiver.SOURCE_SHA256


def test_interpreter_path_keeps_venv_spelling(tmp_path: Path) -> None:
    """The command path stays lexical even when its filesystem entry is a symlink."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    candidate = venv / "python"
    with (
        patch.object(Path, "is_file", return_value=True),
        patch("os.access", return_value=True),
        patch.object(
            Path, "resolve", side_effect=AssertionError("must not dereference venv executable")
        ),
    ):
        actual = receiver.lexical_executable({}, candidate)
    assert str(actual) == os.path.abspath(os.fspath(candidate))


def test_fit_result_cutoff_checks_timestamp_and_atomic_file_mtime(tmp_path: Path) -> None:
    result_path = tmp_path / "cell-result.json"
    result_path.write_text("{}", encoding="utf-8")
    os.utime(result_path, (99.0, 99.0))
    receiver.validate_fit_result_time(result_path, {"at": 99.0}, 100.0)
    with pytest.raises(TimeoutError, match="after the original fit cutoff"):
        receiver.validate_fit_result_time(result_path, {"at": 100.1}, 100.0)
    os.utime(result_path, (101.0, 101.0))
    with pytest.raises(TimeoutError, match="after the original fit cutoff"):
        receiver.validate_fit_result_time(result_path, {"at": 99.0}, 100.0)


def test_fit_result_rejects_nonfinite_timestamp(tmp_path: Path) -> None:
    result_path = tmp_path / "cell-result.json"
    result_path.write_text("{}", encoding="utf-8")
    with pytest.raises(TimeoutError):
        receiver.validate_fit_result_time(result_path, {"at": float("nan")}, 100.0)


def test_contrasts_pair_only_matching_seeds_and_expected_directions() -> None:
    scores = {}
    for cell, (seed, kernel, _) in receiver.SPEC.items():
        arm_delta = {0: 0.0, 1: 0.01, 16: 0.04}[kernel]
        scores[cell] = {
            "within_tolerance": True,
            "t0": seed / 1_000_000 + arm_delta,
            "t1": seed / 2_000_000 + arm_delta * 2,
            "wp": seed / 1_500_000 + arm_delta * 1.5,
        }
    rows = receiver.derive_contrasts(scores)
    assert [(row["seed"], row["contrast"]) for row in rows] == [
        (203301, "K16 minus k1"),
        (203301, "K16 minus plain"),
        (203302, "K16 minus k1"),
        (203302, "K16 minus plain"),
    ]
    for row in rows:
        scale = 1 if row["contrast"] == "K16 minus k1" else 4 / 3
        assert row["t0"] == pytest.approx(0.03 * scale)
        assert row["t1"] == pytest.approx(0.06 * scale)
        assert row["wp"] == pytest.approx(0.045 * scale)


def test_contrasts_withhold_incomplete_or_mismatched_scores() -> None:
    scores = {
        cell: {"within_tolerance": True, "t0": 0.1, "t1": 0.2, "wp": 0.15}
        for cell in receiver.CELLS
    }
    scores["k16_s203302"]["within_tolerance"] = False
    with pytest.raises(ValueError, match="all six"):
        receiver.derive_contrasts(scores)


def test_membership_uses_exact_manifest_ids_including_sparse_train_groups() -> None:
    """The official train roster is sparse; a contiguous range must be rejected."""
    split = {
        "fit": {"train": [0, 2, 5]},
        "evaluation": {"valid": [0, 3]},
    }

    def exact_manifest_validator(fit: dict[str, object], manifest: dict[str, object]) -> None:
        fit_section = manifest["fit"]
        evaluation_section = manifest["evaluation"]
        assert isinstance(fit_section, dict) and isinstance(evaluation_section, dict)
        expected_fit = [["train", group] for group in fit_section["train"]]
        expected_valid = [["valid", group] for group in evaluation_section["valid"]]
        if (
            fit.get("fit_members") != expected_fit
            or fit.get("evaluation_members") != expected_valid
        ):
            raise ValueError("native run membership differs from the exact official primary split")

    common = {"validate_fit_membership": exact_manifest_validator}
    receiver.validate_official_membership(
        common, [["train", 0], ["train", 2], ["train", 5]], [["valid", 0], ["valid", 3]], split
    )
    with pytest.raises(ValueError, match="native run membership"):
        receiver.validate_official_membership(
            common, [["train", 0], ["train", 1], ["train", 2]], [["valid", 0], ["valid", 3]], split
        )
