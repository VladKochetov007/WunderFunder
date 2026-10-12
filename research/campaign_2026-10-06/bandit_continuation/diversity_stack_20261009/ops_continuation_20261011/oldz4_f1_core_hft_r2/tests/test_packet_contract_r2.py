from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PACKET_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PACKET_ROOT / "run_f1_core_hft_stage_r2.py"
SPEC = importlib.util.spec_from_file_location("f1_core_hft_r2", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_snapshot_closure_accepts_only_pinned_extra(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "trainer.py").write_bytes(b"frozen trainer\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "declared_extra.cpp").write_bytes(b"separately bound source\n")
    files = {"scripts/trainer.py": _digest(tmp_path / "scripts" / "trainer.py")}
    extra = {"src/declared_extra.cpp": _digest(tmp_path / "src" / "declared_extra.cpp")}

    RUNNER._verify_file_manifest(tmp_path, files, "fixture", extra)

    (tmp_path / "unlisted.py").write_bytes(b"must not be silently accepted")
    with pytest.raises(ValueError, match="file list differs"):
        RUNNER._verify_file_manifest(tmp_path, files, "fixture", extra)


def test_supplemental_manifest_digest_is_checked(tmp_path: Path) -> None:
    (tmp_path / "declared.py").write_bytes(b"declared")
    (tmp_path / "extra.cpp").write_bytes(b"extra")
    files = {"declared.py": _digest(tmp_path / "declared.py")}
    extra = {"extra.cpp": "0" * 64}

    with pytest.raises(ValueError, match="source bytes changed: extra.cpp"):
        RUNNER._verify_file_manifest(tmp_path, files, "fixture", extra)


def test_hft_profile_and_fit_keep_f0_stage_semantics() -> None:
    profile = json.loads((PACKET_ROOT / "HFT_PROFILE_CONTRACT_R2.json").read_text())
    fit = json.loads((PACKET_ROOT / "HFT_FIT_CONTRACT_R2.json").read_text())
    profile_argv = profile["trainer_argv"]
    fit_argv = fit["trainer_argv"]

    assert "--train-source" in fit_argv
    assert fit_argv[fit_argv.index("--train-source") + 1] == "valid"
    assert "--mask-weight" in fit_argv
    assert "--mask-weight" in profile_argv
    assert "--save-training-state" not in profile_argv
    assert "--save-training-state" in fit_argv
    assert "--aux-struct" not in fit_argv and "--aux-next" not in fit_argv
    assert fit["membership"]["total_source_qualified_direct_fit_groups"] == 1416
    assert fit["membership"]["outer_valid_eval_groups_excluded"] == 457
    assert fit["fit_authorized"] is False
    assert profile["profile_authorized"] is False


def test_f1_student_parent_pins_match_actual_artifacts() -> None:
    bindings = json.loads((PACKET_ROOT / "SOURCE_INPUT_BINDINGS_R2.json").read_text())
    pins = bindings["f1_student_cp1"]
    assert pins["run"] == "oldz4_h230_student_f1_s3_r4"
    for key in ("checkpoint", "provenance", "training_state"):
        path = Path(pins["paths"][key])
        assert path.is_file()
        assert _digest(path) == pins["sha256"][key]


def test_contract_hashes_close_over_source_and_fit_bindings() -> None:
    binding_path = PACKET_ROOT / "SOURCE_INPUT_BINDINGS_R2.json"
    binding_sha = _digest(binding_path)
    for name in (
        "CORE_CROP_CONTRACT_R2.json",
        "HFT_PROFILE_CONTRACT_R2.json",
        "HFT_FIT_CONTRACT_R2.json",
        "HFT_EXPORT_CONTRACT_R2.json",
        "F1_STAGE_PLAN_R2.json",
    ):
        contract = json.loads((PACKET_ROOT / name).read_text())
        assert contract["source_input_bindings_sha256"] == binding_sha
    export = json.loads((PACKET_ROOT / "HFT_EXPORT_CONTRACT_R2.json").read_text())
    assert export["fit_contract_sha256"] == _digest(PACKET_ROOT / "HFT_FIT_CONTRACT_R2.json")
