"""Verify the declared E520 WMSE treatment against the pinned source archive."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from fit_guards import FitGateError, trainer_argv, validate_training_target_semantics

PACKET_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    """Load one packet JSON object with a focused type check."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(payload: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def test_wmse_semantics_match_config_plan_and_pinned_trainer() -> None:
    config = _load_json(PACKET_ROOT / "FIT_STAGE_CONFIG.json")
    plan = _load_json(PACKET_ROOT / "E520_FIT_STAGE_PLAN.json")
    declared = config["recipe"]["training_target_semantics"]

    assert declared == plan["recipe"]["training_target_semantics"]
    validate_training_target_semantics(config, PACKET_ROOT)
    argv = trainer_argv(3, "e520_raw_baseline_s3_f0")
    assert argv[argv.index("--loss") + 1] == "wmse"
    assert argv[argv.index("--loss-norm") + 1] == "chunk"

    with tarfile.open(PACKET_ROOT / "source_archive.tar", "r") as archive:
        member = archive.getmember(declared["source_file"])
        source_bytes = archive.extractfile(member).read()
    source = source_bytes.decode("utf-8")
    function_start = source.index("def weighted_mse_loss(")
    function_end = source.index("\n\ndef weighted_huber_loss", function_start)
    function_source = source[function_start:function_end]

    assert _sha256(source_bytes) == declared["source_file_sha256"]
    assert _sha256(function_source.encode("utf-8")) == declared["source_function_sha256"]
    assert "METRIC_CLIP = 2.0" in source
    assert "target.float().clamp(-METRIC_CLIP, METRIC_CLIP)" in function_source
    assert "w = y.abs() * valid[..., None]" in function_source
    assert "(pred.float() - y) ** 2" in function_source
    assert "w.sum().clamp_min(1e-8)" in function_source


def test_wmse_semantics_and_plan_binding_fail_closed() -> None:
    config = _load_json(PACKET_ROOT / "FIT_STAGE_CONFIG.json")
    config["recipe"]["training_target_semantics"]["clip_constant_value"] = 1.0
    with pytest.raises(FitGateError, match="semantics differ"):
        validate_training_target_semantics(config, PACKET_ROOT)

    config = _load_json(PACKET_ROOT / "FIT_STAGE_CONFIG.json")
    config["fit_stage_plan_sha256"] = "0" * 64
    with pytest.raises(FitGateError, match="plan bytes differ"):
        validate_training_target_semantics(config, PACKET_ROOT)
