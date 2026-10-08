"""Source-only checks for frozen H272 grading and actual parent metadata."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any  # Frozen producer sidecars contain heterogeneous JSON values.

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
CAMPAIGN = PACKAGE.parents[1]
GRADE_PATH = PACKAGE / "grade_h272_direct_label_readouts.py"
PREREG_PATH = PACKAGE / "H272_DIRECT_LABEL_COMPARISON_PREREG_V1_20261008.json"
H233_PREREG_PATH = CAMPAIGN / "H233_FIXED_EP1_COMPARISON_PREREG_20261008_R1.json"
H233_GRADER_PATH = CAMPAIGN / "grade_h233_readouts.py"
SIDECAR_PATH = Path(__file__).resolve().parent / "fixtures/h233_c1_valid_sidecar.json"
SIDECAR_SHA256 = "bb335392edc599f181c0bbcb6523fd430b036647454bc0d76facf5b592211ad3"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sidecar() -> dict[str, Any]:
    raw = SIDECAR_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SIDECAR_SHA256
    return json.loads(raw)


def test_h272_prereg_keeps_h233_comparison_math_and_weights() -> None:
    h272 = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    h233 = json.loads(H233_PREREG_PATH.read_text(encoding="utf-8"))

    for field in (
        "checkpoint_epoch",
        "metric",
        "primary",
        "modes",
        "secondary",
        "weights_on_readout",
        "bootstrap",
        "selection",
        "serving",
    ):
        assert h272[field] == h233[field]
    assert h272["weights_on_readout"] == [0, 0.1, 0.25, 0.5, 1]
    assert h272["bootstrap"] == {
        "unit": "paired frozen connected components",
        "replicates": 2_000,
        "seed": 271_828,
    }
    assert h272["status"] == "preregistered_before_H272_candidate_export_and_grade"


def test_actual_h233_sidecar_binds_producer_checkpoint_not_alias() -> None:
    sidecar = _load_sidecar()
    grader = _load_module("h272_grade_sidecar_contract", GRADE_PATH)

    assert sidecar["run"] == "bd26_c1v2_s3_f0"
    assert sidecar["split"] == "valid"
    assert sidecar["epoch"] == 1
    assert sidecar["inference_precision"] == "fp16"
    assert len(sidecar["row_groups"]) == 1_873
    assert sidecar["checkpoint_sha256"] == grader.PARENT_SHA
    assert (
        sidecar["checkpoint_sha256"]
        != "5f21361d47cc1f010abb363e6aca6e3cbf50a3d5d40d88ad81851aed7519fc4d"
    )

    grader.validate_parent_sidecar(sidecar, sidecar["row_groups"])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("split", "train"),
        ("epoch", 0),
        ("checkpoint_sha256", "0" * 64),
        ("prediction_sha256", "1" * 64),
        ("split_manifest_sha256", "2" * 64),
        ("inference_precision", "fp32"),
        ("row_groups", [1, 2, 3]),
    ),
)
def test_parent_sidecar_semantic_drift_is_rejected(field: str, value: object) -> None:
    sidecar = _load_sidecar()
    sidecar[field] = value
    grader = _load_module("h272_grade_sidecar_negative", GRADE_PATH)

    with pytest.raises(ValueError, match="frozen producer or valid row order"):
        grader.validate_parent_sidecar(sidecar, _load_sidecar()["row_groups"])


def test_h272_grader_pins_frozen_h233_grader_and_shared_scoring_sources() -> None:
    grader = _load_module("h272_grade_pins", GRADE_PATH)
    base = _load_module("h233_grade_pins", H233_GRADER_PATH)

    assert hashlib.sha256(H233_GRADER_PATH.read_bytes()).hexdigest() == grader.BASE_GRADER_SHA
    assert grader.PAIR == base.PAIR
    assert grader.SHARED == base.SHARED


def test_grader_source_closure_verifies_without_reading_candidate_artifacts() -> None:
    grader = _load_module("h272_grade_source_closure", GRADE_PATH)
    shared = _load_module("h272_shared_grade_source_closure", grader.SHARED)

    pinned = grader._verify_frozen_scoring_sources(shared)

    assert pinned["h272_preregistration"] == grader.PREREG_SHA
    assert pinned["h233_base_grader"] == grader.BASE_GRADER_SHA
    assert pinned["h272_recipe"] == grader.RECIPE_SHA
    assert pinned["h272_source_lock"] == grader.SOURCE_LOCK_SHA


def test_grader_rejects_unapproved_admission_before_artifact_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader = _load_module("h272_grade_unapproved", GRADE_PATH)
    authorization = tmp_path / "authorization.json"
    output = tmp_path / "grade.json"
    authorization.write_text(json.dumps({"status": "pending"}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [str(GRADE_PATH), "--authorization", str(authorization), "--output", str(output)],
    )

    with pytest.raises(ValueError, match="Missing actual-artifact root H272 grading admission"):
        grader.main()

    assert not output.exists()
