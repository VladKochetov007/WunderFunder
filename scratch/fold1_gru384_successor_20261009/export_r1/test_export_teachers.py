from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("export_teachers.py")
SPEC = importlib.util.spec_from_file_location("export_teachers", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ROOT = Path(__file__).resolve().parents[3]


def test_existing_and_dangling_outputs_are_rejected(tmp_path: Path) -> None:
    """The predictor must not replace a final, partial, sidecar or stale cache."""
    for path in MODULE.output_paths(tmp_path, "example", "valid"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(tmp_path / "missing")
        with pytest.raises(FileExistsError):
            MODULE.verify_absent([path])
        path.unlink()


def test_frozen_source_drift_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"before")
    pins = {"source.py": hashlib.sha256(b"before").hexdigest()}
    MODULE.verify_pins(tmp_path, pins)
    source.write_bytes(b"after")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        MODULE.verify_pins(tmp_path, pins)


def test_actual_existing_fold1_producer_is_accepted() -> None:
    """Exercise the exact post-export consumer on a saved real teacher export."""
    from scripts.build_distill_cache import _audit_ancestor_closure, _inspect_teacher
    from wnn.splits import load_campaign_split

    split = load_campaign_split(
        ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json", ROOT
    )
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as target:
        order = target["row_group"].tolist()
    run = "c20261003_combo_s3_f1"
    closure = _audit_ancestor_closure(run, split, ROOT / "artifacts/runs", "model_ep3.pt")
    record, arrays, caches = _inspect_teacher(
        run, split, ROOT / "cache", ROOT / "artifacts/runs", {"valid": order}, 3
    )
    assert closure and record["epoch"] == 3
    assert arrays["valid"].shape == (1873, 20000, 2)
    assert (
        caches["valid"]["prediction_sha256"]
        == "95db5d74bf00c211d40991202a43322c58551ea0af6b2e360daa2401809bd600"
    )
