"""Measure a declared attention/TCN interaction without fitting or weight search."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "research/campaign_2026-10-06/bandit_continuation"


def load_module(name: str, path: Path) -> ModuleType:
    """Load an already validated frozen metric helper."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Score every declared endpoint after binding input and helper identities."""
    output = HERE / "H264_FIXED_EQUAL_MIXTURE_RESULT_20261008_R1.json"
    if output.exists():
        raise FileExistsError("Preserve the original result; do not overwrite")
    prereg_path = HERE / "H264_FIXED_EQUAL_MIXTURE_PREREG_20261008_R1.json"
    prereg = json.loads(prereg_path.read_text())
    grade = load_module(
        "h264_shared_grade", BASE / "attention_training_H218/grade_fold0_reference_blends.py"
    )
    for entry in prereg["bound_inputs"]:
        if grade.sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {entry['path']}")
    scorer = load_module("h264_score", grade.SCORER_PATH)
    helpers = load_module("h264_pair", grade.H217_REFERENCE_HELPERS)
    split = json.loads(grade.SPLIT_PATH.read_text())
    reference_sidecar = json.loads((ROOT / prereg["reference_sidecar"]).read_text())
    reference = np.load(ROOT / prereg["reference_prediction"], mmap_mode="r", allow_pickle=False)
    attention = np.load(ROOT / prereg["attention_prediction"], mmap_mode="r", allow_pickle=False)
    attention_sidecar = json.loads((ROOT / prereg["attention_sidecar"]).read_text())
    targets = scorer.mmap_stored_npz(grade.TARGET_PATH)
    target_groups = targets["row_group"].tolist()
    evaluation = split["evaluation"]["valid"]
    attention_groups = attention_sidecar["evaluation_row_groups"]
    reference_groups = reference_sidecar["row_groups"]
    helpers.verify_reference_lineage(split, reference_sidecar)
    alignment = grade.align_evaluation_rows(
        evaluation, attention_groups, reference_groups, target_groups
    )
    components_by_group = {
        node["row_group"]: node["component"] for node in split["sequences"]["valid"]
    }
    components = np.asarray([components_by_group[row[0][1]] for row in alignment], dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="wnn-h264-") as temporary:
        temp = Path(temporary)
        with tarfile.open(ROOT / prereg["tcn_archive"]) as archive:
            stream = archive.extractfile(prereg["tcn_prediction_member"])
            if stream is None:
                raise FileNotFoundError("Frozen TCN prediction member missing")
            tcn_path = temp / "tcn.npy"
            with stream, tcn_path.open("wb") as destination:
                shutil.copyfileobj(stream, destination, length=1 << 20)
            sidecar_stream = archive.extractfile(prereg["tcn_sidecar_member"])
            if sidecar_stream is None:
                raise FileNotFoundError("Frozen TCN sidecar missing")
            with sidecar_stream:
                tcn_sidecar = json.load(sidecar_stream)
        if grade.sha256_file(tcn_path) != prereg["tcn_prediction_sha256"]:
            raise ValueError("Delivered TCN prediction differs from its completed grade")
        tcn = np.load(tcn_path, mmap_mode="r", allow_pickle=False)
        tcn_index = grade.source_group_index("valid", tcn_sidecar["row_groups"])
        grade.validate_prediction_array(tcn, len(tcn_sidecar["row_groups"]))
        grade.validate_prediction_array(attention, len(attention_groups))
        grade.validate_prediction_array(reference, len(reference_groups))
        mixture = np.lib.format.open_memmap(
            temp / "mixture.npy", mode="w+", dtype=np.float32, shape=attention.shape
        )
        for group, attention_index, _, _ in alignment:
            mixture[attention_index] = (
                np.float64(0.5) * attention[attention_index].astype(np.float64)
                + np.float64(0.5) * tcn[tcn_index[group]].astype(np.float64)
            ).astype(np.float32)
        mixture.flush()
        result = grade.score_predeclared_blends(
            targets["y"],
            targets["mask"],
            mixture,
            reference,
            alignment,
            components,
            scorer,
            helpers,
        )
        for blend in result["predeclared_blends_all_reported"]:
            blend["alpha_on_equal_attention_tcn_mixture"] = blend.pop("alpha_on_h218")
    report = {
        "record_id": "H264_FIXED_EQUAL_MIXTURE_RESULT_20261008_R1",
        "created_utc": datetime.now(UTC).isoformat(),
        "preregistration_sha256": grade.sha256_file(prereg_path),
        "grader_sha256": grade.sha256_file(Path(__file__)),
        "hypothesis": "H-264",
        "metric": "global_wp_clip2_v1",
        "status": "conditional_exposed_discovery_interaction",
        "fixed_rule": "p=(1-alpha)*reference+alpha*(0.5*attention+0.5*TCN), raw blending before official clipping",
        "prediction_exposure": "Both parent outcomes known; joint mixture not scored before this preregistration",
        "limits": "468 exposed groups; conditional component uncertainty excludes seed/public transfer; no weight selection, fresh confirmation, public forecast or promotion",
        **result,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        [
            (r["alpha_on_equal_attention_tcn_mixture"], r["score"]["wp"])
            for r in report["predeclared_blends_all_reported"]
        ]
    )


if __name__ == "__main__":
    main()
