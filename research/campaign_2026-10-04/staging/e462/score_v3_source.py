"""Verify E462 q8 parity and score fixed packages without synthesizing run sidecars."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
E416_SCRIPTS = (
    CAMPAIGN / "kaggle/E419_qat_dualT4_v4/source_root/scripts"
)
SOURCE_ZIP = ROOT / "submissions/c20261004_combo_s3_i16/submission.zip"
SOURCE_SHA256 = "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb"
PARENT_RUN = "c20261004_combo_s3_final_r1"
HEAD_RUN = "c20261004_combo_head_s3_final"
PARENT_SHA256 = "64c2b28651b414778d6af3c33bb9d54f653a46fb0be087de2e3c44ca88c31166"
HEAD_SHA256 = "f1cb88f38b6d5b5b67859346b439e69d610928bc836baac8867e84e7b1c95d85"
E457_DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
VARIANTS = {"q8r": "recurrent", "q8x": "recurrent_head"}
MAX_ABS = 3e-4
MAX_WP_DELTA = 5e-6


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one new immutable JSON report."""
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_package(zip_path: Path) -> tuple[Path, object]:
    """Extract one trusted ZIP into a temporary directory and load its callback."""
    temporary = tempfile.TemporaryDirectory(prefix="wnn-e462-package-")
    package_dir = Path(temporary.name)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"invalid package archive: {zip_path}")
        archive.extractall(package_dir)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(package_dir))
    sys.modules.pop("wnn_native", None)
    spec = importlib.util.spec_from_file_location(
        f"e462_callback_{hashlib.sha256(zip_path.name.encode()).hexdigest()[:10]}",
        package_dir / "solution.py",
    )
    if spec is None or spec.loader is None:
        temporary.cleanup()
        raise ImportError(f"package has no loadable callback: {zip_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return temporary, module.PredictionModel()


def _stream_rows(path: Path, groups: list[int]):
    """Yield full fixed-length rows in specified Parquet row-group order."""
    sys.path.insert(0, str(ROOT))
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH

    parquet = pq.ParquetFile(path)
    columns = ["seq_ix", "need_prediction", "is_scored", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    for group in groups:
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {group} is not one full sequence")
        sequence = int(table.column("seq_ix")[0].as_py())
        features = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
        ).astype(np.float32)
        targets = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in TARGET_COLUMNS]
        ).astype(np.float32)
        need = table.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool)
        mask = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool) & need
        yield group, sequence, features, targets, need, mask


def _callback_rows(model: object, sequence: int, features: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Replay the actual per-row scorer callback and retain finite required outputs."""
    from utils import DataPoint, SEQUENCE_LENGTH

    predictions = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    for step in range(SEQUENCE_LENGTH):
        value = model.predict(DataPoint(sequence, step, bool(need[step]), features[step]))
        if not need[step]:
            if value is not None:
                raise ValueError("callback returned a warmup prediction")
            continue
        row = np.asarray(value, dtype=np.float32)
        if row.shape != (2,) or not np.isfinite(row).all():
            raise ValueError("callback returned malformed or nonfinite scores")
        predictions[step] = row
    return predictions


def _load_reference_helpers():
    """Load E416's independent Torch reference with current root WNN sources first."""
    root_src = str(ROOT / "src")
    if root_src not in sys.path:
        sys.path.insert(0, root_src)
    path = E416_SCRIPTS / "probe_q8_model.py"
    spec = importlib.util.spec_from_file_location("e462_e416_q8_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing E416 Torch reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.torch_reference


def _load_exported_weights(package_zip: Path) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Read package-local q8 export through the same checked parser used by its callback."""
    temporary = tempfile.TemporaryDirectory(prefix="wnn-e462-weights-")
    package_dir = Path(temporary.name)
    with zipfile.ZipFile(package_zip) as archive:
        archive.extract("weights_0.npz", package_dir)
    solution_path = Path(__file__).with_name("solution_q8.py")
    spec = importlib.util.spec_from_file_location("e462_package_q8_reader", solution_path)
    if spec is None or spec.loader is None:
        temporary.cleanup()
        raise ImportError("cannot load staged package q8 reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    weights, codes, preset = module._read_export(package_dir / "weights_0.npz")
    temporary.cleanup()
    return weights, codes, preset


def _parity_member(
    zip_path: Path, label: str, level: str, output: Path, precision: str | None = None
) -> None:
    """Run one package precision/build combination on all twenty E457 streams."""
    os.environ["WNN_NATIVE_LEVEL"] = level
    if precision is None:
        os.environ.pop("WNN_PRECISION", None)
    else:
        os.environ["WNN_PRECISION"] = precision
    temporary, model = _read_package(zip_path)
    try:
        data_path = ROOT / "cache/valid_subset20.parquet"
        parity_rows = []
        for group, sequence, features, targets, need, mask in _stream_rows(
            data_path, list(range(20))
        ):
            del group
            predictions = _callback_rows(model, sequence, features, need)
            parity_rows.append((targets, predictions, mask, np.zeros(1, dtype=np.int64)))
        np.savez_compressed(output, **{f"pred_{index}": row[1] for index, row in enumerate(parity_rows)})
    finally:
        temporary.cleanup()


def _parity(output: Path, build_dir: Path) -> None:
    """Run independent represented-weight parity and native portability/reset gates."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    build_report = json.loads((build_dir / "build_report.json").read_text())
    if build_report.get("source", {}).get("source_zip_sha256") != SOURCE_SHA256:
        raise ValueError("q8 package build is not bound to the public E412 ZIP")
    if sha256_file(SOURCE_ZIP) != SOURCE_SHA256:
        raise ValueError("public E412 source ZIP changed after the q8 build")
    for label in VARIANTS:
        package_zip = build_dir / label / "submission.zip"
        expected_hash = build_report.get("variants", {}).get(label, {}).get("zip_sha256")
        if expected_hash is None or sha256_file(package_zip) != expected_hash:
            raise ValueError(f"{label}: package ZIP changed after the bounded build")
    e416_reference = _load_reference_helpers()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_path = ROOT / "cache/valid_subset20.parquet"
    e457_readiness_path = CAMPAIGN / "E457_head_transfer/readiness.json"
    if sha256_file(data_path) != E457_DATA_SHA256:
        raise ValueError("E462 parity data differs from the fixed E457 20-stream panel")
    e457_readiness = json.loads(e457_readiness_path.read_text())
    expected_sequences = list(map(int, e457_readiness["parent"]["dataset_identity"]["sequence_ids"]))
    groups = list(range(20))
    sequences = list(_stream_rows(data_path, groups))
    if (
        len(sequences) != 20
        or any(len(row[2]) != 20_000 for row in sequences)
        or [int(row[1]) for row in sequences] != expected_sequences
    ):
        raise ValueError("E457 parity panel is not twenty full 20k sequences")
    raw_features = np.stack([row[2] for row in sequences])
    features_sha = hashlib.sha256(np.ascontiguousarray(raw_features).tobytes()).hexdigest()
    from wnn.features import event_features

    event_digest = hashlib.sha256()
    for features in raw_features:
        event_digest.update(np.ascontiguousarray(event_features(features)).tobytes())
    targets = [row[3] for row in sequences]
    needs = [row[4] for row in sequences]
    masks = [row[5] for row in sequences]
    references: dict[str, np.ndarray] = {}
    export_records = {}
    for label, preset in VARIANTS.items():
        package_zip = build_dir / label / "submission.zip"
        weights, codes, actual_preset = _load_exported_weights(package_zip)
        if actual_preset != preset:
            raise ValueError(f"{label}: archived q8 preset changed")
        references[label] = np.stack(
            [e416_reference(weights, features) for features in raw_features]
        )
        export_records[label] = {
            "zip_sha256": sha256_file(package_zip),
            "effective_weights": sorted(weights),
            "codes": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                }
                for name, value in codes.items()
            },
        }

    _parity_member(SOURCE_ZIP, "i16_v3", "v3", output / "i16_v3.npz")
    for label in VARIANTS:
        for level in ("v2", "v3"):
            _parity_member(
                build_dir / label / "submission.zip",
                f"{label}_{level}",
                level,
                output / f"{label}_{level}.npz",
            )
            _parity_member(
                build_dir / label / "submission.zip",
                f"{label}_{level}_portable",
                level,
                output / f"{label}_{level}_portable.npz",
                precision=f"{label}_portable",
            )

    score_module = _load_score_module()
    report = {
        "schema": "e462-public-combo-q8-parity-v1",
        "source_zip_sha256": SOURCE_SHA256,
        "build_report_sha256": sha256_file(build_dir / "build_report.json"),
        "data_sha256": sha256_file(data_path),
        "raw_features_sha256": features_sha,
        "event_features_sha256": event_digest.hexdigest(),
        "rows": 400_000,
        "sequences": 20,
        "variants": {},
        "parity_gate": {"max_abs": MAX_ABS, "absolute_wp_delta": MAX_WP_DELTA},
    }
    base_predictions = _load_predictions(output / "i16_v3.npz", 20)
    for label in VARIANTS:
        outputs = {
            mode: _load_predictions(output / f"{label}_{mode}.npz", 20)
            for mode in ("v2", "v3", "v2_portable", "v3_portable")
        }
        if not all(
            np.array_equal(outputs["v2"], outputs[mode], equal_nan=True)
            for mode in ("v3", "v2_portable", "v3_portable")
        ):
            raise ValueError(f"{label}: portable/v2/v3 callback predictions differ")
        prediction = outputs["v3"]
        reference = references[label]
        required = np.broadcast_to(np.arange(20_000) >= 99, prediction.shape[:2])
        error = np.abs(prediction[required] - reference[required])
        reference_score = _score_arrays(score_module, targets, reference, masks)
        native_score = _score_arrays(score_module, targets, prediction, masks)
        i16_score = _score_arrays(score_module, targets, base_predictions, masks)
        max_abs = float(error.max())
        wp_delta = float(native_score["pooled"]["wp"] - reference_score["pooled"]["wp"])
        reset_ok, future_ok = _reset_and_future_gates(
            build_dir / label / "submission.zip", sequences[0], sequences[1]
        )
        report["variants"][label] = {
            "max_abs_native_vs_own_reference": max_abs,
            "wp_delta_native_vs_own_reference": wp_delta,
            "native_vs_reference": native_score,
            "reference": reference_score,
            "i16_control": i16_score,
            "q8_minus_i16_wp_descriptive": (
                native_score["pooled"]["wp"] - i16_score["pooled"]["wp"]
            ),
            "portable_v2_v3_bitwise": True,
            "reset_repeat_bitwise": reset_ok,
            "future_suffix_invariance": future_ok,
            "passed": bool(
                max_abs <= MAX_ABS
                and abs(wp_delta) <= MAX_WP_DELTA
                and reset_ok
                and future_ok
            ),
            "export": export_records[label],
        }
    report["passed"] = all(item["passed"] for item in report["variants"].values())
    report["scope"] = (
        "Twenty exposed full 20k streams; native q8 is compared to its own represented-weight "
        "Torch dense reference. Not an independent score confirmation."
    )
    _write_json(output / "parity_report.json", report)
    if not report["passed"]:
        raise RuntimeError("E462 q8 arithmetic/reset/causality gate failed")


def _load_predictions(path: Path, count: int) -> np.ndarray:
    """Load a fixed set of named per-sequence outputs from a parity archive."""
    with np.load(path, allow_pickle=False) as archive:
        predictions = np.stack([archive[f"pred_{index}"] for index in range(count)])
    if predictions.shape != (count, 20_000, 2) or not np.isfinite(predictions).all():
        raise ValueError(f"invalid callback prediction shape/content: {path}")
    return predictions


def _score_arrays(scorer: object, targets: list[np.ndarray], predictions: np.ndarray, masks: list[np.ndarray]) -> dict:
    """Score one fixed 20-stream prediction array with frozen metric arithmetic."""
    moments = []
    blocks = []
    selected = 0
    for y, prediction, mask in zip(targets, predictions, masks, strict=True):
        result = scorer.score_sequence(y, mask, {"p": prediction})
        moments.append(result["moments"]["p"])
        blocks.append(result["blocks"]["p"])
        selected += result["selected_rows"]
    return {
        "pooled": scorer.pooled(np.stack(moments)),
        "mean_per_sequence_wp": float(np.mean([value for value in blocks if value is not None])),
        "selected_rows": selected,
    }


def _load_score_module() -> object:
    """Load the campaign's frozen scoring arithmetic without run-sidecar identity checks."""
    path = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
    spec = importlib.util.spec_from_file_location("e462_campaign_score", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen campaign scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fresh_callback(zip_path: Path, level: str = "v3", precision: str | None = None):
    """Load an isolated package callback under one precise build/variant."""
    os.environ["WNN_NATIVE_LEVEL"] = level
    if precision is None:
        os.environ.pop("WNN_PRECISION", None)
    else:
        os.environ["WNN_PRECISION"] = precision
    return _read_package(zip_path)


def _reset_and_future_gates(
    zip_path: Path,
    first: tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[bool, bool]:
    """Check A,B,A reset replay and invariance to future raw-feature suffix changes."""
    group_a, seq_a, features_a, _target_a, need_a, _mask_a = first
    group_b, seq_b, features_b, _target_b, need_b, _mask_b = second
    del group_a, group_b, _target_a, _target_b, _mask_a, _mask_b
    temporary, model = _fresh_callback(zip_path)
    try:
        out_a1 = _callback_rows(model, seq_a, features_a, need_a)
        out_b = _callback_rows(model, seq_b, features_b, need_b)
        out_a2 = _callback_rows(model, seq_a, features_a, need_a)
        reset_ok = np.array_equal(out_a1, out_a2)
    finally:
        temporary.cleanup()
    changed = features_a.copy()
    changed[10_000:] += np.float32(0.375)
    temporary, model = _fresh_callback(zip_path)
    try:
        original = _callback_rows(model, seq_a, features_a, need_a)
    finally:
        temporary.cleanup()
    temporary, model = _fresh_callback(zip_path)
    try:
        suffix_changed = _callback_rows(model, seq_a, changed, need_a)
    finally:
        temporary.cleanup()
    del out_b
    future_ok = np.array_equal(original[:10_000], suffix_changed[:10_000])
    return reset_ok, future_ok


def _score_member(
    zip_path: Path,
    label: str,
    output: Path,
    build_dir: Path,
    parity_path: Path,
) -> None:
    """Stream one package over exactly the historically excluded 368 groups."""
    import pyarrow.parquet as parquet

    from wnn.splits import is_holdout
    from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH

    if output.exists():
        raise FileExistsError(output)
    build_report_path = build_dir / "build_report.json"
    lineage = _verify_composite_lineage(build_report_path, build_dir)
    build_report_sha256 = sha256_file(build_report_path)
    parity_report_sha256 = sha256_file(parity_path)
    target_path = ROOT / "cache/valid_targets.npz"
    score_module = _load_score_module()
    target = score_module.mmap_stored_npz(target_path)
    target_positions = {int(group): index for index, group in enumerate(target["row_group"])}
    data_path = ROOT / "datasets/valid.parquet"
    source = parquet.ParquetFile(data_path)
    sequence_ids = source.read(columns=["seq_ix"], use_threads=False).column(0).to_numpy()[::SEQUENCE_LENGTH]
    holdout_groups = [
        group for group, sequence_id in enumerate(sequence_ids) if is_holdout(int(sequence_id))
    ]
    if len(holdout_groups) != 368 or any(group not in target_positions for group in holdout_groups):
        raise ValueError("holdout row-group list differs from frozen 368-sequence target cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(holdout_groups), SEQUENCE_LENGTH, 2)
    )
    temporary, model = _fresh_callback(zip_path)
    sequence_records = []
    try:
        for index, group in enumerate(holdout_groups):
            table = source.read_row_group(
                int(group), columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS], use_threads=False
            )
            sequence_id = int(table.column("seq_ix")[0].as_py())
            features = np.column_stack(
                [table.column(name).to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
            ).astype(np.float32)
            need = table.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool)
            if len(features) != SEQUENCE_LENGTH or not np.isfinite(features).all():
                raise ValueError(f"invalid raw feature sequence in holdout row group {group}")
            prediction_file[index] = _callback_rows(model, sequence_id, features, need)
            sequence_records.append(
                {
                    "row_group": int(group),
                    "seq_ix": sequence_id,
                    "target_position": target_positions[int(group)],
                }
            )
    finally:
        prediction_file.flush()
        del prediction_file
        temporary.cleanup()
    _write_json(
        output.with_suffix(".json"),
        {
            "schema": "e462-package-prediction-v1",
            "candidate": label,
            "zip_sha256": sha256_file(zip_path),
            "composite_lineage": lineage,
            "build_report_sha256": build_report_sha256,
            "parity_report_sha256": parity_report_sha256,
            "target_cache_sha256": sha256_file(target_path),
            "dataset_sha256": sha256_file(data_path),
            "sequence_records": sequence_records,
            "prediction_sha256": sha256_file(output),
            "provenance_type": "converted-artifact-composite-lineage; not a training-run sidecar",
        },
    )


def _verify_composite_lineage(build_report_path: Path, build_dir: Path) -> dict:
    """Recheck frozen two-checkpoint lineage from direct source records, never fake sidecars."""
    report = json.loads(build_report_path.read_text())
    lineage = report.get("source", {})
    if lineage.get("source_zip_sha256") != SOURCE_SHA256:
        raise ValueError("build proof lost the fixed public ZIP identity")
    if sha256_file(SOURCE_ZIP) != SOURCE_SHA256:
        raise ValueError("public E412 source ZIP changed after the q8 build")
    paths = {
        "parent_checkpoint": ROOT / "artifacts/runs" / PARENT_RUN / "model_ep3.pt",
        "parent_provenance": ROOT / "artifacts/runs" / PARENT_RUN / "provenance.json",
        "head_checkpoint": ROOT / "artifacts/runs" / HEAD_RUN / "model_ep0.pt",
        "head_provenance": ROOT / "artifacts/runs" / HEAD_RUN / "provenance.json",
        "parent_job": CAMPAIGN / "runner/attempts/e412-combo-final-s3-v2.json",
        "head_job": CAMPAIGN / "runner/attempts/e412-final-head-v1.json",
    }
    expected = {
        "parent_checkpoint": PARENT_SHA256,
        "parent_provenance": lineage["parent"]["provenance_sha256"],
        "head_checkpoint": HEAD_SHA256,
        "head_provenance": lineage["head"]["provenance_sha256"],
        "parent_job": lineage["parent_job_sha256"],
        "head_job": lineage["head_job_sha256"],
    }
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"source lineage changed after E462 build: {name}")
    parent_job = json.loads(paths["parent_job"].read_text())
    head_job = json.loads(paths["head_job"].read_text())
    for job, checkpoint, checkpoint_hash in (
        (parent_job, "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt", PARENT_SHA256),
        (head_job, "artifacts/runs/c20261004_combo_head_s3_final/model_ep0.pt", HEAD_SHA256),
    ):
        artifacts = {entry["path"]: entry["sha256"] for entry in job.get("artifacts", [])}
        if job.get("status") != "finished" or job.get("exit_code") != 0:
            raise ValueError("original parent/head fit job is not recorded successful")
        if artifacts.get(checkpoint) != checkpoint_hash:
            raise ValueError("original fit job does not bind its fixed checkpoint")
    for run, path, epoch, expected_init in (
        (PARENT_RUN, paths["parent_provenance"], 3, None),
        (HEAD_RUN, paths["head_provenance"], 0, f"{PARENT_RUN}@ep3"),
    ):
        provenance = json.loads(path.read_text())
        args = provenance.get("args", {})
        actual_init = args.get("init")
        if expected_init is None and actual_init == "":
            actual_init = None
        if (
            args.get("name") != run
            or args.get("checkpoint_policy") != "fixed"
            or args.get("checkpoint_epoch") != epoch
            or provenance.get("selected_epoch") != epoch
            or actual_init != expected_init
            or args.get("include_heldout_valid")
        ):
            raise ValueError(f"{run}: source provenance no longer matches fixed ancestry")
    if json.loads(paths["head_provenance"].read_text()).get("initial_checkpoint_sha256") != PARENT_SHA256:
        raise ValueError("head provenance does not point to the exact parent checkpoint")
    for label in VARIANTS:
        candidate = build_dir / label / "submission.zip"
        expected_zip_hash = report.get("variants", {}).get(label, {}).get("zip_sha256")
        if expected_zip_hash is None or sha256_file(candidate) != expected_zip_hash:
            raise ValueError(f"{label}: package ZIP changed after the q8 build")
        with zipfile.ZipFile(candidate) as archive:
            package_proof = json.loads(archive.read("E462_PROVENANCE.json"))
        if package_proof.get("lineage") != lineage:
            raise ValueError(f"{label}: package composite lineage differs from build proof")
    return lineage


def _validate_parity_gate(build_dir: Path, parity_path: Path) -> dict:
    """Require the immutable full-stream q8 parity report before any score replay."""
    report = json.loads(parity_path.read_text())
    if (
        report.get("schema") != "e462-public-combo-q8-parity-v1"
        or report.get("source_zip_sha256") != SOURCE_SHA256
        or report.get("build_report_sha256") != sha256_file(build_dir / "build_report.json")
        or report.get("rows") != 400_000
        or report.get("sequences") != 20
        or not report.get("passed")
    ):
        raise ValueError("E462 full-stream parity gate is missing, failed or bound to another build")
    for label in VARIANTS:
        item = report.get("variants", {}).get(label, {})
        expected_zip_hash = json.loads((build_dir / "build_report.json").read_text()).get(
            "variants", {}
        ).get(label, {}).get("zip_sha256")
        if not item.get("passed") or item.get("export", {}).get("zip_sha256") != expected_zip_hash:
            raise ValueError(f"{label}: parity report does not pass for the current package")
    return report


def _score_report(build_dir: Path, output: Path, parity_path: Path) -> None:
    """Score all three immutable callback arrays with paired component uncertainty."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    parity = _validate_parity_gate(build_dir, parity_path)
    build_report_path = build_dir / "build_report.json"
    lineage = _verify_composite_lineage(build_report_path, build_dir)
    build_report_sha256 = sha256_file(build_report_path)
    parity_report_sha256 = sha256_file(parity_path)
    score_module = _load_score_module()
    from wnn.splits import load_campaign_split

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = load_campaign_split(split_path, ROOT)
    target_path = ROOT / "cache/valid_targets.npz"
    target = score_module.mmap_stored_npz(target_path)
    groups = [int(group) for group in target["row_group"]]
    target_positions = {group: index for index, group in enumerate(groups)}
    from wnn.splits import is_holdout

    sequence_ids = target.get("seq_ix")
    if sequence_ids is None:
        source = pq.ParquetFile(ROOT / "datasets/valid.parquet")
        sequence_ids = source.read(columns=["seq_ix"], use_threads=False).column(0).to_numpy()[
            ::20_000
        ]
    package_paths = {
        "i16": SOURCE_ZIP,
        "q8r": build_dir / "q8r/submission.zip",
        "q8x": build_dir / "q8x/submission.zip",
    }
    package_hashes = {name: sha256_file(path) for name, path in package_paths.items()}
    holdout_groups = {
        int(group)
        for group, sequence_id in zip(groups, sequence_ids, strict=True)
        if is_holdout(int(sequence_id))
    }
    for run_name in ("parent", "head"):
        fitted = set(map(int, lineage[run_name]["fit_valid_row_groups"]))
        if fitted & holdout_groups:
            raise ValueError(f"{run_name} ancestry contains historical holdout fit rows")
    holdout_groups = [group for group in groups if group in holdout_groups]
    component_map = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    if len(holdout_groups) != 368 or any(group not in component_map for group in holdout_groups):
        raise ValueError("holdout sequence/component identity mismatch")
    classes, component_ids = [], []
    moments_by_case: dict[str, list[np.ndarray]] = {name: [] for name in ("i16", *VARIANTS)}
    blocks_by_case: dict[str, list[float | None]] = {name: [] for name in moments_by_case}
    selected_rows = 0
    reports = {}
    for label in ("i16", *VARIANTS):
        prediction_path = output.parent / "predictions" / f"pred_{label}.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        if (
            sidecar.get("candidate") != label
            or sidecar.get("zip_sha256") != package_hashes[label]
            or sidecar.get("composite_lineage") != lineage
            or sidecar.get("build_report_sha256") != build_report_sha256
            or sidecar.get("parity_report_sha256") != parity_report_sha256
            or sidecar.get("prediction_sha256") != sha256_file(prediction_path)
            or sidecar.get("target_cache_sha256") != sha256_file(target_path)
            or sidecar.get("dataset_sha256") != sha256_file(ROOT / "datasets/valid.parquet")
            or sidecar.get("provenance_type")
            != "converted-artifact-composite-lineage; not a training-run sidecar"
            or len(sidecar.get("sequence_records", [])) != 368
        ):
            raise ValueError(f"{label}: prediction composite record mismatch")
        predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if predictions.shape != (368, 20_000, 2) or predictions.dtype != np.float32:
            raise ValueError(f"{label}: malformed holdout prediction array")
        for index, record in enumerate(sidecar["sequence_records"]):
            group = int(record["row_group"])
            position = target_positions.get(group)
            if (
                group != holdout_groups[index]
                or position is None
                or record.get("target_position") != position
                or record.get("seq_ix") != int(sequence_ids[position])
            ):
                raise ValueError(f"{label}: prediction row order differs from target cache")
            result = score_module.score_sequence(
                target["y"][position], target["mask"][position], {label: predictions[index]}
            )
            moments_by_case[label].append(result["moments"][label])
            blocks_by_case[label].append(result["blocks"][label])
            if label == "i16":
                selected_rows += result["selected_rows"]
                classes.append(int(target["market_class"][position]))
                component_ids.append(component_map[group])
        reports[label] = {
            "zip_sha256": sidecar["zip_sha256"],
            "prediction_sha256": sidecar["prediction_sha256"],
            "build_report_sha256": sidecar["build_report_sha256"],
            "parity_report_sha256": sidecar["parity_report_sha256"],
            "composite_lineage": sidecar["composite_lineage"],
        }
        del predictions
    design = {
        "comparisons": [["q8r", "i16"], ["q8x", "i16"]],
        "n_boot": 2000,
        "bootstrap_seed": 271828,
    }
    moments = {name: np.stack(values) for name, values in moments_by_case.items()}
    soft = {"0.02": {}, "0.05": {}}
    summary = score_module.summary(
        moments,
        soft,
        blocks_by_case,
        np.asarray(classes),
        np.asarray(component_ids),
        design,
    )
    result = {
        "schema": "e462-public-combo-q8-holdout-score-v1",
        "metric_id": score_module.METRIC_ID,
        "metric_code_sha256": sha256_file(
            ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
        ),
        "split_manifest_sha256": sha256_file(split_path),
        "target_cache_sha256": sha256_file(target_path),
        "dataset_sha256": sha256_file(ROOT / "datasets/valid.parquet"),
        "composite_lineage": lineage,
        "parity_report_sha256": sha256_file(parity_path),
        "parity_passed": parity["passed"],
        "candidates": reports,
        "n_sequences": len(holdout_groups),
        "n_components": len(set(component_ids)),
        "selected_rows": selected_rows,
        "pooled_and_diagnostics": summary,
        "scope": (
            "Historically excluded but previously inspected holdout; paired descriptive score only. "
            "No selection, tuning, public-score transfer, or upload."
        ),
    }
    _write_json(output / "score_report.json", result)


def run_score(build_dir: Path, output: Path, parity_path: Path) -> None:
    """Run separate exact package callbacks, then pair their holdout moments."""
    if output.exists():
        raise FileExistsError(output)
    _validate_parity_gate(build_dir, parity_path)
    _verify_composite_lineage(build_dir / "build_report.json", build_dir)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    packages = {
        "i16": SOURCE_ZIP,
        "q8r": build_dir / "q8r/submission.zip",
        "q8x": build_dir / "q8x/submission.zip",
    }
    for label, zip_path in packages.items():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "score-member",
            "--zip",
            str(zip_path),
            "--label",
            label,
            "--build-dir",
            str(build_dir),
            "--parity-report",
            str(parity_path),
            "--output",
            str(prediction_dir / f"pred_{label}.npy"),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    _score_report(build_dir, output / "report", parity_path)


def main() -> None:
    """Run only the explicitly selected E462 parity or score phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("parity", "parity-member", "score", "score-member", "score-report"),
        required=True,
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--label", choices=("i16", "q8r", "q8x"))
    parser.add_argument("--level", choices=("v2", "v3"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "parity":
        if args.build_dir is None:
            parser.error("parity requires --build-dir")
        _parity(output, args.build_dir.resolve())
    elif args.phase == "parity-member":
        if args.zip is None or args.label is None or args.level is None:
            parser.error("parity-member requires --zip, --label, and --level")
        _parity_member(args.zip.resolve(), args.label, args.level, output)
    elif args.phase == "score":
        if args.build_dir is None or args.parity_report is None:
            parser.error("score requires --build-dir and --parity-report")
        run_score(args.build_dir.resolve(), output, args.parity_report.resolve())
    elif args.phase == "score-member":
        if args.zip is None or args.label is None or args.build_dir is None or args.parity_report is None:
            parser.error("score-member requires --zip, --label, --build-dir and --parity-report")
        build_dir = args.build_dir.resolve()
        _validate_parity_gate(build_dir, args.parity_report.resolve())
        _verify_composite_lineage(build_dir / "build_report.json", build_dir)
        expected_zip = (
            SOURCE_ZIP
            if args.label == "i16"
            else build_dir / args.label / "submission.zip"
        )
        if args.zip.resolve() != expected_zip.resolve():
            parser.error("score-member ZIP does not match its fixed label/build")
        _score_member(
            args.zip.resolve(),
            args.label,
            output,
            build_dir,
            args.parity_report.resolve(),
        )
    else:
        if args.build_dir is None or args.parity_report is None:
            parser.error("score-report requires --build-dir and --parity-report")
        _score_report(args.build_dir.resolve(), output, args.parity_report.resolve())


if __name__ == "__main__":
    main()
