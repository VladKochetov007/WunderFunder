#!/usr/bin/env python3
"""Stream official metric and raw target/prediction diagnostics by frozen F0 subset."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "src"))

from wnn.gmetric import METRIC_CLIP, METRIC_ID, WARMUP, GlobalAccumulator

CAMPAIGN = ROOT / "research/campaign_2026-10-15/student_quality_tradeoff_r1"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGETS_PATH = ROOT / "cache/valid_targets.npz"
TARGETS_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
ROWS = 20_000
CHUNK_ROWS = 2_000
BATCH_GROUPS = 16
RUNS = (
    ("alpha0_cp1", "student_f0_warm_s3_alpha0_r1_p1", 1, "f0_grade_alpha0_r2/ep2"),
    ("alpha09_cp1", "clean_t3_f0_control_s3_a9_t3r5_r2_p1", 1, "f0_grade_alpha09_r2/ep2"),
    ("alpha0_cp3", "student_f0_warm_s3_alpha0_r1_p2_envr1", 3, "f0_grade_alpha0_r2/ep4"),
    ("alpha09_cp3", "clean_t3_f0_control_s3_a9_t3r5_r2_p2_envr1", 3, "f0_grade_alpha09_r2/ep4"),
    ("alpha0_cp5", "student_f0_warm_s3_alpha0_r1_p3_envr1", 5, "f0_grade_alpha0_r2/ep6"),
    ("alpha09_cp5", "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1", 5, ""),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_header(stream) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    version = np.lib.format.read_magic(stream)
    shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
    return tuple(shape), bool(fortran_order), np.dtype(dtype), stream.tell()


def read_exact(stream, count: int) -> bytes:
    blocks = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ValueError("truncated NPY member or prediction file")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def read_small_member(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as stream:
        shape, fortran_order, dtype, _ = npy_header(stream)
        if fortran_order or dtype.kind not in "iu" or math.prod(shape) > 10_000:
            raise ValueError(f"unexpected small NPY member: {name}")
        raw = read_exact(stream, math.prod(shape) * dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def target_stream(archive: zipfile.ZipFile, name: str, expected_shape: tuple[int, ...]):
    stream = archive.open(name)
    shape, fortran_order, dtype, _ = npy_header(stream)
    if shape != expected_shape or fortran_order:
        raise ValueError(f"unexpected target member shape/order for {name}: {shape}")
    return stream, dtype


def main() -> None:
    output = CAMPAIGN / "ALPHA0_FIT_EVAL_DIAGNOSTIC_R1.json"
    if output.exists():
        raise FileExistsError(output)
    split_raw = SPLIT_PATH.read_bytes()
    split_sha = hashlib.sha256(split_raw).hexdigest()
    if split_sha != SPLIT_SHA256:
        raise ValueError("fold-0 split bytes changed")
    split = json.loads(split_raw)
    fit_valid = set(map(int, split["fit"]["valid"]))
    evaluation = set(map(int, split["evaluation"]["valid"]))
    if len(fit_valid) != 1_405 or len(evaluation) != 468 or fit_valid & evaluation:
        raise ValueError("unexpected frozen F0 fit-valid/evaluation memberships")

    targets_sha = sha256_file(TARGETS_PATH)
    if targets_sha != TARGETS_SHA256:
        raise ValueError("valid-target cache bytes changed")

    names = [f"pred_{run}_valid.npy" for _, run, _, _ in RUNS]
    sidecars = []
    pred_streams = []
    pred_hashes = []
    pred_shapes = []
    pred_dtypes = []
    pred_bytes = []
    for label, run, epoch, grade_dir in RUNS:
        pred_path = ROOT / "cache" / f"pred_{run}_valid.npy"
        sidecar_path = pred_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        if (
            sidecar.get("run") != run
            or int(sidecar.get("epoch")) != epoch
            or sidecar.get("split") != "valid"
            or sidecar.get("split_manifest_sha256") != split_sha
            or len(sidecar.get("row_groups", [])) != 1_873
        ):
            raise ValueError(f"prediction identity/order mismatch: {run}")
        stream = pred_path.open("rb")
        shape, fortran_order, dtype, header_bytes = npy_header(stream)
        if shape != (1_873, ROWS, 2) or fortran_order or dtype.kind != "f" or dtype.itemsize != 4:
            raise ValueError(f"unexpected prediction NPY header: {run}, {shape}, {dtype}")
        stream.seek(0)
        prefix = read_exact(stream, header_bytes)
        stream.seek(header_bytes)
        sidecars.append(sidecar)
        pred_streams.append(stream)
        pred_hashes.append(hashlib.sha256(prefix))
        pred_shapes.append(shape)
        pred_dtypes.append(dtype)
        pred_bytes.append(pred_path.stat().st_size)

    groups_per_subset = {"fit_valid_in_sample": 0, "eval_exposed": 0}
    official = {
        label: {"fit_valid_in_sample": GlobalAccumulator(), "eval_exposed": GlobalAccumulator()}
        for label, _, _, _ in RUNS
    }
    # Raw target mean/RMS and raw prediction clipping pressure by 2,000-row time block.
    stats = {}
    for label, _, _, _ in RUNS:
        for subset in ("fit_valid_in_sample", "eval_exposed"):
            stats[label, subset] = [
                [
                    {"n": 0, "y_sum": 0.0, "y_sq_sum": 0.0, "pred_abs_gt2": 0}
                    for _target in range(2)
                ]
                for _chunk in range(ROWS // CHUNK_ROWS)
            ]

    with zipfile.ZipFile(TARGETS_PATH) as archive:
        row_groups = read_small_member(archive, "row_group.npy").astype(np.int64, copy=False)
        seq_ix = read_small_member(archive, "seq_ix.npy").astype(np.int64, copy=False)
        if row_groups.shape != (1_873,) or seq_ix.shape != (1_873,):
            raise ValueError("unexpected target-cache sequence index dimensions")
        if len(set(map(int, row_groups))) != 1_873 or set(map(int, row_groups)) != fit_valid | evaluation:
            raise ValueError("target cache row groups do not match frozen split membership")
        for sidecar in sidecars:
            if list(map(int, sidecar["row_groups"])) != list(map(int, row_groups)):
                raise ValueError("prediction row order differs from valid-target cache")
        y_stream, y_dtype = target_stream(archive, "y.npy", (1_873, ROWS, 2))
        mask_stream, mask_dtype = target_stream(archive, "mask.npy", (1_873, ROWS))
        if y_dtype.kind != "f" or y_dtype.itemsize != 4 or mask_dtype.kind != "b":
            raise ValueError("unexpected target/mask dtypes")
        y_group_bytes = ROWS * 2 * y_dtype.itemsize
        mask_group_bytes = ROWS * mask_dtype.itemsize
        for first in range(0, len(row_groups), BATCH_GROUPS):
            last = min(first + BATCH_GROUPS, len(row_groups))
            count = last - first
            y_raw = read_exact(y_stream, count * y_group_bytes)
            mask_raw = read_exact(mask_stream, count * mask_group_bytes)
            y = np.frombuffer(y_raw, dtype=y_dtype).reshape(count, ROWS, 2)
            mask = np.frombuffer(mask_raw, dtype=mask_dtype).reshape(count, ROWS).astype(bool, copy=False)
            ids = row_groups[first:last]
            fit_select = np.fromiter((int(g) in fit_valid for g in ids), bool, count=count)
            eval_select = np.fromiter((int(g) in evaluation for g in ids), bool, count=count)
            if int(fit_select.sum()) + int(eval_select.sum()) != count:
                raise ValueError("unclassified validation sequence in target cache")
            groups_per_subset["fit_valid_in_sample"] += int(fit_select.sum())
            groups_per_subset["eval_exposed"] += int(eval_select.sum())
            scored = mask & (np.arange(ROWS) >= WARMUP)[None, :]
            for policy_i, (label, run, _epoch, _grade_dir) in enumerate(RUNS):
                raw = read_exact(pred_streams[policy_i], count * y_group_bytes)
                pred_hashes[policy_i].update(raw)
                pred = np.frombuffer(raw, dtype=pred_dtypes[policy_i]).reshape(count, ROWS, 2)
                if not np.isfinite(pred[scored]).all():
                    raise ValueError(f"nonfinite scored prediction for {run}")
                for subset, select in (("fit_valid_in_sample", fit_select), ("eval_exposed", eval_select)):
                    if select.any():
                        official[label][subset].add(y[select], pred[select], scored[select])
                    for chunk_i in range(ROWS // CHUNK_ROWS):
                        lo, hi = chunk_i * CHUNK_ROWS, (chunk_i + 1) * CHUNK_ROWS
                        local_mask = scored[select, lo:hi]
                        if not local_mask.any():
                            continue
                        local_y = y[select, lo:hi]
                        local_pred = pred[select, lo:hi]
                        for side in range(2):
                            y_values = local_y[:, :, side][local_mask]
                            pred_values = local_pred[:, :, side][local_mask]
                            record = stats[label, subset][chunk_i][side]
                            record["n"] += int(y_values.size)
                            record["y_sum"] += float(y_values.sum(dtype=np.float64))
                            record["y_sq_sum"] += float(np.square(y_values, dtype=np.float64).sum())
                            record["pred_abs_gt2"] += int(np.count_nonzero(np.abs(pred_values) > METRIC_CLIP))
    for stream in [*pred_streams, y_stream, mask_stream]:
        stream.close()

    results = {}
    for label, run, epoch, grade_dir in RUNS:
        results[label] = {"run": run, "epoch_index": epoch, "subsets": {}}
        for subset in ("fit_valid_in_sample", "eval_exposed"):
            result = dict(official[label][subset].result())
            chunks = []
            for chunk_i, pair in enumerate(stats[label, subset]):
                targets_summary = []
                for side, item in enumerate(pair):
                    n = item["n"]
                    targets_summary.append(
                        {
                            "target": side,
                            "scored_rows": n,
                            "raw_target_mean": item["y_sum"] / n if n else None,
                            "raw_target_rms": math.sqrt(item["y_sq_sum"] / n) if n else None,
                            "raw_prediction_abs_gt2_fraction": item["pred_abs_gt2"] / n if n else None,
                        }
                    )
                chunks.append({"rows": [chunk_i * CHUNK_ROWS, (chunk_i + 1) * CHUNK_ROWS], "targets": targets_summary})
            results[label]["subsets"][subset] = {"official_metric": result, "time_chunks": chunks}

    for index, (label, run, epoch, grade_dir) in enumerate(RUNS):
        if pred_hashes[index].hexdigest() != sidecars[index].get("prediction_sha256"):
            raise ValueError(f"prediction bytes differ from sidecar hash: {run}")
        results[label]["prediction_sha256"] = pred_hashes[index].hexdigest()
        results[label]["prediction_bytes"] = pred_bytes[index]
        if grade_dir:
            grade_path = CAMPAIGN / grade_dir / f"{run}.grade.json"
            grade = json.loads(grade_path.read_text())
            observed = results[label]["subsets"]["eval_exposed"]["official_metric"]
            expected = grade["result"]
            if (
                grade.get("run") != run
                or int(grade.get("epoch")) != epoch
                or grade.get("prediction_sha256") != results[label]["prediction_sha256"]
                or grade.get("split_manifest_sha256") != split_sha
            ):
                raise ValueError(f"saved grade identity does not bind the streamed prediction: {run}")
            results[label]["existing_eval_grade"] = expected
            results[label]["eval_wp_difference_from_existing_grade"] = float(observed["wp"]) - float(expected["wp"])
            if abs(results[label]["eval_wp_difference_from_existing_grade"]) > 2e-7:
                raise ValueError(f"streamed eval metric does not reproduce the saved grade: {run}")

    cp5_reference = json.loads((CAMPAIGN / "CURRENT_T3_PARENT_PAIRED_GRADE_R1.json").read_text())
    a9_cp5 = results["alpha09_cp5"]["subsets"]["eval_exposed"]["official_metric"]
    a9_cp5_sha = results["alpha09_cp5"]["prediction_sha256"]
    if cp5_reference["candidate_members"][0]["prediction_sha256"] != a9_cp5_sha:
        raise ValueError("A9 CP5 prediction identity disagrees with the frozen paired grade")
    results["alpha09_cp5"]["existing_eval_grade"] = cp5_reference["official"]
    results["alpha09_cp5"]["eval_wp_difference_from_existing_grade"] = float(a9_cp5["wp"]) - float(cp5_reference["official"]["wp"])
    if abs(results["alpha09_cp5"]["eval_wp_difference_from_existing_grade"]) > 2e-7:
        raise ValueError("streamed A9 CP5 eval metric does not reproduce the frozen grade")

    contrasts = []
    for cp in (1, 3, 5):
        a0, a9 = results[f"alpha0_cp{cp}"], results[f"alpha09_cp{cp}"]
        contrasts.append(
            {
                "epoch_index": cp,
                "fit_valid_alpha0_minus_alpha09_wp": a0["subsets"]["fit_valid_in_sample"]["official_metric"]["wp"]
                - a9["subsets"]["fit_valid_in_sample"]["official_metric"]["wp"],
                "eval_alpha0_minus_alpha09_wp": a0["subsets"]["eval_exposed"]["official_metric"]["wp"]
                - a9["subsets"]["eval_exposed"]["official_metric"]["wp"],
            }
        )
    record = {
        "schema": "wnn.alpha0-fit-valid-vs-exposed-eval-diagnostic.r1",
        "status": "completed_descriptive_diagnostic",
        "metric_id": METRIC_ID,
        "official_metric": "global_wp_clip2_v1 over pooled scored rows with target clipping +/-2 and absolute clipped target weights",
        "membership": {
            "fold": 0,
            "split_manifest_sha256": split_sha,
            "source": "explicit fold0.json fit.valid and evaluation.valid row-group lists joined to valid_targets.npz row_group in target order",
            "fit_valid_groups": groups_per_subset["fit_valid_in_sample"],
            "evaluation_groups": groups_per_subset["eval_exposed"],
            "selection_labels": {
                "fit_valid_in_sample": "reused labels from the 1,405 valid groups included in fitting; in-sample diagnostic",
                "eval_exposed": "the 468 fixed F0 evaluation groups; repeatedly inspected exposed development evidence, not fresh confirmation",
            },
            "row_mask": "frozen valid_targets mask AND row index >= 99",
        },
        "targets": {"path": str(TARGETS_PATH.relative_to(ROOT)), "sha256": targets_sha},
        "streaming": {"sequence_rows": ROWS, "group_batch": BATCH_GROUPS, "temporal_chunk_rows": CHUNK_ROWS, "prediction_hashes_recomputed_during_stream": True},
        "source": {
            "script_sha256": sha256_file(Path(__file__)),
            "gmetric_sha256": sha256_file(ROOT / "src/wnn/gmetric.py"),
            "metric_clip": METRIC_CLIP,
            "warmup_rows_excluded": WARMUP,
        },
        "results": results,
        "stage_contrasts_alpha0_minus_alpha09": contrasts,
        "limits": [
            "Descriptive reuse of fitting groups versus exposed evaluation groups; no causal generalization, fresh-CV or public-transfer claim.",
            "Official metric scores are streamed by batches through the unchanged GlobalAccumulator; fixed-prediction/one-seed evidence only.",
            "Raw target mean/RMS are unweighted and un-clipped; prediction exceedance is the fraction of scored rows with absolute raw prediction greater than two.",
        ],
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(output), "stage_contrasts_alpha0_minus_alpha09": contrasts}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
