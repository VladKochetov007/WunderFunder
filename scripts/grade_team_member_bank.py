"""Grade frozen FP32 team-member exports and their predeclared mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import wnn.gmetric as gmetric_module  # noqa: E402
from scripts.team_member_grade_inputs import (  # noqa: E402
    MemberInput,
    _evict_maps,
    _read_target_group,
    _validate_member_result,
)
from scripts.team_member_grade_support import (  # noqa: E402
    EXPECTED_ROW_GROUPS,
    N_OUTPUTS,
    TARGET_COLUMNS,
    CovarianceAccumulator,
    _as_int,
    _as_object,
    _as_sha256,
    _as_string,
    _plan_records,
    _read_json,
    blend_predictions,
    digest_file,
    stable_file_identity,
)
from wnn.data import SEQUENCE_LENGTH, WARMUP  # noqa: E402
from wnn.gmetric import METRIC_CLIP, METRIC_ID, GlobalAccumulator  # noqa: E402

logger = logging.getLogger("grade_team_member_bank")


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _grader_source_records() -> dict[str, dict[str, str]]:
    paths = {
        "entrypoint": Path(__file__).resolve(),
        "input_validation": Path(__file__).with_name("team_member_grade_inputs.py").resolve(),
        "scoring_support": Path(__file__).with_name("team_member_grade_support.py").resolve(),
        "npy_header_helper": Path(__file__).with_name("alias_grade_prediction.py").resolve(),
        "official_metric": Path(gmetric_module.__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": digest_file(path)}
        for name, path in paths.items()
        if path.is_file()
    }


def _parse_args() -> argparse.Namespace:
    campaign = ROOT / "research/campaign_2026-10-13/team_models_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=campaign / "MEMBER_BANK_PLAN_R1.json")
    parser.add_argument("--inventory", type=Path, default=campaign / "INVENTORY.json")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/valid.parquet")
    parser.add_argument("--bank-root", type=Path, default=campaign / "member_bank")
    parser.add_argument("--out", type=Path, required=True, help="new grade output directory")
    parser.add_argument("--first-group", type=int, default=0)
    parser.add_argument("--group-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Validate a terminal member bank, score fixed outputs, and save diagnostics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.out.resolve().is_relative_to(args.bank_root.resolve()):
        raise ValueError("grade output must be isolated outside the member-bank directory")
    if args.first_group < 0:
        raise ValueError("first group cannot be negative")
    if args.group_count is None:
        if args.first_group != 0:
            raise ValueError("a nonzero first group requires an explicit group count")
        group_count = EXPECTED_ROW_GROUPS
    else:
        group_count = args.group_count
    if group_count < 1 or args.first_group + group_count > EXPECTED_ROW_GROUPS:
        raise ValueError("requested group range is outside the complete validation set")
    group_ids = tuple(range(args.first_group, args.first_group + group_count))
    grader_sources = _grader_source_records()
    if len(grader_sources) != 5:
        raise ValueError("one or more grader source files are missing")

    plan_sha256 = digest_file(args.plan)
    inventory_sha256 = digest_file(args.inventory)
    plan = _read_json(args.plan)
    inventory = _read_json(args.inventory)
    if digest_file(args.plan) != plan_sha256 or digest_file(args.inventory) != inventory_sha256:
        raise ValueError("plan or inventory changed while the grader loaded it")
    plan_members, mixtures = _plan_records(plan)
    inventory_data = _as_object(inventory.get("data"), "inventory.data")
    data_sha256 = _as_sha256(inventory_data.get("sha256"), "inventory.data.sha256")
    data_bytes = _as_int(inventory_data.get("bytes"), "inventory.data.bytes")
    if args.data.stat().st_size != data_bytes:
        raise ValueError("validation data size differs from the frozen inventory")
    data_stat = args.data.stat()
    data_stat_identity = (
        data_stat.st_dev,
        data_stat.st_ino,
        data_stat.st_size,
        data_stat.st_mtime_ns,
    )
    parquet = pq.ParquetFile(args.data)
    if parquet.num_row_groups != EXPECTED_ROW_GROUPS:
        raise ValueError(f"expected {EXPECTED_ROW_GROUPS} validation row groups")
    schema_names = set(parquet.schema_arrow.names)
    required_columns = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", *TARGET_COLUMNS}
    if not required_columns.issubset(schema_names):
        raise ValueError("validation Parquet lacks required labels, mask, or order columns")

    members: list[MemberInput] = []
    source_digests: dict[str, str] | None = None
    source_paths: dict[str, str] | None = None
    source_records: list[dict[str, object]] = []
    environment: dict[str, object] | None = None
    try:
        for record in plan_members:
            loaded, dep_digests, dep_paths, source_record = _validate_member_result(
                record,
                inventory,
                args.inventory,
                inventory_sha256,
                args.data,
                data_sha256,
                group_ids,
                args.bank_root,
            )
            members.append(loaded)
            if source_digests is None:
                source_digests, source_paths = dep_digests, dep_paths
            elif dep_digests != source_digests or dep_paths != source_paths:
                raise ValueError("member exports used different source dependency revisions")
            result_environment = _as_object(loaded.result.get("environment"), "result.environment")
            if environment is None:
                environment = result_environment
            elif result_environment != environment:
                raise ValueError("member exports used different runtime environments")
            source_records.append(source_record)
        if source_digests is None or source_paths is None or environment is None:
            raise ValueError("frozen member bank is empty")
        for name, path_string in source_paths.items():
            path = Path(path_string)
            if not path.is_file() or digest_file(path) != source_digests[name]:
                raise ValueError(f"current source dependency differs from export: {name}")

        member_ids = [_as_string(record.get("id"), "member.id") for record in plan_members]
        member_lookup = {
            member_id: item for member_id, item in zip(member_ids, members, strict=True)
        }
        unit_names = [*member_ids, *mixtures]
        graders = {name: GlobalAccumulator() for name in unit_names}
        target_hash = hashlib.sha256()
        mask_hash = hashlib.sha256()
        order_hash = hashlib.sha256()
        sequence_ids: list[int] = []
        seen_sequence_ids: set[int] = set()
        prediction_correlations = [CovarianceAccumulator(len(unit_names)) for _ in range(N_OUTPUTS)]
        residual_correlations = [CovarianceAccumulator(len(unit_names)) for _ in range(N_OUTPUTS)]
        started = time.monotonic()

        for output_index, group in enumerate(group_ids):
            seq_ix, targets, mask = _read_target_group(parquet, group)
            if seq_ix in seen_sequence_ids:
                raise ValueError(f"source sequence ID is duplicated at row group {group}")
            seen_sequence_ids.add(seq_ix)
            sequence_ids.append(seq_ix)
            target_hash.update(targets.tobytes(order="C"))
            mask_hash.update(mask.tobytes(order="C"))
            order_hash.update(np.asarray([group, seq_ix], dtype="<i8").tobytes())

            for item in members:
                group_result = item.group_results[output_index]
                if _as_int(group_result.get("seq_ix"), "group.seq_ix") != seq_ix:
                    raise ValueError(
                        f"sequence ordering differs for {item.plan_record['id']} at group {group}"
                    )
            unit_predictions = np.empty(
                (SEQUENCE_LENGTH, len(unit_names), N_OUTPUTS), dtype=np.float32
            )
            prediction_views: dict[str, np.ndarray] = {}
            for member_index, member_id in enumerate(member_ids):
                item = member_lookup[member_id]
                prediction = item.predictions[output_index]
                if (
                    prediction.shape != (SEQUENCE_LENGTH, N_OUTPUTS)
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError(
                        f"invalid or non-finite predictions for {member_id} at group {group}"
                    )
                if np.any(prediction[:WARMUP] != 0.0):
                    raise ValueError(
                        f"exported warmup is not zeroed for {member_id} at group {group}"
                    )
                raw_bytes = memoryview(prediction).cast("B")
                item.prediction_hasher.update(raw_bytes)
                expected_group_hash = _as_sha256(
                    item.group_results[output_index].get("prediction_sha256"),
                    "group.prediction_sha256",
                )
                if hashlib.sha256(raw_bytes).hexdigest() != expected_group_hash:
                    raise ValueError(
                        f"per-group prediction digest differs for {member_id} at group {group}"
                    )
                unit_predictions[:, member_index, :] = prediction
                prediction_views[member_id] = prediction
                graders[member_id].add(targets, prediction, mask)

            for mixture_index, (mixture_name, weights) in enumerate(
                mixtures.items(), start=len(member_ids)
            ):
                prediction = blend_predictions(prediction_views, weights)
                unit_predictions[:, mixture_index, :] = prediction
                graders[mixture_name].add(targets, prediction, mask)

            if mask.any():
                clipped_targets = np.clip(targets[mask], -METRIC_CLIP, METRIC_CLIP).astype(
                    np.float64, copy=False
                )
                for target_index in range(N_OUTPUTS):
                    clipped_predictions = np.clip(
                        unit_predictions[mask, :, target_index], -METRIC_CLIP, METRIC_CLIP
                    ).astype(np.float64, copy=False)
                    prediction_correlations[target_index].add(clipped_predictions)
                    residual_correlations[target_index].add(
                        clipped_targets[:, target_index, None] - clipped_predictions
                    )

            del unit_predictions, prediction_views, targets, mask
            if (output_index + 1) % 16 == 0 or output_index + 1 == len(group_ids):
                _evict_maps(members)
                logger.info(
                    json.dumps(
                        {
                            "completed": output_index + 1,
                            "total": len(group_ids),
                            "wall_seconds": time.monotonic() - started,
                            "utc": datetime.now(UTC).isoformat(),
                        }
                    )
                )

        prediction_file_records: list[dict[str, object]] = []
        for item in members:
            if stable_file_identity(item.result_path.stat()) != stable_file_identity(
                item.result_stat
            ):
                raise ValueError(f"member RESULT changed during grading: {item.result_path}")
            if digest_file(item.result_path) != item.result_sha256:
                raise ValueError(f"member RESULT bytes changed during grading: {item.result_path}")
            if stable_file_identity(item.weight_path.stat()) != stable_file_identity(
                item.weight_stat
            ):
                raise ValueError(f"original weight file changed during grading: {item.weight_path}")
            expected_prediction_hash = _as_sha256(
                item.result.get("prediction_sha256"), "result.prediction_sha256"
            )
            if item.prediction_hasher.hexdigest() != expected_prediction_hash:
                raise ValueError(f"NPY bytes differ from terminal hash: {item.prediction_path}")
            if stable_file_identity(item.prediction_path.stat()) != stable_file_identity(
                item.prediction_stat
            ):
                raise ValueError(f"prediction file changed during grading: {item.prediction_path}")
            prediction_file_records.append(
                {
                    "member_id": item.plan_record["id"],
                    "path": str(item.prediction_path.resolve()),
                    "sha256": expected_prediction_hash,
                    "shape": list(item.predictions.shape),
                    "dtype": str(item.predictions.dtype),
                }
            )

        final_data_stat = args.data.stat()
        final_data_identity = (
            final_data_stat.st_dev,
            final_data_stat.st_ino,
            final_data_stat.st_size,
            final_data_stat.st_mtime_ns,
        )
        if final_data_identity != data_stat_identity:
            raise ValueError("validation data file changed during grading")
        if _grader_source_records() != grader_sources:
            raise ValueError("grader or official metric source changed during grading")

        grades = {name: accumulator.result() for name, accumulator in graders.items()}
        correlations = {
            "population": (
                "unweighted Pearson over need_prediction AND is_scored rows after clipping each "
                "prediction and target to the official [-2, 2] range"
            ),
            "residual": "clipped target minus clipped prediction",
            "constant_columns": "correlation is recorded as zero when variance is zero",
            "unit_order": unit_names,
            "targets": {
                target_name: {
                    "prediction": prediction_correlations[index].correlation().tolist(),
                    "residual": residual_correlations[index].correlation().tolist(),
                    "rows": prediction_correlations[index].count,
                }
                for index, target_name in enumerate(TARGET_COLUMNS)
            },
        }
        full_run = group_ids == tuple(range(EXPECTED_ROW_GROUPS))
        result = {
            "status": "complete" if full_run else "partial_diagnostic",
            "created_utc": datetime.now(UTC).isoformat(),
            "grader_sha256": grader_sources["entrypoint"]["sha256"],
            "grader_sources": grader_sources,
            "git_revision": _git_revision(),
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": plan_sha256,
            "inventory_path": str(args.inventory.resolve()),
            "inventory_sha256": inventory_sha256,
            "data_path": str(args.data.resolve()),
            "data_bytes": data_bytes,
            "data_sha256": data_sha256,
            "data_hash_source": "verified by each terminal exporter and matched to inventory; not rehashed here",
            "row_groups": list(group_ids),
            "sequence_order": {
                "seq_ix_by_row_group": sequence_ids,
                "row_group_seq_ix_sha256": order_hash.hexdigest(),
            },
            "selected_rows": next(iter(grades.values()))["selected_rows"],
            "metric_id": METRIC_ID,
            "metric_clip": METRIC_CLIP,
            "official_mask": "need_prediction AND is_scored",
            "mixture_semantics": "fixed FP32 weighted sum first; official metric clips the final blend",
            "runtime_environment": environment,
            "source_dependencies": source_digests,
            "source_dependency_paths": source_paths,
            "member_sources": source_records,
            "prediction_files": prediction_file_records,
            "target_sha256_float32_group_order": target_hash.hexdigest(),
            "official_mask_sha256_bool_group_order": mask_hash.hexdigest(),
            "mixture_weights": mixtures,
            "grades": grades,
            "correlations": correlations,
            "scope": (
                "supplied-valid FP32 member and predeclared-mixture diagnostics; final fitted or "
                "unknown-overlap assets; no new fitting, mixture optimization, clean-CV, or selection claim"
            ),
        }
        args.out.mkdir(parents=True)
        result_path = args.out / "RESULT.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary_path.replace(result_path)
        logger.info(json.dumps({"status": result["status"], "result": str(result_path)}))
    finally:
        for item in members:
            mapped = getattr(item.predictions, "_mmap", None)
            if mapped is not None:
                mapped.close()


if __name__ == "__main__":
    main()
