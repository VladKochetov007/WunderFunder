#!/usr/bin/env python3
"""Grade a source-bound campaign evaluation export with the official metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wnn.data import SEQUENCE_LENGTH  # noqa: E402
from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split  # noqa: E402

MAX_TARGET_BYTES = 4 * 1024**3
COPY_BYTES = 8 * 1024**2
SEQUENCE_CHUNK = 8
RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def membership_sha256(groups: Sequence[int]) -> str:
    """Hash ordered source row-group membership in the export's canonical format."""
    values = np.asarray(groups, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_export_manifest(
    manifest: Mapping[str, object],
    *,
    run: str,
    epoch: int,
    split_sha256: str,
    row_groups: Sequence[int],
) -> None:
    """Require an evaluation-only export bound to the exact ordered fold membership."""
    expected = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "split_manifest_sha256": split_sha256,
        "campaign_split_manifest_sha256": split_sha256,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": list(map(int, row_groups)),
        "evaluation_membership_sha256": membership_sha256(row_groups),
    }
    if type(manifest.get("epoch")) is not int or type(manifest.get("feature_only")) is not bool:
        raise ValueError("prediction manifest epoch/feature_only have invalid types")
    actual_groups = manifest.get("row_groups")
    if not isinstance(actual_groups, list) or any(
        type(group) is not int for group in actual_groups
    ):
        raise ValueError("prediction manifest row_groups must be integer sequence IDs")
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"prediction manifest {field} does not match the requested endpoint")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _group_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} must be a list of nonnegative row-group integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate row groups")
    return value


def validate_checkpoint(
    root: Path, run: str, epoch: int, split: CampaignSplit
) -> dict[str, object]:
    """Bind checkpoint, saved provenance and fit membership to the loaded split."""
    import torch

    run_dir = root / "artifacts" / "runs" / run
    checkpoint_path = run_dir / f"model_ep{epoch}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("epoch")) is not int
        or checkpoint.get("epoch") != epoch
    ):
        raise ValueError("checkpoint epoch does not match the requested endpoint")
    args, embedded = checkpoint.get("args"), checkpoint.get("provenance")
    run_provenance = _read_json(run_dir / "provenance.json")
    if not isinstance(args, dict) or not isinstance(embedded, dict):
        raise ValueError("checkpoint lacks saved arguments or embedded provenance")
    if args.get("name") != run or args.get("valid_fold") != split.fold:
        raise ValueError("checkpoint run or fold differs from the requested campaign split")
    if args.get("split_manifest_sha256") != split.sha256:
        raise ValueError("checkpoint arguments name a different split manifest")
    if embedded.get("split_manifest_sha256") != split.sha256:
        raise ValueError("embedded checkpoint provenance names a different split")
    if run_provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError("run provenance names a different split manifest")
    if run_provenance.get("args") != args or embedded.get("args") != args:
        raise ValueError("checkpoint and run provenance arguments disagree")
    fit = run_provenance.get("fit_row_groups")
    if not isinstance(fit, dict) or embedded.get("fit_row_groups") != fit:
        raise ValueError("checkpoint and run fit-membership provenance disagree")
    train_groups = _group_list(fit.get("train"), "fit train groups")
    valid_groups = _group_list(fit.get("valid"), "fit valid groups")
    split.assert_fit_groups(train_groups, valid_groups)
    return {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "provenance_sha256": file_sha256(run_dir / "provenance.json"),
        "training_args_sha256": hashlib.sha256(
            json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "fit_membership": {
            "train_group_count": len(train_groups),
            "train_membership_sha256": membership_sha256(sorted(train_groups)),
            "valid_group_count": len(valid_groups),
            "valid_membership_sha256": membership_sha256(sorted(valid_groups)),
        },
    }


def validate_prediction(
    root: Path, run: str, epoch: int, groups: Sequence[int], split: CampaignSplit
) -> tuple[np.ndarray, dict[str, object]]:
    """Check prediction bytes, shape, finiteness and the producer sidecar."""
    path = root / "cache" / f"pred_{run}_valid.npy"
    manifest_path = path.with_suffix(".json")
    manifest = _read_json(manifest_path)
    checkpoint = validate_checkpoint(root, run, epoch, split)
    validate_export_manifest(
        manifest, run=run, epoch=epoch, split_sha256=split.sha256, row_groups=groups
    )
    if manifest.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
        raise ValueError("prediction export references a different checkpoint")
    prediction_sha = file_sha256(path)
    if manifest.get("prediction_sha256") != prediction_sha:
        raise ValueError("prediction export hash differs from its sidecar")
    prediction = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(groups), SEQUENCE_LENGTH, 2)
    if prediction.shape != expected_shape or prediction.dtype != np.float32:
        raise ValueError(f"prediction shape/dtype must be {expected_shape} float32")
    for first in range(0, len(groups), SEQUENCE_CHUNK):
        if not np.isfinite(prediction[first : first + SEQUENCE_CHUNK]).all():
            raise ValueError("prediction contains nonfinite values")
    return prediction, {
        **checkpoint,
        "prediction_sha256": prediction_sha,
        "sidecar_sha256": file_sha256(manifest_path),
    }


@contextmanager
def open_targets_memmapped(path: Path, scratch_dir: Path) -> Iterator[tuple[np.ndarray, ...]]:
    """Extract only y/mask to bounded temporary NPY files and memory-map them."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="priority25-grade-", dir=scratch_dir)
    directory = Path(temporary.name)
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"row_group.npy", "seq_ix.npy", "y.npy", "mask.npy"}
            if not required.issubset(names):
                raise ValueError("valid target cache lacks required index, target or mask arrays")
            uncompressed = sum(archive.getinfo(name).file_size for name in ("y.npy", "mask.npy"))
            if uncompressed > MAX_TARGET_BYTES:
                raise ValueError("target and mask arrays exceed the 4 GiB extraction bound")
            for name in ("y.npy", "mask.npy"):
                with archive.open(name) as source, (directory / name).open("wb") as output:
                    copied = 0
                    while block := source.read(COPY_BYTES):
                        copied += len(block)
                        if copied > archive.getinfo(name).file_size:
                            raise ValueError(f"invalid uncompressed size for {name}")
                        output.write(block)
        with np.load(path, allow_pickle=False) as cache:
            groups = np.asarray(cache["row_group"], dtype=np.int64).copy()
            seq_ix = np.asarray(cache["seq_ix"], dtype=np.int64).copy()
        targets = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(directory / "mask.npy", mmap_mode="r", allow_pickle=False)
        if (
            targets.dtype != np.float32
            or targets.ndim != 3
            or targets.shape[1:] != (SEQUENCE_LENGTH, 2)
        ):
            raise ValueError("target cache y must be float32 with shape (N, 20000, 2)")
        if masks.shape != targets.shape[:2] or masks.dtype.kind not in "bui":
            raise ValueError(
                "target cache mask must align with targets and have a boolean/integer dtype"
            )
        if masks.dtype.kind != "b" and not np.isin(masks, (0, 1)).all():
            raise ValueError("target cache mask contains values outside {0, 1}")
        if groups.shape != (len(targets),) or seq_ix.shape != groups.shape:
            raise ValueError("target cache sequence index does not align with arrays")
        if len(set(groups.tolist())) != len(groups) or np.any(groups < 0):
            raise ValueError("target cache row groups must be unique and nonnegative")
        yield groups, seq_ix, targets, masks
        del targets, masks
    finally:
        temporary.cleanup()


def validate_source_rows(
    root: Path,
    groups: Sequence[int],
    cache_positions: Mapping[int, int],
    seq_ix: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Match target-cache mask/order and build the official scored prediction mask."""
    import pyarrow.parquet as pq

    source = pq.ParquetFile(root / "datasets" / "valid.parquet")
    if max(groups) >= source.num_row_groups:
        raise ValueError("evaluation row group exceeds valid.parquet")
    steps_expected = np.arange(SEQUENCE_LENGTH, dtype=np.int64)
    official_masks = np.empty((len(groups), SEQUENCE_LENGTH), dtype=bool)
    for index, group in enumerate(groups):
        table = source.read_row_group(
            int(group),
            columns=["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"],
            use_threads=False,
        )
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"valid.parquet row group {group} has the wrong sequence length")
        source_seq = table.column("seq_ix").to_numpy().astype(np.int64, copy=False)
        steps = table.column("step_in_seq").to_numpy().astype(np.int64, copy=False)
        need_prediction = table.column("need_prediction").to_numpy().astype(bool, copy=False)
        source_mask = table.column("is_scored").to_numpy().astype(bool, copy=False)
        source_targets = np.stack(
            [table.column(name).to_numpy() for name in ("t0", "t1")], axis=-1
        ).astype(np.float32, copy=False)
        position = cache_positions[int(group)]
        if not np.all(source_seq == source_seq[0]) or int(source_seq[0]) != int(seq_ix[position]):
            raise ValueError(f"target-cache sequence identity differs at row group {group}")
        if not np.array_equal(steps, steps_expected):
            raise ValueError(f"sequence step order differs at row group {group}")
        if not np.array_equal(np.asarray(masks[position], dtype=bool), source_mask):
            raise ValueError(
                f"target-cache scored mask differs from valid.parquet at row group {group}"
            )
        if not np.array_equal(np.asarray(targets[position]), source_targets, equal_nan=True):
            raise ValueError(f"target-cache labels differ from valid.parquet at row group {group}")
        if np.any((source_mask & need_prediction) & (steps < WARMUP)):
            raise ValueError(f"official mask selects warm-up rows at row group {group}")
        official_masks[index] = source_mask & need_prediction & (steps >= WARMUP)
    return official_masks


def grade_endpoint(
    root: Path,
    run: str,
    epoch: int,
    split_path: Path,
    output: Path,
    reference: tuple[str, int] | None = None,
    n_boot: int = 2000,
    scratch_dir: Path | None = None,
) -> dict[str, object]:
    """Grade one exported endpoint, optionally with paired component inference."""
    if not RUN_PATTERN.fullmatch(run) or (
        reference is not None and not RUN_PATTERN.fullmatch(reference[0])
    ):
        raise ValueError(
            "run names may contain only ASCII letters, digits, underscore, dot and hyphen"
        )
    root = root.resolve()
    split_path = split_path if split_path.is_absolute() else root / split_path
    split_path = split_path.resolve()
    split = load_campaign_split(split_path, root)
    groups = list(split.evaluation_groups)
    prediction, candidate_binding = validate_prediction(root, run, epoch, groups, split)
    reference_prediction = None
    reference_binding = None
    if reference is not None:
        reference_prediction, reference_binding = validate_prediction(
            root, reference[0], reference[1], groups, split
        )
    target_path = root / "cache" / "valid_targets.npz"
    scratch = scratch_dir or root / "cache"
    accum = GlobalAccumulator()
    reference_accum = GlobalAccumulator() if reference_prediction is not None else None
    candidate_moments, reference_moments = [], []
    component_of = {
        group: component for source, group, component in split.components if source == "valid"
    }
    components = np.asarray([component_of[group] for group in groups], dtype=np.int64)
    with open_targets_memmapped(target_path, scratch) as (cache_groups, seq_ix, targets, masks):
        cache_positions = {int(group): index for index, group in enumerate(cache_groups)}
        if not set(groups).issubset(cache_positions):
            raise ValueError("valid target cache omits one or more evaluation row groups")
        official_masks = validate_source_rows(root, groups, cache_positions, seq_ix, masks, targets)
        target_positions = np.asarray([cache_positions[group] for group in groups], dtype=np.int64)
        for first in range(0, len(groups), SEQUENCE_CHUNK):
            last = min(first + SEQUENCE_CHUNK, len(groups))
            selected_targets = np.asarray(targets[target_positions[first:last]], dtype=np.float32)
            selected_mask = official_masks[first:last]
            candidate_chunk = np.asarray(prediction[first:last], dtype=np.float32)
            accum.add(selected_targets, candidate_chunk, selected_mask)
            if reference_prediction is not None and reference_accum is not None:
                reference_chunk = np.asarray(reference_prediction[first:last], dtype=np.float32)
                reference_accum.add(selected_targets, reference_chunk, selected_mask)
                cand, ref = paired_component_moments(
                    selected_targets, candidate_chunk, reference_chunk, selected_mask
                )
                candidate_moments.append(cand)
                reference_moments.append(ref)
    result: dict[str, object] = {
        "schema": "priority25.endpoint_grade.v1",
        "classification": "previously exposed fold; screening evidence, not held-out confirmation",
        "metric_id": METRIC_ID,
        "fold": split.fold,
        "n_folds": split.n_folds,
        "split_manifest_sha256": split.sha256,
        "evaluation_row_groups": groups,
        "evaluation_components": int(len(np.unique(components))),
        "target_cache_sha256": file_sha256(target_path),
        "endpoint": {"run": run, "epoch": epoch, **candidate_binding, "score": accum.result()},
        "exposure_note": (
            "Fold exposure is inherited from the campaign; this grade does not imply a pristine holdout."
        ),
    }
    if reference is not None and reference_accum is not None and reference_binding is not None:
        paired = paired_component_bootstrap(
            np.concatenate(candidate_moments),
            np.concatenate(reference_moments),
            components,
            n_boot=n_boot,
            seed=271828,
            centered=True,
        )
        result["reference"] = {
            "run": reference[0],
            "epoch": reference[1],
            **reference_binding,
            "score": reference_accum.result(),
        }
        result["paired_component_bootstrap"] = {
            "resamples": n_boot,
            "seed": 271828,
            "components": int(len(np.unique(components))),
            "method": "paired whole-component bootstrap using centered moments",
            **{key: value for key, value in paired.items() if key not in {"deltas"}},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _endpoint(value: str) -> tuple[str, int]:
    try:
        run, epoch_text = value.rsplit("@", 1)
        epoch = int(epoch_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("reference must be RUN@EPOCH") from error
    if not RUN_PATTERN.fullmatch(run) or epoch < 1:
        raise argparse.ArgumentTypeError("reference must be RUN@EPOCH with a positive epoch")
    return run, epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="campaign artifact/data root")
    parser.add_argument("--run", required=True, help="candidate run directory name")
    parser.add_argument("--split", type=Path, required=True, help="campaign split manifest JSON")
    parser.add_argument("--epoch", type=int, required=True, help="checkpoint epoch to grade")
    parser.add_argument("--output", type=Path, required=True, help="grade JSON output path")
    parser.add_argument("--reference", type=_endpoint, help="optional paired baseline as RUN@EPOCH")
    parser.add_argument("--bootstrap", type=int, default=2000, help="paired component resamples")
    parser.add_argument(
        "--scratch-dir", type=Path, help="disk scratch for bounded target extraction"
    )
    args = parser.parse_args()
    if args.epoch < 1 or args.bootstrap < 2:
        parser.error("epoch must be positive and bootstrap must be at least 2")
    grade_endpoint(
        args.root,
        args.run,
        args.epoch,
        args.split,
        args.output,
        args.reference,
        args.bootstrap,
        args.scratch_dir,
    )


if __name__ == "__main__":
    main()
