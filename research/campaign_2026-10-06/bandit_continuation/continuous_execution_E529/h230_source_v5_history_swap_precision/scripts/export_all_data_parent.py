"""Export all supplied parent predictions in FP32 without reading labels or masks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
sys.path.insert(0, str(SOURCE_ROOT / "src"))
import train_rnn  # noqa: E402

from wnn.data import SEQUENCE_LENGTH, SequenceFile  # noqa: E402
from wnn.distillation_cache import row_groups_sha256  # noqa: E402
from wnn.sentinel import SentinelPreprocessor  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

PARENT_RUN = "bd26_h230_all_data_parent_b16_ep3_retry3_v1"
PARENT_EPOCH = 3
PARENT_CHECKPOINT_SHA256 = "478773bd3d0754d261f8f3c9d46a135d56c3596c269d2a856b760caef05b93c7"
PREDICTION_WIDTH = 2
EXPECTED_COUNTS = {"train": 10606, "valid": 1873}


class FeatureOnlyGroups(Dataset[torch.Tensor]):
    """Read feature tensors for an explicit target-index group order only."""

    def __init__(self, parquet_path: Path, groups: list[int]) -> None:
        self.path = parquet_path
        self.groups = groups
        self._file: SequenceFile | None = None

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._file is None:
            self._file = SequenceFile(self.path)
        sequence = self._file.read(self.groups[index], targets=False, mask=False)
        return torch.from_numpy(sequence.features)

    def batch_tensor_bytes(self, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return batch_size * SEQUENCE_LENGTH * train_rnn.N_FEATURES * np.dtype(np.float32).itemsize


def _target_groups(project_root: Path, split: str) -> list[int]:
    with np.load(project_root / "cache" / f"{split}_targets.npz", allow_pickle=False) as index:
        values = np.asarray(index["row_group"], dtype=np.int64)
    groups = list(map(int, values))
    if values.ndim != 1 or len(groups) != len(set(groups)):
        raise ValueError(f"{split} target index must contain unique row groups")
    if len(groups) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split} target index count differs from the frozen all-data pool")
    return groups


def _configure_checkpoint(
    checkpoint: dict[str, Any], model: torch.nn.Module, device: torch.device
) -> None:
    args = checkpoint["args"]
    train_rnn.PRIVILEGED_MODE = str(args.get("privileged_mode", "none"))
    train_rnn.PRIVILEGED_HORIZON = int(args.get("privileged_horizon", 8))
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing")
        else None
    )
    train_rnn.RAW_ONLY = bool(args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(args.get("online_norm"))
    train_rnn.HEURISTICS = bool(args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = args.get("research_features", "none")
    input_propensity = args.get("prop_input")
    if input_propensity:
        from wnn.propensity import load_propensity

        train_rnn.INPUT_PROPENSITY = load_propensity(
            Path(os.environ["WNN_PROJECT_ROOT"])
            / "artifacts"
            / "runs"
            / input_propensity
            / "model.pt",
            device,
        )
    else:
        train_rnn.INPUT_PROPENSITY = None
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [tensor.to(device) for tensor in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = checkpoint.get("readout_tilted", False)
    else:
        train_rnn.READOUT = None
        train_rnn.READOUT_TILTED = False


def _check_parent(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    train_args = checkpoint.get("args", {})
    if (
        args.run != PARENT_RUN
        or checkpoint.get("epoch") != PARENT_EPOCH
        or train_args.get("name") != PARENT_RUN
        or train_args.get("arch") != "gru"
        or train_args.get("cell") != "gru"
        or train_args.get("hidden") != 256
        or train_args.get("layers") != 2
        or train_args.get("seed") != 3
        or train_args.get("train_source") != "train+valid"
        or train_args.get("valid_fold") != -1
        or train_args.get("include_heldout_valid") is not True
        or train_args.get("no_eval_during_fit") is not True
        or train_args.get("checkpoint_policy") != "fixed"
        or train_args.get("checkpoint_epoch") != PARENT_EPOCH
        or train_args.get("epochs") != 4
        or train_args.get("precision") != "fp16"
        or train_args.get("raw_only") is not False
        or train_args.get("online_norm") is not False
        or train_args.get("heuristics") is not False
        or train_args.get("research_features") != "none"
        or train_args.get("prop_input") not in (None, "")
        or train_args.get("privileged_mode") != "none"
        or train_args.get("qat") != "none"
        or train_args.get("recurrent_rank") != 0
        or checkpoint.get("n_in") != 211
    ):
        raise ValueError("checkpoint is not the frozen all-data H230 epoch-3 parent")


def _first_n_seconds(start: float, count: int, target: int) -> float | None:
    return time.monotonic() - start if count >= target else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "valid"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run", default=PARENT_RUN)
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="optional first-pass export for deterministic repeat comparison",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=4000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 16 or args.chunk_rows != 4000:
        parser.error("use batch size 1..16 and the frozen 4000-row inference chunk")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.output_name):
        parser.error("output name must be a plain 1..80 character filename component")
    project_root = args.project_root.resolve(strict=True)
    os.environ["WNN_PROJECT_ROOT"] = str(project_root)
    train_rnn.ROOT = project_root
    checkpoint_path = project_root / "artifacts" / "runs" / args.run / f"model_ep{PARENT_EPOCH}.pt"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint differs from the frozen all-data epoch-3 artifact")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _check_parent(checkpoint, args)
    if checkpoint["model"]["head.2.weight"].shape[0] != 23:
        raise ValueError("H230 parent checkpoint must have 23 ordinary output rows")
    groups = _target_groups(project_root, args.split)
    fit_groups = checkpoint.get("provenance", {}).get("fit_row_groups", {})
    fitted = list(map(int, fit_groups.get(args.split, [])))
    if len(fitted) != len(set(fitted)) or set(fitted) != set(groups):
        raise ValueError(
            "parent checkpoint fit membership does not include the full supplied split"
        )
    started = time.monotonic()
    train_rnn.configure_torch()
    device = torch.device("cuda")
    output_width = int(checkpoint["model"]["head.2.weight"].shape[0])
    model = train_rnn.make_model(checkpoint["args"], checkpoint["n_in"], output_width).to(device)
    model.load_state_dict(checkpoint["model"])
    _configure_checkpoint(checkpoint, model, device)
    train_rnn.configure_factorized(model, int(checkpoint["args"].get("recurrent_rank", 0)))
    train_rnn.configure_qat(model, str(checkpoint["args"].get("qat", "none")))
    model._inference_fp32 = True
    model.eval()
    setup_seconds = time.monotonic() - started
    dataset = FeatureOnlyGroups(project_root / "datasets" / f"{args.split}.parquet", groups)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    final = project_root / "cache" / f"pred_{args.output_name}_{args.split}.npy"
    sidecar = final.with_suffix(".json")
    temporary = final.with_name(final.name + ".partial")
    if final.exists() or sidecar.exists() or temporary.exists():
        raise FileExistsError(f"refusing to replace existing prediction output: {final}")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), SEQUENCE_LENGTH, PREDICTION_WIDTH),
    )
    write_start = time.monotonic()
    first_100_seconds = None
    written = 0
    for batch in loader:
        prediction = train_rnn.predict(model, batch, args.chunk_rows, device)
        if prediction.ndim != 3 or prediction.shape[-1] < PREDICTION_WIDTH:
            raise ValueError("parent model prediction does not contain two primary outputs")
        values = np.asarray(prediction[..., :PREDICTION_WIDTH], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("parent FP32 export produced nonfinite values")
        output[written : written + len(values)] = values
        written += len(values)
        if first_100_seconds is None:
            first_100_seconds = _first_n_seconds(write_start, written, 100)
    output.flush()
    del output
    if written != len(groups):
        raise RuntimeError(f"wrote {written} of {len(groups)} {args.split} groups")
    temporary.replace(final)
    if args.compare_to is not None:
        reference = np.load(args.compare_to.resolve(strict=True), mmap_mode="r", allow_pickle=False)
        actual = np.load(final, mmap_mode="r", allow_pickle=False)
        if reference.shape != actual.shape or reference.dtype != actual.dtype:
            raise ValueError("determinism comparison output schema differs")
        max_abs_diff = 0.0
        for start in range(0, len(actual), 16):
            diff = np.abs(actual[start : start + 16] - reference[start : start + 16])
            if not np.isfinite(diff).all():
                raise ValueError("determinism comparison found nonfinite differences")
            max_abs_diff = max(max_abs_diff, float(diff.max(initial=0.0)))
        if max_abs_diff > 1e-5:
            raise ValueError(f"repeat prediction differs by {max_abs_diff} > 1e-5")
    else:
        max_abs_diff = None
    metadata = {
        "run": args.run,
        "split": args.split,
        "epoch": PARENT_EPOCH,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": checkpoint["args"].get("split_manifest_sha256"),
        "inference_precision": "fp32",
        "row_groups": groups,
        "row_groups_sha256": row_groups_sha256(groups),
        "dtype": "float32",
        "shape": [len(groups), SEQUENCE_LENGTH, PREDICTION_WIDTH],
        "features_only": True,
        "sequence_state_reset": (
            "predict() starts with state=None for each independent batch; state is carried only "
            "between 4000-row chunks within each sequence"
        ),
        "source_trainer_sha256": file_sha256(SOURCE_ROOT / "scripts" / "train_rnn.py"),
        "source_exporter_sha256": file_sha256(Path(__file__).resolve()),
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)),
        "output_name": args.output_name,
        "timing": {
            "checkpoint_and_model_setup_seconds": setup_seconds,
            "first_100_sequences_wall_seconds": first_100_seconds,
            "total_export_wall_seconds": time.monotonic() - write_start,
            "sequences_per_second": written / max(time.monotonic() - write_start, 1e-9),
            "sequence_rows_per_second": written
            * SEQUENCE_LENGTH
            / max(time.monotonic() - write_start, 1e-9),
        },
        "repeat_comparison_max_abs_diff": max_abs_diff,
    }
    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
    sidecar_partial.write_text(json.dumps(metadata, indent=2) + "\n")
    sidecar_partial.replace(sidecar)
    print(
        json.dumps(
            {
                "prediction_path": str(final),
                "sidecar_path": str(sidecar),
                "prediction_sha256": metadata["prediction_sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "shape": metadata["shape"],
                "sequences_per_second": metadata["timing"]["sequences_per_second"],
                "repeat_max_abs_diff": max_abs_diff,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
