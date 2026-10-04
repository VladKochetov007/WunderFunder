"""Prepare inner splits or assemble origin-aligned, component-held-out teacher caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import train_rnn as training

from wnn.data import SequenceFile
from wnn.privileged import (
    inner_manifest,
    validate_crossfit_cache,
    validate_teacher_checkpoint,
)
from wnn.resources import configure_torch
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]


def prepare(outer: CampaignSplit, directory: Path, folds: int, seed: int) -> None:
    """Freeze every inner membership before fitting either equal-capacity teacher arm."""
    directory.mkdir(parents=True, exist_ok=True)
    for fold in range(folds):
        value = inner_manifest(outer, fold, folds, seed)
        path = directory / f"inner{fold}.json"
        content = json.dumps(value, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"refusing to replace a different frozen inner split: {path}")
        path.write_text(content)


def _restore_inputs(checkpoint: dict) -> None:
    args = checkpoint["args"]
    training.RAW_ONLY = bool(args.get("raw_only"))
    training.ONLINE_NORM = bool(args.get("online_norm"))
    training.HEURISTICS = bool(args.get("heuristics"))
    training.RESEARCH_FEATURES = args.get("research_features", "none")
    training.PRIVILEGED_MODE = args["privileged_mode"]
    training.PRIVILEGED_HORIZON = args["privileged_horizon"]
    training.INPUT_PROPENSITY = training.READOUT = None
    training.READOUT_TILTED = False
    if training.HEURISTICS or args.get("loss") != "wmse" or args.get("arch", "gru") != "gru":
        raise ValueError("bounded origin teacher extraction supports non-heuristic GRU wmse only")


@torch.no_grad()
def extract(
    outer: CampaignSplit,
    checkpoints: list[Path],
    name: str,
    device: torch.device,
    batch_size: int,
    chunk_rows: int,
) -> None:
    """Write only excluded inner-component rows, keeping global target-cache row order."""
    if not name or Path(name).name != name or batch_size < 1 or chunk_rows < 1:
        raise ValueError("cache name and inference batch/chunk sizes must be valid")
    teachers, loaded = [], []
    for path in checkpoints:
        path = path.resolve()
        relative = path.relative_to(ROOT.resolve())
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        inner = validate_teacher_checkpoint(checkpoint, outer)
        teachers.append(
            {
                "checkpoint": str(relative),
                "checkpoint_sha256": file_sha256(path),
                "predicted_groups": inner["predict"],
            }
        )
        loaded.append(checkpoint)
    lineage = hashlib.sha256(json.dumps(teachers, sort_keys=True).encode()).hexdigest()
    sources = {
        split: SequenceFile(ROOT / "datasets" / f"{split}.parquet") for split in ("train", "valid")
    }
    output, metadata, positions = {}, {}, {}
    for split, eligible in (("train", outer.train_groups), ("valid", outer.valid_groups)):
        row_path = ROOT / "cache" / f"{split}_targets.npz"
        groups = np.load(row_path)["row_group"].tolist()
        if len(set(groups)) != len(groups) or not set(eligible).issubset(groups):
            raise ValueError("target row map is duplicated or does not cover the fit pool")
        sidecar = {
            "run": name,
            "split": split,
            "epoch": None,
            "crossfit_format": "wnn.origin-teacher-crossfit.v1",
            "split_manifest_sha256": outer.sha256,
            "checkpoint_sha256": lineage,
            "row_map_sha256": file_sha256(row_path),
            "row_groups": groups,
            "eligible_row_groups": sorted(eligible),
            "teachers": teachers,
            "origin_alignment": "prediction[t] estimates label[t]; no label or weight shift",
            "outside_fit_pool": "NaN; not consumable as teacher targets",
        }
        validate_crossfit_cache(sidecar, outer, ROOT, split)
        final = ROOT / "cache" / f"pred_{name}_{split}.npy"
        if final.exists() or final.with_suffix(".json").exists():
            raise ValueError(f"refusing to overwrite a teacher cache: {final}")
        temporary = final.with_name(final.name + ".partial")
        output[split] = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), training.SEQUENCE_LENGTH, 2),
        )
        output[split][:] = np.nan
        positions[split] = {group: index for index, group in enumerate(groups)}
        metadata[split] = (final, temporary, sidecar)
    for checkpoint, teacher in zip(loaded, teachers, strict=True):
        _restore_inputs(checkpoint)
        model = training.make_model(
            checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0]
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for split in ("train", "valid"):
            groups = teacher["predicted_groups"][split]
            for start in range(0, len(groups), batch_size):
                batch = groups[start : start + batch_size]
                raw = np.stack(
                    [
                        sources[split].read(group, targets=False, mask=False).features
                        for group in batch
                    ]
                )
                predicted = training.predict(model, torch.from_numpy(raw), chunk_rows, device)
                if not np.isfinite(predicted).all():
                    raise ValueError("teacher generated nonfinite held-out predictions")
                output[split][[positions[split][group] for group in batch]] = predicted
        del model
    for split, values in output.items():
        final, temporary, sidecar = metadata[split]
        for group in sidecar["eligible_row_groups"]:
            if not np.isfinite(values[positions[split][group]]).all():
                raise ValueError("an eligible crossfit row was not predicted")
        values.flush()
        temporary.replace(final)
        sidecar["prediction_sha256"] = file_sha256(final)
        final.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser("prepare")
    plan.add_argument("--outer-split", type=Path, required=True)
    plan.add_argument("--directory", type=Path, required=True)
    plan.add_argument("--inner-folds", type=int, default=2)
    plan.add_argument("--seed", type=int, default=98)
    export = subcommands.add_parser("extract")
    export.add_argument("--outer-split", type=Path, required=True)
    export.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    export.add_argument("--batch", type=int, default=1)
    export.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args()
    outer = load_campaign_split(args.outer_split, ROOT)
    if args.command == "prepare":
        prepare(outer, args.directory, args.inner_folds, args.seed)
    else:
        configure_torch()
        extract(
            outer, args.checkpoints, args.name, torch.device(args.device), args.batch, args.chunk
        )


if __name__ == "__main__":
    main()
