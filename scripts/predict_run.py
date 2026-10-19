"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from scripts import train_rnn
    from scripts.train_rnn import Sequences, make_model, predict
else:
    import train_rnn
    from train_rnn import Sequences, make_model, predict

from wnn import prediction_profile
from wnn.data import N_FEATURES, SEQUENCE_LENGTH, SequenceFile
from wnn.prediction_profile import (
    first_profile_groups,
    run_prediction_profile,
    write_json_create_only,
)
from wnn.propensity import load_propensity
from wnn.research_features import (
    MOMENT_FEATURE_MODES,
    research_feature_control_width,
    validate_causal_moment_prior,
)
from wnn.resources import configure_strict_fp32, configure_torch, dataloader_budget
from wnn.run_identity import normalization_identity
from wnn.sentinel import SentinelPreprocessor
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.input_representation import (
    FittedInputRepresentation,
    deserialize_input_representation,
    input_representation_sha256,
    representation_input_width,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", SOURCE_ROOT)).resolve()


def family_output_slice(checkpoint_args: Mapping[str, object], n_outputs: int) -> slice:
    """Locate family rows before optional logvariance rows in an output tensor."""
    if checkpoint_args.get("distill_family_weight", 0.0) <= 0:
        raise ValueError("family output export requires a checkpoint with a family head")
    trailing_variance = 2 if checkpoint_args.get("mean_variance_head", 0) else 0
    stop = n_outputs - trailing_variance
    start = stop - 8
    if start < 2:
        raise ValueError("checkpoint output width cannot contain its declared family head")
    return slice(start, stop)


def restore_input_representation(
    checkpoint: dict[str, object], device: torch.device
) -> tuple[str | None, FittedInputRepresentation | None]:
    """Restore the exact transform required by a representation-enabled checkpoint."""
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        raise ValueError("checkpoint training arguments are missing")
    raw_mode = checkpoint_args.get("representation_mode")
    if raw_mode is not None and not isinstance(raw_mode, str):
        raise ValueError("checkpoint representation mode is malformed")
    mode = None if raw_mode in (None, "none") else raw_mode
    if representation_input_width(mode):
        if checkpoint.get("n_in") != 467:
            raise ValueError("representation checkpoint must use the fixed 467-wide input layout")
        payload = checkpoint.get("input_representation")
        if payload is None:
            raise ValueError("representation checkpoint lacks its fitted input transform")
        return mode, deserialize_input_representation(payload, device)
    if checkpoint.get("input_representation") is not None:
        raise ValueError("legacy checkpoint unexpectedly contains an input representation")
    return None, None


def campaign_prediction_groups(
    manifest_path: Path,
    checkpoint: dict,
    split: str,
    *,
    evaluation_only: bool = False,
    full_valid: bool = False,
) -> np.ndarray:
    """Bind feature-only campaign exports to checkpoint ancestry and row order."""
    if split not in ("train", "valid"):
        raise ValueError("campaign exports require train or valid source")
    if evaluation_only and full_valid:
        raise ValueError("evaluation-only and full-valid selectors are mutually exclusive")
    if (evaluation_only or full_valid) and split != "valid":
        raise ValueError("evaluation/full-valid selectors require the valid source")
    manifest = load_campaign_split(manifest_path, ROOT)
    if manifest.sha256 != checkpoint.get("args", {}).get("split_manifest_sha256"):
        raise ValueError("campaign manifest differs from the fitted checkpoint")
    if full_valid:
        with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as targets:
            groups = np.asarray(targets["row_group"], dtype=np.int64)
    elif evaluation_only:
        groups = np.asarray(manifest.evaluation_groups, dtype=np.int64)
    else:
        groups = np.asarray(
            manifest.train_groups if split == "train" else manifest.valid_groups,
            dtype=np.int64,
        )
    if groups.ndim != 1 or groups.size == 0 or len(set(groups.tolist())) != groups.size:
        raise ValueError("campaign export row groups must be nonempty and unique")
    return groups


class FeatureOnlySequences(Dataset[torch.Tensor]):
    """Read only features for an explicit, frozen sequence membership."""

    def __init__(self, split: str, groups: list[int]) -> None:
        self.path = ROOT / "datasets" / f"{split}.parquet"
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
        """Estimate the feature tensors retained by one prediction batch."""
        if batch_size <= 0:
            raise ValueError("prediction batch size must be positive")
        return batch_size * SEQUENCE_LENGTH * N_FEATURES * np.dtype(np.float32).itemsize


def load_e540_evaluation_groups(path: Path, checkpoint: dict[str, object]) -> list[int]:
    """Validate and return the exact evaluation membership used by an E540 run."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("experiment_id") != "E540"
    ):
        raise ValueError("evaluation membership must be an E540 fit-subset manifest")
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_subset_hash = checkpoint_args.get("fit_subset_manifest_sha256")
    if not checkpoint_subset_hash or file_sha256(path) != checkpoint_subset_hash:
        raise ValueError("evaluation membership differs from the E540 training manifest")
    checkpoint_split_hash = checkpoint.get("provenance", {}).get("split_manifest_sha256")
    if not checkpoint_split_hash or manifest.get("split_manifest_sha256") != checkpoint_split_hash:
        raise ValueError("evaluation membership belongs to a different campaign split")
    groups = manifest.get("evaluation_groups")
    if (
        not isinstance(groups, list)
        or not groups
        or any(type(group) is not int or group < 0 for group in groups)
        or groups != sorted(set(groups))
        or type(manifest.get("evaluation_group_count")) is not int
        or manifest.get("evaluation_group_count") != len(groups)
    ):
        raise ValueError("E540 evaluation groups must be unique sorted nonnegative integers")
    digest = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    if digest != manifest.get("evaluation_membership_sha256"):
        raise ValueError("E540 evaluation membership digest is invalid")
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--campaign-split-manifest", type=Path)
    parser.add_argument("--campaign-evaluation-only", action="store_true")
    parser.add_argument("--campaign-full-valid-feature-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--fit-subset-manifest",
        type=Path,
        default=None,
        help="predict only the exact E540 evaluation groups with feature-only reads",
    )
    parser.add_argument(
        "--inference-precision",
        choices=["checkpoint", "fp32"],
        default="checkpoint",
        help="force common fp32 evaluation for paired fp16/fp32 training controls",
    )
    parser.add_argument(
        "--dump-family",
        action="store_true",
        help="also save [main 2 outputs, last 8 family-head outputs] to cache/allout_<name>.npy (research probe)",
    )
    parser.add_argument("--gate-mode", choices=["soft", "hard", "confident"], default="soft")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--create-only", action="store_true")
    return parser


def validate_profile_checkpoint(checkpoint: dict[str, object]) -> None:
    """Require the declared raw-112, model-211 two-layer GRU inference geometry."""
    checkpoint_args = checkpoint.get("args")
    model_state = checkpoint.get("model")
    if not isinstance(checkpoint_args, dict) or not isinstance(model_state, dict):
        raise ValueError("profile requires a complete GRU checkpoint")
    if (
        checkpoint.get("n_in") != 211
        or N_FEATURES != 112
        or checkpoint_args.get("arch") != "gru"
        or checkpoint_args.get("hidden") != 384
        or checkpoint_args.get("layers") != 2
        or checkpoint_args.get("privileged_mode", "none") != "none"
        or checkpoint.get("readout") is not None
        or checkpoint.get("quantization") is not None
        or checkpoint.get("factorization") is not None
    ):
        raise ValueError("profile checkpoint differs from raw-112/model-211 H384x2 GRU inference")
    output = model_state.get("head.2.weight")
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (23, 384):
        raise ValueError("profile checkpoint must preserve its 23-output auxiliary teacher head")


def source_hashes() -> dict[str, str]:
    """Hash the predictor and all directly reused inference-path source modules."""
    paths = {
        Path(__file__).resolve(),
        Path(train_rnn.__file__).resolve(),
        Path(predict.__code__.co_filename).resolve(),
        Path(make_model.__code__.co_filename).resolve(),
        Path(normalization_identity.__code__.co_filename).resolve(),
        Path(prediction_profile.__file__).resolve(),
    }
    paths.update(
        (SOURCE_ROOT / relative).resolve()
        for relative in (
            "src/wnn/data.py",
            "src/wnn/nn.py",
            "src/wnn/resources.py",
            "src/wnn/sentinel.py",
            "src/wnn/splits.py",
            "src/wnn/research_features.py",
            "src/wnn/training/compatibility.py",
            "src/wnn/training/environment.py",
            "src/wnn/training/evaluation.py",
            "src/wnn/training/forward.py",
            "src/wnn/training/input_representation.py",
            "src/wnn/training/inputs.py",
            "src/wnn/training/models.py",
            "src/wnn/numeric_ple.py",
        )
    )
    return {path.relative_to(SOURCE_ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}


def reserve_create_only_prediction_paths(
    output_dir: Path, output_stem: str, split: str
) -> tuple[Path, Path, Path]:
    """Reserve the partial path only when final array and sidecar are both absent."""
    if Path(output_stem).name != output_stem:
        raise ValueError("create-only output name must be a single filename component")
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"pred_{output_stem}_{split}.npy"
    partial = final.with_name(final.name + ".partial")
    sidecar = final.with_suffix(".json")
    existing = [path for path in (final, partial, sidecar) if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"create-only prediction outputs already exist: {existing}")
    with partial.open("xb"):
        pass
    return final, partial, sidecar


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    name, split = args.name, args.split
    if args.loader_workers is not None and args.loader_workers < 0:
        parser.error("--loader-workers must be nonnegative")
    if args.profile_only:
        if (
            args.profile_output is None
            or args.campaign_split_manifest is None
            or split != "train"
            or args.campaign_evaluation_only
            or args.campaign_full_valid_feature_only
            or args.fit_subset_manifest is not None
            or args.inference_precision != "fp32"
            or args.device != "cuda"
            or args.loader_workers != 0
            or args.output_dir is not None
            or args.create_only
            or args.dump_family
        ):
            parser.error(
                "profile-only requires train campaign-fit membership, FP32 CUDA, "
                "--loader-workers 0, and --profile-output"
            )
    elif args.profile_output is not None:
        parser.error("--profile-output requires --profile-only")
    if args.output_dir is not None and not args.create_only:
        parser.error("--output-dir requires --create-only to protect paired cache outputs")
    if args.create_only and (args.output_dir is None or args.dump_family):
        parser.error("--create-only requires --output-dir and excludes --dump-family")
    if args.create_only and (
        args.campaign_split_manifest is None
        or split not in ("train", "valid")
        or args.campaign_evaluation_only
        or args.campaign_full_valid_feature_only
        or args.fit_subset_manifest is not None
    ):
        parser.error("create-only fit export requires train/valid campaign membership")
    if (args.campaign_evaluation_only or args.campaign_full_valid_feature_only) and (
        args.campaign_split_manifest is None
    ):
        parser.error("campaign row selectors require --campaign-split-manifest")
    if args.fit_subset_manifest is not None and args.campaign_split_manifest is not None:
        parser.error("E540 and campaign selectors are mutually exclusive")
    checkpoint_name = f"model_ep{args.epoch}.pt" if args.epoch is not None else "model.pt"
    checkpoint_path = ROOT / "artifacts" / "runs" / name / checkpoint_name
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ck["args"].get("privileged_mode", "none") != "none":
        raise ValueError(
            "privileged teachers require component-crossfit extraction, not causal prediction export"
        )
    family_columns = None
    if args.dump_family:
        try:
            family_columns = family_output_slice(ck["args"], ck["model"]["head.2.weight"].shape[0])
        except ValueError as exc:
            parser.error(str(exc))
    feature_only = args.fit_subset_manifest is not None or args.campaign_split_manifest is not None
    if args.campaign_split_manifest is not None:
        groups = campaign_prediction_groups(
            args.campaign_split_manifest,
            ck,
            split,
            evaluation_only=args.campaign_evaluation_only,
            full_valid=args.campaign_full_valid_feature_only,
        )
    elif args.fit_subset_manifest is not None:
        if (
            split != "valid"
            or ck["args"].get("numeric_map", "none") == "none"
            or ck["args"].get("fit_subset_manifest_sha256") is None
        ):
            parser.error("E540 fit-subset export requires a numeric-map checkpoint and valid split")
        groups = np.asarray(
            load_e540_evaluation_groups(args.fit_subset_manifest, ck), dtype=np.int64
        )
    else:
        with np.load(ROOT / "cache" / f"{split}_targets.npz", allow_pickle=False) as targets:
            groups = targets["row_group"]
    if args.profile_only:
        groups = np.asarray(first_profile_groups(groups.tolist()), dtype=np.int64)
    quote_cache = ck.get("provenance", {}).get("quote_cache")
    if ck["args"].get("quote_cache_root") and quote_cache is None:
        raise ValueError("checkpoint lacks quote cache provenance")
    if args.profile_only:
        validate_profile_checkpoint(ck)
        if not torch.cuda.is_available():
            raise RuntimeError("profile-only requires an available CUDA device")
    configure_torch()
    device = torch.device(args.device)
    strict_fp32 = (
        args.profile_only
        or args.inference_precision == "fp32"
        or ck["args"].get("precision") == "fp32"
        or ck.get("quantization") is not None
        or ck.get("factorization") is not None
    )
    if strict_fp32:
        configure_strict_fp32()
    if args.profile_only:
        torch.set_num_threads(1)
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(ck["sentinel_preprocessing"])
        if ck.get("sentinel_preprocessing")
        else None
    )
    if strict_fp32:
        model._inference_fp32 = True
    if ck["args"].get("arch") == "gated":
        model.gate_mode = args.gate_mode
    elif args.gate_mode != "soft":
        parser.error("--gate-mode requires a gated checkpoint")
    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    train_rnn.HEURISTICS = bool(ck["args"].get("heuristics"))
    train_rnn.INPUT_LEAD = int(ck["args"].get("input_lead", 0))
    if ck["args"].get("snap_codebook"):
        import numpy as _np

        _book = _np.load(ck["args"]["snap_codebook"])
        train_rnn.SNAP_CODEBOOK = {
            int(k[1:]): (torch.tensor((v[1:] + v[:-1]) / 2), torch.tensor(v))
            for k, v in _book.items()
        }
    train_rnn.RESEARCH_FEATURES = ck["args"].get("research_features", "none")
    train_rnn.RESEARCH_FEATURE_CONTROL = ck["args"].get("research_feature_control", "none")
    research_feature_control_width(train_rnn.RESEARCH_FEATURES, train_rnn.RESEARCH_FEATURE_CONTROL)
    if train_rnn.RESEARCH_FEATURES in MOMENT_FEATURE_MODES:
        prior = ck.get("research_feature_prior")
        if not isinstance(prior, dict):
            raise ValueError("causal moment checkpoint lacks its fitting-only feature prior")
        train_rnn.RESEARCH_FEATURE_PRIOR = validate_causal_moment_prior(prior)
    else:
        train_rnn.RESEARCH_FEATURE_PRIOR = None
    representation_mode, fitted_representation = restore_input_representation(ck, device)
    train_rnn.INPUT_REPRESENTATION = fitted_representation
    train_rnn.REPRESENTATION_MODE = representation_mode
    input_propensity = ck["args"].get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts" / "runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    dataset = (
        FeatureOnlySequences(split, groups.tolist()) if feature_only else Sequences(split, groups)
    )
    workers, prefetch = (
        (args.loader_workers, None)
        if args.loader_workers is not None
        else dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
    )
    if args.profile_only:
        run_prediction_profile(
            model,
            loader,
            predict,
            device,
            groups.tolist(),
            args.profile_output,
            checkpoint_sha256=file_sha256(checkpoint_path),
            source_sha256=source_hashes(),
            split_manifest_sha256=ck["args"].get("split_manifest_sha256"),
            model_input_width=int(ck["n_in"]),
            sequence_length=SEQUENCE_LENGTH,
        )
        return
    output_stem = args.output_name or name
    output_dir = args.output_dir.resolve() if args.output_dir is not None else ROOT / "cache"
    if args.create_only:
        final, tmp, sidecar_path = reserve_create_only_prediction_paths(
            output_dir, output_stem, split
        )
    else:
        final = output_dir / f"pred_{output_stem}_{split}.npy"
        tmp = final.with_name(final.name + ".partial")
        sidecar_path = final.with_suffix(".json")
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(groups), SEQUENCE_LENGTH, 2)
    )
    i = 0
    fam = None
    if args.dump_family:
        fam = np.lib.format.open_memmap(
            ROOT / "cache" / f"allout_{args.output_name or name}_{split}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), SEQUENCE_LENGTH, 10),
        )
    for batch in loader:
        x = batch if feature_only else batch[0]
        if fam is None:
            p = predict(model, x, 4000, device)
        else:  # research probe: keep the main targets and the declared family head
            model.eval()
            with torch.no_grad():
                inputs = train_rnn.build_inputs(x.to(device, non_blocking=True))
            state, parts = None, []
            for s0 in range(0, inputs.shape[1], 4000):
                with (
                    torch.no_grad(),
                    torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
                ):
                    o, state = train_rnn.forward_chunk(model, inputs, slice(s0, s0 + 4000), state)
                parts.append(o.float())
            full = torch.cat(parts, dim=1).detach().cpu().numpy()
            p = full[..., :2]
            fam[i : i + len(p)] = np.concatenate(
                [full[..., :2], full[..., family_columns]], axis=-1
            )
        out[i : i + len(p)] = p[..., :2]
        i += len(p)
        if args.create_only:
            print(
                json.dumps(
                    {
                        "event": "prediction_progress",
                        "completed_sequences": i,
                        "total_sequences": len(groups),
                    }
                ),
                flush=True,
            )
    out.flush()
    del out
    if i != len(groups):
        raise RuntimeError(f"wrote {i} of {len(groups)} sequences")
    if args.create_only:
        os.link(tmp, final)
        tmp.unlink()
    else:
        stale = ROOT / "cache" / f"seqsums_{args.output_name or name}.npz"
        if stale.exists():
            stale.unlink()
        tmp.replace(final)
    metadata = {
        "run": name,
        "split": split,
        "epoch": ck.get("epoch"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": ck["args"].get("split_manifest_sha256"),
        "quote_cache": quote_cache,
        "gate_mode": args.gate_mode,
        "inference_precision": "fp32" if getattr(model, "_inference_fp32", False) else "fp16",
        "quantization_preset": ck.get("quantization", {}).get("preset"),
        "recurrent_rank": ck.get("factorization", {}).get("rank"),
        "sentinel_preprocessing": ck.get("sentinel_preprocessing"),
        "numeric_preprocessing": ck.get("provenance", {}).get("numeric_preprocessing"),
        "representation_mode": representation_mode,
        "input_representation_state_sha256": (
            input_representation_sha256(train_rnn.INPUT_REPRESENTATION)
            if train_rnn.INPUT_REPRESENTATION is not None
            else None
        ),
        "row_groups": groups.tolist(),
        "evaluation_membership_sha256": (
            hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
            if feature_only
            else None
        ),
        "fit_subset_manifest_sha256": (
            file_sha256(args.fit_subset_manifest) if args.fit_subset_manifest is not None else None
        ),
        "campaign_export_scope": (
            "full-valid-feature-only"
            if args.campaign_full_valid_feature_only
            else "evaluation"
            if args.campaign_evaluation_only
            else "fit"
            if args.campaign_split_manifest is not None
            else None
        ),
        "campaign_split_manifest_sha256": (
            file_sha256(args.campaign_split_manifest)
            if args.campaign_split_manifest is not None
            else None
        ),
        "feature_only": feature_only,
    }
    if args.output_dir is not None:
        metadata["normalization_identity"] = normalization_identity(model)
        metadata["source_sha256"] = source_hashes()
        write_json_create_only(sidecar_path, metadata)
    else:
        sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
