"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, SequenceFile
from wnn.parsed_sequence_cache_reader import ParsedCacheError, ParsedSequenceCache
from wnn.propensity import load_propensity
from wnn.resources import configure_torch, dataloader_budget
from wnn.sentinel import SentinelPreprocessor
from wnn.splits import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class FeatureOnlySequences(Dataset[torch.Tensor]):
    """Read only features for an explicit, frozen sequence membership."""

    def __init__(
        self,
        split: str,
        groups: list[int],
        parsed_cache: ParsedSequenceCache | None = None,
        parsed_cache_role: str | None = None,
    ) -> None:
        self.path = ROOT / "datasets" / f"{split}.parquet"
        self.split = split
        self.groups = groups
        self._file: SequenceFile | None = None
        self.parsed_cache = parsed_cache
        self.parsed_cache_role = parsed_cache_role

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.parsed_cache is not None:
            features = self.parsed_cache.read_features(
                self.split,
                self.groups[index],
                self.parsed_cache_role,
            )
            return torch.from_numpy(features)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--fit-subset-manifest",
        type=Path,
        default=None,
        help="predict only the exact E540 evaluation groups with feature-only reads",
    )
    parser.add_argument("--parsed-cache-manifest", type=Path)
    parser.add_argument("--parsed-cache-validation-receipt", type=Path)
    parser.add_argument("--expected-parsed-cache-manifest-sha256", default="")
    parser.add_argument("--expected-parsed-cache-validation-sha256", default="")
    parser.add_argument("--expected-parsed-cache-key", default="")
    parser.add_argument(
        "--inference-precision",
        choices=["checkpoint", "fp32"],
        default="checkpoint",
        help="force common fp32 evaluation for paired fp16/fp32 training controls",
    )
    parser.add_argument("--gate-mode", choices=["soft", "hard", "confident"], default="soft")
    args = parser.parse_args()
    name, split = args.name, args.split
    checkpoint_name = f"model_ep{args.epoch}.pt" if args.epoch is not None else "model.pt"
    checkpoint_path = ROOT / "artifacts" / "runs" / name / checkpoint_name
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ck["args"].get("privileged_mode", "none") != "none":
        raise ValueError(
            "privileged teachers require component-crossfit extraction, not causal prediction export"
        )
    parsed_cache = None
    cache_arguments = (
        args.parsed_cache_manifest,
        args.parsed_cache_validation_receipt,
        args.expected_parsed_cache_manifest_sha256,
        args.expected_parsed_cache_validation_sha256,
        args.expected_parsed_cache_key,
    )
    if any(cache_arguments):
        if not all(cache_arguments) or args.fit_subset_manifest is not None or split != "valid":
            parser.error(
                "parsed-cache export requires its full pins and valid feature-only inference"
            )
        provenance = ck.get("provenance", {})
        try:
            parsed_cache = ParsedSequenceCache.open(
                args.parsed_cache_manifest,
                expected_manifest_sha256=args.expected_parsed_cache_manifest_sha256,
                validation_receipt_path=args.parsed_cache_validation_receipt,
                expected_validation_receipt_sha256=args.expected_parsed_cache_validation_sha256,
                expected_cache_key=args.expected_parsed_cache_key,
                expected_split_manifest_sha256=provenance["split_manifest_sha256"],
                expected_fold=int(ck["args"]["valid_fold"]),
            )
        except (ParsedCacheError, OSError, json.JSONDecodeError, KeyError) as exc:
            parser.error(f"parsed-cache inference validation failed: {exc}")
        cache_provenance = provenance.get("parsed_sequence_cache", {})
        if (
            cache_provenance.get("cache_key") != parsed_cache.manifest["cache_key"]
            or cache_provenance.get("manifest_sha256") != args.expected_parsed_cache_manifest_sha256
            or cache_provenance.get("validation_receipt_sha256")
            != args.expected_parsed_cache_validation_sha256
            or cache_provenance.get("membership_sha256")
            != parsed_cache.manifest["membership_artifact"]["sha256"]
        ):
            parser.error("checkpoint parsed-cache provenance differs from the inference cache")
        trained_groups = set(map(int, provenance.get("fit_row_groups", {}).get("train", [])))
        cache_groups = set(parsed_cache.groups("fit_train"))
        eval_groups = set(map(int, provenance.get("evaluation_row_groups", [])))
        if trained_groups != cache_groups or eval_groups != set(
            parsed_cache.groups("evaluation_valid")
        ):
            parser.error(
                "checkpoint fit/evaluation membership differs from parsed-cache membership"
            )
        groups = np.asarray(parsed_cache.groups("evaluation_valid"), dtype=np.int64)
        feature_only = True
    else:
        feature_only = args.fit_subset_manifest is not None
    if parsed_cache is None and feature_only:
        if (
            split != "valid"
            or ck["args"].get("numeric_map", "none") == "none"
            or ck["args"].get("fit_subset_manifest_sha256") is None
        ):
            parser.error("E540 fit-subset export requires a numeric-map checkpoint and valid split")
        groups = np.asarray(
            load_e540_evaluation_groups(args.fit_subset_manifest, ck), dtype=np.int64
        )
    elif parsed_cache is None:
        groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    quote_cache = ck.get("provenance", {}).get("quote_cache")
    if ck["args"].get("quote_cache_root") and quote_cache is None:
        raise ValueError("checkpoint lacks quote cache provenance")
    configure_torch()
    device = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(ck["sentinel_preprocessing"])
        if ck.get("sentinel_preprocessing")
        else None
    )
    if (
        args.inference_precision == "fp32"
        or ck["args"].get("precision") == "fp32"
        or ck.get("quantization") is not None
        or ck.get("factorization") is not None
    ):
        model._inference_fp32 = True
    if ck["args"].get("arch") == "gated":
        model.gate_mode = args.gate_mode
    elif args.gate_mode != "soft":
        parser.error("--gate-mode requires a gated checkpoint")
    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    train_rnn.HEURISTICS = bool(ck["args"].get("heuristics"))
    train_rnn.RESEARCH_FEATURES = ck["args"].get("research_features", "none")
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
        FeatureOnlySequences(
            split,
            groups.tolist(),
            parsed_cache=parsed_cache,
            parsed_cache_role="evaluation_valid" if parsed_cache is not None else None,
        )
        if feature_only
        else Sequences(split, groups)
    )
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
    )
    final = ROOT / "cache" / f"pred_{args.output_name or name}_{split}.npy"
    tmp = final.with_name(final.name + ".partial")  # renamed only after every sequence is written
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(groups), SEQUENCE_LENGTH, 2)
    )
    i = 0
    for batch in loader:
        x = batch if feature_only else batch[0]
        p = predict(model, x, 4000, device)
        out[i : i + len(p)] = p[..., :2]
        i += len(p)
    out.flush()
    del out
    if i != len(groups):
        raise RuntimeError(f"wrote {i} of {len(groups)} sequences")
    stale = (
        ROOT / "cache" / f"seqsums_{args.output_name or name}.npz"
    )  # cached metric sums of an older prediction file
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
        "row_groups": groups.tolist(),
        "evaluation_membership_sha256": (
            hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
            if feature_only
            else None
        ),
        "fit_subset_manifest_sha256": (
            file_sha256(args.fit_subset_manifest) if args.fit_subset_manifest is not None else None
        ),
        "parsed_sequence_cache": (
            {
                "cache_key": parsed_cache.manifest["cache_key"],
                "manifest_sha256": args.expected_parsed_cache_manifest_sha256,
                "validation_receipt_sha256": args.expected_parsed_cache_validation_sha256,
                "membership_sha256": parsed_cache.manifest["membership_artifact"]["sha256"],
            }
            if parsed_cache is not None
            else None
        ),
    }
    final.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
