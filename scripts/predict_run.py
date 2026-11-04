"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from wnn.distillation_cache import row_groups_sha256
from wnn.final_submit_manifest import FinalFitMembership, load_final_fit_membership
from wnn.propensity import load_propensity
from wnn.research_features import (
    MOMENT_FEATURE_MODES,
    research_feature_control_width,
    validate_causal_moment_prior,
)
from wnn.resources import configure_strict_fp32, configure_torch, dataloader_budget
from wnn.sentinel import SentinelPreprocessor
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.chronological_teacher import (
    select_chronological_full_train_index,
)
from wnn.training.final_submit_bridge import (
    final_fit_identity,
    validate_final_fit_artifact,
    validate_final_fit_parent,
    validate_source_validation_receipt,
)
from wnn.training.prediction_inputs import configure_prediction_inputs

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", SOURCE_ROOT)).resolve()


def campaign_prediction_groups(
    manifest_path: Path,
    checkpoint: dict,
    split: str,
    *,
    evaluation_only: bool = False,
    full_valid: bool = False,
    full_train_index: bool = False,
    checkpoint_file: str | None = None,
    checkpoint_sha256: str | None = None,
    run_provenance: dict[str, object] | None = None,
) -> np.ndarray:
    """Bind feature-only campaign exports to checkpoint ancestry and row order."""
    if split not in ("train", "valid"):
        raise ValueError("campaign exports require train or valid source")
    selectors = sum((evaluation_only, full_valid, full_train_index))
    if selectors > 1:
        raise ValueError("campaign row selectors are mutually exclusive")
    if (evaluation_only or full_valid) and split != "valid":
        raise ValueError("evaluation/full-valid selectors require the valid source")
    if full_train_index and split != "train":
        raise ValueError("full-train-index selector requires the train source")
    manifest = load_campaign_split(manifest_path, ROOT)
    if manifest.sha256 != checkpoint.get("args", {}).get("split_manifest_sha256"):
        raise ValueError("campaign manifest differs from the fitted checkpoint")
    if full_train_index:
        if checkpoint_file is None or checkpoint_sha256 is None or run_provenance is None:
            raise ValueError("full train index export needs checkpoint and run provenance")
        groups = select_chronological_full_train_index(
            ROOT,
            manifest,
            checkpoint,
            checkpoint_file,
            checkpoint_sha256,
            run_provenance,
            str(checkpoint.get("args", {}).get("name", "teacher")),
        )
    elif full_valid:
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


def final_fit_prediction_groups(
    manifest_path: Path,
    receipt_path: Path,
    source_validation_sha256: str,
    checkpoint: dict,
    split: str,
    run_provenance: dict[str, object],
) -> tuple[np.ndarray, FinalFitMembership]:
    """Select source-index rows after validating the exact all-data checkpoint lineage."""
    if split not in ("train", "valid"):
        raise ValueError("final-fit exports require a train or valid source")
    membership = load_final_fit_membership(manifest_path, ROOT, source_validation_sha256)
    validate_source_validation_receipt(receipt_path, source_validation_sha256, membership)
    validate_final_fit_parent(checkpoint, membership)
    run_args = run_provenance.get("args")
    if not isinstance(run_args, dict) or type(checkpoint.get("epoch")) is not int:
        raise ValueError("final-fit run provenance omits configured checkpoint arguments")
    validate_final_fit_artifact(
        run_args, run_provenance, membership, checkpoint_epoch=checkpoint["epoch"]
    )
    checkpoint_provenance = checkpoint.get("provenance", {})
    identity = final_fit_identity(membership)
    if (
        not isinstance(checkpoint_provenance, dict)
        or checkpoint_provenance.get("final_fit") != identity
        or run_provenance.get("final_fit") != identity
        or run_provenance.get("evaluation_row_groups") != []
    ):
        raise ValueError("final-fit export has different run or checkpoint provenance")
    with np.load(ROOT / "cache" / f"{split}_targets.npz", allow_pickle=False) as targets:
        raw_groups = np.asarray(targets["row_group"])
    if raw_groups.ndim != 1 or raw_groups.dtype.kind not in "iu":
        raise ValueError(f"final-fit {split} target index is malformed")
    groups = raw_groups.astype(np.int64, copy=False)
    expected = membership.train_groups if split == "train" else membership.valid_groups
    if groups.tolist() != list(expected):
        raise ValueError(f"final-fit {split} target index does not preserve source row order")
    fit_groups = run_provenance.get("fit_row_groups")
    expected_members = {
        "train": membership.train_groups,
        "valid": membership.valid_groups,
    }
    if not isinstance(fit_groups, dict) or any(
        not isinstance(fit_groups.get(source), list) or set(fit_groups[source]) != set(members)
        for source, members in expected_members.items()
    ):
        raise ValueError("final-fit export parent does not cover both labeled sources")
    return groups, membership


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--campaign-split-manifest", type=Path)
    parser.add_argument("--campaign-evaluation-only", action="store_true")
    parser.add_argument("--campaign-full-valid-feature-only", action="store_true")
    parser.add_argument("--final-fit-manifest", type=Path)
    parser.add_argument("--final-fit-source-validation-receipt", type=Path)
    parser.add_argument("--final-fit-source-validation-sha256")
    parser.add_argument(
        "--campaign-full-train-index-feature-only",
        action="store_true",
        help="export a chronological teacher in the full source train-index order",
    )
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
    args = parser.parse_args()
    name, split = args.name, args.split
    if (
        args.campaign_evaluation_only
        or args.campaign_full_valid_feature_only
        or args.campaign_full_train_index_feature_only
    ) and (args.campaign_split_manifest is None):
        parser.error("campaign row selectors require --campaign-split-manifest")
    if args.fit_subset_manifest is not None and args.campaign_split_manifest is not None:
        parser.error("E540 and campaign selectors are mutually exclusive")
    final_fit_requested = args.final_fit_manifest is not None
    receipt_requested = (
        args.final_fit_source_validation_receipt is not None
        or args.final_fit_source_validation_sha256 is not None
    )
    if final_fit_requested != receipt_requested or (
        final_fit_requested
        and (
            args.final_fit_source_validation_receipt is None
            or args.final_fit_source_validation_sha256 is None
        )
    ):
        parser.error("--final-fit-manifest requires source-validation receipt path and SHA256")
    if final_fit_requested and (
        args.campaign_split_manifest is not None or args.fit_subset_manifest is not None
    ):
        parser.error("final-fit, campaign and E540 export selectors are mutually exclusive")
    checkpoint_name = f"model_ep{args.epoch}.pt" if args.epoch is not None else "model.pt"
    checkpoint_path = ROOT / "artifacts" / "runs" / name / checkpoint_name
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    run_provenance_path = ROOT / "artifacts" / "runs" / name / "provenance.json"
    run_provenance = None
    run_provenance_sha256 = None
    if args.campaign_full_train_index_feature_only or final_fit_requested:
        run_provenance = json.loads(run_provenance_path.read_text(encoding="utf-8"))
        run_provenance_sha256 = file_sha256(run_provenance_path)
    if ck["args"].get("privileged_mode", "none") != "none":
        raise ValueError(
            "privileged teachers require component-crossfit extraction, not causal prediction export"
        )
    final_fit_membership = None
    feature_only = (
        args.fit_subset_manifest is not None
        or args.campaign_split_manifest is not None
        or final_fit_requested
    )
    if final_fit_requested:
        groups, final_fit_membership = final_fit_prediction_groups(
            args.final_fit_manifest,
            args.final_fit_source_validation_receipt,
            args.final_fit_source_validation_sha256,
            ck,
            split,
            run_provenance,
        )
    elif args.campaign_split_manifest is not None:
        groups = campaign_prediction_groups(
            args.campaign_split_manifest,
            ck,
            split,
            evaluation_only=args.campaign_evaluation_only,
            full_valid=args.campaign_full_valid_feature_only,
            full_train_index=args.campaign_full_train_index_feature_only,
            checkpoint_file=checkpoint_name,
            checkpoint_sha256=checkpoint_sha256,
            run_provenance=run_provenance,
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
    quote_cache = ck.get("provenance", {}).get("quote_cache")
    if ck["args"].get("quote_cache_root") and quote_cache is None:
        raise ValueError("checkpoint lacks quote cache provenance")
    configure_torch()
    device = torch.device(args.device)
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
        configure_strict_fp32()
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
    input_propensity = ck["args"].get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts" / "runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    prediction_environment = train_rnn._api.environment()
    configure_prediction_inputs(prediction_environment, ck.get("args", {}))
    dataset = (
        FeatureOnlySequences(split, groups.tolist()) if feature_only else Sequences(split, groups)
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
            p = predict(model, x, 4000, device, environment=prediction_environment)
        else:  # research probe: keep every output column (family heads are the last 8)
            model.eval()
            with torch.no_grad():
                inputs = train_rnn.build_inputs(
                    x.to(device, non_blocking=True), environment=prediction_environment
                )
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
            fam[i : i + len(p)] = np.concatenate([full[..., :2], full[..., -8:]], axis=-1)
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
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_file": checkpoint_name,
        "run_provenance_sha256": (
            run_provenance_sha256
            if run_provenance_sha256 is not None
            else file_sha256(run_provenance_path)
            if run_provenance_path.is_file()
            else None
        ),
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": ck["args"].get("split_manifest_sha256"),
        "final_fit_manifest_sha256": (
            None if final_fit_membership is None else final_fit_membership.manifest_sha256
        ),
        "final_fit_members_sha256": (
            None if final_fit_membership is None else final_fit_membership.fit_members_sha256
        ),
        "source_validation_sha256": (
            None if final_fit_membership is None else final_fit_membership.source_validation_sha256
        ),
        "prediction_role": (
            "fitting-only teacher features; no independent score"
            if final_fit_membership is not None
            else None
        ),
        "independent_score_claim": ("forbidden" if final_fit_membership is not None else None),
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
            if feature_only and not final_fit_requested
            else None
        ),
        "row_groups_sha256": row_groups_sha256(groups.tolist()) if feature_only else None,
        "fit_subset_manifest_sha256": (
            file_sha256(args.fit_subset_manifest) if args.fit_subset_manifest is not None else None
        ),
        "campaign_export_scope": (
            "final-fit-source-index-feature-only"
            if final_fit_requested
            else "full-valid-feature-only"
            if args.campaign_full_valid_feature_only
            else "chronological-full-train-index-feature-only"
            if args.campaign_full_train_index_feature_only
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
    final.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
