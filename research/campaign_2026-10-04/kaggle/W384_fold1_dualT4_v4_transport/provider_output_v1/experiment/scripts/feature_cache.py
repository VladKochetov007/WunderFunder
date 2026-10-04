from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from feature_cues import causal_cues, lagged
from feature_screen import digest_file

from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel
from wnn.research_features import research_features_torch
from wnn.resources import configure_torch
from wnn.splits import load_campaign_split


def bound_path(root: Path, descriptor: dict) -> Path:
    """Read only a controller-declared file with matching content identity."""
    path = root / descriptor["path"]
    if digest_file(path) != descriptor["sha256"]:
        raise ValueError(f"Input hash mismatch: {descriptor['path']}")
    return path


def fitted_components(payload: dict, components: dict) -> set[int]:
    """Recover actual fitting membership from the checkpoint's embedded provenance."""
    provenance = payload["provenance"]
    groups = provenance["fit_row_groups"]
    return {
        components[(split, int(group))]
        for split in ("train", "valid")
        for group in groups.get(split, [])
    }


def validate_lineage(payload: dict, ancestry: dict[str, dict], components: dict) -> set[int]:
    """Require every declared learned ancestor, including teacher metadata bindings."""
    seen: set[str] = set()
    fitting: set[int] = set()

    def visit(checkpoint: dict) -> None:
        fitting.update(fitted_components(checkpoint, components))
        provenance = checkpoint["provenance"]
        args = checkpoint["args"]
        if not args.get("no_eval_during_fit") or args.get("checkpoint_policy") != "fixed":
            raise ValueError("Frozen model used evaluation-dependent checkpoint selection")
        if checkpoint["epoch"] != args["checkpoint_epoch"]:
            raise ValueError("Checkpoint is not the declared fixed epoch")
        references: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("initial_checkpoint_sha256", "propensity_checkpoint_sha256"):
                        references.add(str(item))
                    elif key == "checkpoint_sha256" and isinstance(item, str):
                        references.add(item)
                    elif key != "checkpoint_sha256":
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(provenance)
        collect(args)
        if args.get("distill") and not args.get("distill_provenance"):
            raise ValueError("Teacher lineage is not embedded in the checkpoint")
        for reference in references - seen:
            if reference not in ancestry:
                raise ValueError("Missing learned ancestor checkpoint in sealed lineage")
            seen.add(reference)
            visit(ancestry[reference])

    visit(payload)
    return fitting


def frozen_state(
    model: RNNModel, raw: np.ndarray, mode: str, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run one full causal sequence, retaining its actual head input and predictions."""
    tensor = torch.as_tensor(raw[None], device=device, dtype=torch.float32)
    inputs = torch.cat(
        (tensor, event_features_torch(tensor), research_features_torch(tensor, mode)), -1
    )
    pieces, predictions, state = [], [], None
    with torch.inference_mode():
        for chunk in inputs.split(1000, dim=1):
            projected = model.encode((chunk - model.mean) / model.std)
            recurrent, state = model.gru(projected, state)
            head_input = torch.cat((recurrent, projected), -1)
            pieces.append(head_input[0].cpu().numpy())
            predictions.append(model.head(head_input)[0, :, :2].cpu().numpy())
    return np.concatenate(pieces), np.concatenate(predictions)


def selected_rows(sequence_id: int, length: int) -> np.ndarray:
    """Jitter one row uniformly per stratum without aliasing an eight-row cycle."""
    seed = int.from_bytes(
        hashlib.sha256(f"feature-rows:{sequence_id}:271828".encode()).digest()[:8], "little"
    )
    starts = np.arange(227, length, 128)
    widths = np.minimum(128, length - starts)
    rng = np.random.default_rng(seed)
    return starts + rng.integers(0, widths)


def extract(root: Path, manifest_path: Path, output: Path, device: str) -> None:
    """Extract component-excluded caches using fixed input-only row sampling."""
    if output.exists() or output.suffix != ".npz":
        raise ValueError("Cache output must be a new .npz file")
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != 1 or manifest["role"] not in ("fit", "evaluation"):
        raise ValueError("Invalid frozen row-selection manifest")
    split_path = bound_path(root, manifest["campaign_split"])
    split = load_campaign_split(split_path, root)
    components = {(source, group): component for source, group, component in split.components}
    graph_sha = digest_file(split_path)
    primary_path = bound_path(root, manifest["checkpoint"])
    primary = torch.load(primary_path, map_location="cpu", weights_only=False)
    ancestry = {
        item["sha256"]: torch.load(bound_path(root, item), map_location="cpu", weights_only=False)
        for item in manifest["ancestors"]
    }
    fitted = validate_lineage(primary, ancestry, components)
    config = primary["args"]
    if config.get("arch", "gru") != "gru" or config.get("enc", "linear") != "linear":
        raise ValueError("Frozen-state screen currently supports the linear-encoder GRU")
    if config.get("privileged_mode", "none") != "none":
        raise ValueError("Privileged teacher checkpoints are not causal screening backbones")
    if config.get("cell", "gru") != "gru" or any(
        config.get(k) for k in ("online_norm", "heuristics", "prop_input", "raw_only")
    ):
        raise ValueError("Unsupported feature construction; supply an exact integration first")
    configure_torch()
    model = RNNModel(
        primary["n_in"],
        config["hidden"],
        config.get("layers", 2),
        n_out=primary["model"]["head.2.weight"].shape[0],
    ).to(device)
    model.load_state_dict(primary["model"])
    model.eval()
    mode = config.get("research_features", "none")
    selected = manifest["row_groups"]
    if selected != sorted(set(selected)) or manifest["source"] not in ("train", "valid"):
        raise ValueError("Row groups must be prespecified sorted unique indices")
    if not selected or len(selected) * 156 > 40_000:
        raise ValueError("Cache row bound exceeded")
    source = manifest["source"]
    row_components = {components[(source, group)] for group in selected}
    protected = {components[("valid", group)] for group in split.evaluation_groups}
    if fitted & (row_components | protected) or not row_components <= protected:
        raise ValueError("Backbone, residual-fit or outer-evaluation component leakage")
    # Campaign split validates source identity (and portable full hashes when enabled).
    sequences = SequenceFile(root / f"datasets/{source}.parquet")
    masks = pq.ParquetFile(bound_path(root, manifest["mask"]))
    arrays: dict[str, list[np.ndarray]] = {}
    sampling_report = []
    for group in selected:
        sequence = sequences.read(group)
        mask_table = masks.read_row_group(group, use_threads=False)
        scored = mask_table.column("is_scored").to_numpy().astype(bool)
        if (
            sequence.targets is None
            or len(scored) != len(sequence.features)
            or not np.all(mask_table.column("seq_ix").to_numpy() == sequence.seq_ix)
            or not np.array_equal(
                mask_table.column("step_in_seq").to_numpy(), np.arange(len(scored))
            )
        ):
            raise ValueError("Screening requires correctly aligned targets and explicit mask")
        planned_rows = selected_rows(sequence.seq_ix, len(sequence.features))
        rows = planned_rows[scored[planned_rows]]
        sampling_report.append(
            {
                "row_group": group,
                "sequence": sequence.seq_ix,
                "component": components[(source, group)],
                "planned_rows": planned_rows.tolist(),
                "selected_rows": rows.tolist(),
                "unscored_sampled_rows": planned_rows[~scored[planned_rows]].tolist(),
                "status": "retained" if len(rows) else "skipped_no_scored_sample",
            }
        )
        if not len(rows):
            continue
        state, prediction = frozen_state(model, sequence.features, mode, device)
        cues = causal_cues(sequence.features)
        values = {
            "state": state[rows],
            "pred": prediction[rows],
            "y": sequence.targets[rows],
            "fit_weight": np.ones(len(rows), dtype=np.float32),
            "component": np.full(len(rows), components[(source, group)]),
            "sequence": np.full(len(rows), sequence.seq_ix),
            "row": rows,
            "model_id": np.full(len(rows), manifest["checkpoint"]["sha256"]),
        }
        for name, cue in cues.items():
            values[f"cue__{name}"] = cue[rows]
            values[f"placebo__{name}"] = lagged(cue, 128)[rows]
        for name, value in values.items():
            arrays.setdefault(name, []).append(value)
    if not arrays:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "status": "empty_cache",
                    "rows": 0,
                    "sampling_report": sampling_report,
                    "skipped_row_groups": [item["row_group"] for item in sampling_report],
                    "selection_manifest_sha256": digest_file(manifest_path),
                },
                indent=2,
            )
            + "\n"
        )
        raise ValueError("No scored rows in the complete prespecified cache sample")
    combined = {name: np.concatenate(value) for name, value in arrays.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **combined)
    model_id = manifest["checkpoint"]["sha256"]
    metadata = {
        "sha256": digest_file(output),
        "rows": len(combined["row"]),
        "state_width": combined["state"].shape[1],
        "outer_fold": split.fold,
        "component_graph_sha256": graph_sha,
        "role": manifest["role"],
        "selection_manifest_sha256": digest_file(manifest_path),
        "sampling": "SHA(seqID,271828) uniform jitter within128-row strata from227, then scoring mask; causal lag128 placebo",
        "sampling_report": sampling_report,
        "skipped_row_groups": [
            item["row_group"]
            for item in sampling_report
            if item["status"] == "skipped_no_scored_sample"
        ],
        "planned_rows": sum(len(item["planned_rows"]) for item in sampling_report),
        "unscored_sampled_rows": sum(
            len(item["unscored_sampled_rows"]) for item in sampling_report
        ),
        "members": {model_id: {"checkpoint_sha256": model_id, "fit_components": sorted(fitted)}},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    """Create one strong frozen-state cache after controller lineage review."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    extract(args.root, args.manifest, args.output, args.device)


if __name__ == "__main__":
    main()
