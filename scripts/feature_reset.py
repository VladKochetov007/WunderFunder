"""Fixed current-student state-reset diagnostics; not a gradient-credit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from feature_cache import bound_path, selected_rows, validate_lineage
from feature_screen import digest_file
from feature_v2_screen import compare

from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel
from wnn.research_features import research_features_torch
from wnn.resources import configure_torch
from wnn.splits import load_campaign_split

LAGS = (250, 500, 1000, 2000)


def panel_groups(
    components: dict[int, list[int]], count: int = 16, group_limit: int = 32
) -> list[int]:
    """Choose complete evaluation components by fixed hash, never by score."""
    if not 4 <= count <= 32 or not 4 <= group_limit <= 64:
        raise ValueError("Panel bounds require4..32 components and4..64 row groups")
    order = sorted(
        components,
        key=lambda component: hashlib.sha256(
            f"reset-panel-v1:{component}:271828".encode()
        ).digest(),
    )
    groups, accepted = [], 0
    for component in order:
        if accepted == count:
            break
        addition = components[component]
        if len(groups) + len(addition) <= group_limit:
            groups.extend(addition)
            accepted += 1
    if accepted < 4:
        raise ValueError("Prespecified panel bound cannot fit at least four complete components")
    return sorted(groups)


def make_panel(
    root: Path, template: Path, output: Path, count: int = 16, group_limit: int = 32
) -> None:
    """Bind the current checkpoint and label-independent panel before any score reads."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(template.read_text())
    split = load_campaign_split(bound_path(root, manifest["campaign_split"]), root)
    component_of = {(source, group): component for source, group, component in split.components}
    groups: dict[int, list[int]] = {}
    for group in split.evaluation_groups:
        groups.setdefault(component_of[("valid", group)], []).append(group)
    manifest.update(
        version=1,
        source="valid",
        role="evaluation",
        row_groups=panel_groups(groups, count, group_limit),
        panel="fixed component hash; historically exposed; no score-based membership",
        max_components=count,
        max_row_groups=group_limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def reset_prediction(
    model: RNNModel, raw: np.ndarray, mode: str, device: str, lag: int | None, keep: float
) -> np.ndarray:
    """Preserve warmup and causal feature histories; reset only carried GRU state."""
    if lag is not None and lag not in LAGS or keep not in (0.0, 0.5, 1.0):
        raise ValueError("Use prespecified reset lags and state-retention controls")
    if len(raw) < 99:
        raise ValueError("The full99-row causal warmup must be available")
    tensor = torch.as_tensor(raw[None], device=device, dtype=torch.float32)
    inputs = torch.cat(
        (tensor, event_features_torch(tensor), research_features_torch(tensor, mode)), -1
    )
    boundaries = set(range(99 + lag, len(raw), lag)) if lag is not None else set()
    # Chunking bounds memory; reset positions are invariant to execution chunks.
    edges = sorted({0, len(raw), *range(1000, len(raw), 1000), *boundaries})
    state, predictions = None, []
    with torch.inference_mode():
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            if start in boundaries and state is not None:
                state = state * keep
            prediction, state = model(inputs[:, start:end], state)
            predictions.append(prediction[0, :, :2].cpu().numpy())
    return np.concatenate(predictions)


def run(
    root: Path,
    manifest_path: Path,
    output: Path,
    device: str,
    allow_fit_overlap: bool = False,
    bootstrap: int = 2000,
) -> None:
    """Score the unchanged current weights under all fixed state disruptions."""
    if output.exists() or bootstrap < 100:
        raise ValueError("Require new output directory and at least100 bootstrap draws")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("panel")
        != "fixed component hash; historically exposed; no score-based membership"
    ):
        raise ValueError("Use the prespecified reset-panel manifest")
    split_path = bound_path(root, manifest["campaign_split"])
    split = load_campaign_split(split_path, root)
    component_of = {(source, group): component for source, group, component in split.components}
    by_component: dict[int, list[int]] = {}
    for group in split.evaluation_groups:
        by_component.setdefault(component_of[("valid", group)], []).append(group)
    if manifest["row_groups"] != panel_groups(
        by_component, manifest["max_components"], manifest["max_row_groups"]
    ):
        raise ValueError("Panel membership differs from its input-only declaration")
    primary = torch.load(
        bound_path(root, manifest["checkpoint"]), map_location="cpu", weights_only=False
    )
    ancestry = {
        item["sha256"]: torch.load(bound_path(root, item), map_location="cpu", weights_only=False)
        for item in manifest["ancestors"]
    }
    fitted = validate_lineage(primary, ancestry, component_of)
    config = primary["args"]
    if (
        config.get("arch", "gru") != "gru"
        or config.get("enc", "linear") != "linear"
        or config.get("cell", "gru") != "gru"
        or config.get("privileged_mode", "none") != "none"
        or any(config.get(key) for key in ("online_norm", "heuristics", "prop_input", "raw_only"))
    ):
        raise ValueError("Current reset probe requires the supported causal linear-encoder GRU")
    exposed = fitted & {component_of[("valid", group)] for group in manifest["row_groups"]}
    if exposed and not allow_fit_overlap:
        raise ValueError(
            "This diagnostic panel overlaps model fitting; explicit declaration required"
        )
    configure_torch()
    model = RNNModel(
        primary["n_in"],
        config["hidden"],
        config.get("layers", 2),
        n_out=primary["model"]["head.2.weight"].shape[0],
        encoder_width=config.get("encoder_width"),
        recurrent_width=config.get("recurrent_width"),
        readout_width=config.get("readout_width"),
        head_layers=config.get("head_layers", 1),
    ).to(device)
    model.load_state_dict(primary["model"])
    model.eval()
    source = SequenceFile(root / "datasets/valid.parquet")
    masks = pq.ParquetFile(bound_path(root, manifest["mask"]))
    arrays: dict[str, list[np.ndarray]] = {}
    sampling = []
    for group in manifest["row_groups"]:
        sequence = source.read(group)
        table = masks.read_row_group(group, use_threads=False)
        scored = table.column("is_scored").to_numpy().astype(bool)
        if (
            sequence.targets is None
            or len(scored) != len(sequence.features)
            or not np.all(table.column("seq_ix").to_numpy() == sequence.seq_ix)
            or not np.array_equal(table.column("step_in_seq").to_numpy(), np.arange(len(scored)))
        ):
            raise ValueError("Reset panel requires aligned targets and explicit scored masks")
        planned = selected_rows(sequence.seq_ix, len(scored))
        rows = planned[scored[planned]]
        sampling.append(
            dict(
                row_group=group,
                sequence=sequence.seq_ix,
                planned_rows=planned.tolist(),
                selected_rows=rows.tolist(),
            )
        )
        if not len(rows):
            continue
        values = dict(
            y=sequence.targets[rows],
            row=rows,
            sequence=np.full(len(rows), sequence.seq_ix),
            component=np.full(len(rows), component_of[("valid", group)]),
        )
        mode = config.get("research_features", "none")
        values["no_reset"] = reset_prediction(model, sequence.features, mode, device, None, 1)[rows]
        for lag in LAGS:
            for keep in (0.0, 0.5):
                name = f"reset{lag}" if keep == 0 else f"attenuate{lag}"
                values[name] = reset_prediction(model, sequence.features, mode, device, lag, keep)[
                    rows
                ]
        for key, value in values.items():
            arrays.setdefault(key, []).append(value)
    if not arrays:
        raise ValueError("No scored rows in the complete prespecified reset panel")
    combined = {key: np.concatenate(value) for key, value in arrays.items()}
    reports = {
        str(lag): compare(
            combined,
            {
                "unchanged": combined["no_reset"],
                "candidate": combined[f"reset{lag}"],
                "attenuated": combined[f"attenuate{lag}"],
            },
            bootstrap,
        )
        for lag in LAGS
    }
    output.mkdir(parents=True)
    np.savez(output / "predictions.npz", **combined)
    (output / "report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "manifest_sha256": digest_file(manifest_path),
                "component_graph_sha256": digest_file(split_path),
                "fit_overlap_components": sorted(exposed),
                "allow_fit_overlap": allow_fit_overlap,
                "rows": len(combined["row"]),
                "sequences": len(np.unique(combined["sequence"])),
                "components": len(np.unique(combined["component"])),
                "sampling_report": sampling,
                "reset_rows": "99+k*lag for integer k>=1; retain full causal feature history",
                "lags": list(LAGS),
                "reports": reports,
                "bootstrap": bootstrap,
                "interpretation": "fixed-weight state-value disruption; not evidence about useful gradients",
                "selection": "none; exposed diagnostic panel; every fixed lag and control reported",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Prepare a label-independent panel or run a separately admitted reset diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("plan")
    execute = sub.add_parser("run")
    execute.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    execute.add_argument("--allow-fit-overlap", action="store_true")
    args = parser.parse_args()
    if args.operation == "plan":
        make_panel(args.root, args.manifest, args.output)
    else:
        run(args.root, args.manifest, args.output, args.device, args.allow_fit_overlap)


if __name__ == "__main__":
    main()
