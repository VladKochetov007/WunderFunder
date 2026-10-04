"""Disjoint-episode mature-gain recoverability from a fixed causal student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from feature_cache import bound_path, frozen_state, selected_rows, validate_lineage
from feature_screen import digest_file, projection_pair, solve
from feature_v2_screen import compare

from wnn.data import SequenceFile
from wnn.nn import RNNModel
from wnn.research_features import ENDPOINT
from wnn.resources import configure_torch
from wnn.splits import load_campaign_split

LABEL_START, LABEL_END, MATURITY_EVENTS, FUTURE_GAP = 1024, 4096, 48, 250


def episode_anchor(raw: np.ndarray) -> int | None:
    """Use only observable quote events to declare when the earlier episode is mature."""
    if raw.ndim != 2 or raw.shape[1] != 112 or len(raw) <= LABEL_END:
        return None
    events = np.flatnonzero(raw[LABEL_END:, 110] == ENDPOINT) + LABEL_END
    if len(events) < MATURITY_EVENTS:
        return None
    anchor = int(events[MATURITY_EVENTS - 1])
    return anchor if anchor + FUTURE_GAP < len(raw) else None


def realised_gain(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Fixed weighted scale estimate, weakly regularized toward one before clipping."""
    y, p = np.clip(target, -2, 2).astype(np.float64), np.clip(prediction, -2, 2).astype(np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or len(y) < 32:
        raise ValueError("Gain target needs at least32 aligned rows of both targets")
    weight = np.abs(y)
    if np.any(weight.sum(0) <= 0) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Gain target has zero weight or nonfinite values")
    prior = 0.01 * weight.sum(0)
    return np.clip(
        ((weight * p * y).sum(0) + prior) / ((weight * p**2).sum(0) + prior), 0.25, 4.0
    ).astype(np.float32)


def input_summary(raw: np.ndarray, anchor: int) -> np.ndarray:
    """Causal raw-history baseline at the same anchor as the frozen GRU state."""
    x = raw[: anchor + 1].astype(np.float64)
    return np.concatenate((x.mean(0), x.std(0), np.abs(np.diff(x, axis=0)).mean(0))).astype(
        np.float32
    )


def extract(root: Path, manifest_path: Path, output: Path, device: str = "cpu") -> None:
    """Cache past gain labels, causal anchor features, and a later disjoint scored episode."""
    if output.exists() or output.with_suffix(".json").exists() or output.suffix != ".npz":
        raise ValueError("Gain cache requires a new .npz path")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"] != "valid" or manifest["row_groups"] != sorted(
        set(manifest["row_groups"])
    ):
        raise ValueError("Use prespecified canonical VALID component partitions")
    if len(manifest["row_groups"]) > 256:
        raise ValueError("At most256 sequences per gain-cache extraction")
    split_path = bound_path(root, manifest["campaign_split"])
    split = load_campaign_split(split_path, root)
    components = {(source, group): component for source, group, component in split.components}
    primary = torch.load(
        bound_path(root, manifest["checkpoint"]), map_location="cpu", weights_only=False
    )
    ancestry = {
        item["sha256"]: torch.load(bound_path(root, item), map_location="cpu", weights_only=False)
        for item in manifest["ancestors"]
    }
    fitted = validate_lineage(primary, ancestry, components)
    protected = {components[("valid", group)] for group in split.evaluation_groups}
    selected = {components[("valid", group)] for group in manifest["row_groups"]}
    if fitted & protected or not selected <= protected:
        raise ValueError("Gain-probe backbone or ancestor fitted protected components")
    config = primary["args"]
    if (
        config.get("arch", "gru") != "gru"
        or config.get("enc", "linear") != "linear"
        or config.get("cell", "gru") != "gru"
        or config.get("privileged_mode", "none") != "none"
        or any(config.get(key) for key in ("online_norm", "heuristics", "prop_input", "raw_only"))
    ):
        raise ValueError("Gain probe requires a supported causal linear-encoder GRU")
    configure_torch()
    model = RNNModel(
        primary["n_in"],
        config["hidden"],
        config.get("layers", 2),
        n_out=primary["model"]["head.2.weight"].shape[0],
    ).to(device)
    model.load_state_dict(primary["model"])
    model.eval()
    source = SequenceFile(root / "datasets/valid.parquet")
    masks = pq.ParquetFile(bound_path(root, manifest["mask"]))
    sequence_arrays: dict[str, list] = {}
    row_arrays: dict[str, list] = {}
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
            raise ValueError("Gain episodes require aligned targets and explicit scored masks")
        anchor = episode_anchor(sequence.features)
        record = dict(
            row_group=group,
            sequence=sequence.seq_ix,
            anchor=anchor,
            component=components[("valid", group)],
        )
        sampling.append(record)
        if anchor is None:
            record["status"] = "insufficient_observed_maturity_or_future_rows"
            continue
        earlier = np.flatnonzero(scored[LABEL_START:LABEL_END]) + LABEL_START
        planned = selected_rows(sequence.seq_ix, len(scored))
        future = planned[scored[planned] & (planned >= anchor + FUTURE_GAP)]
        record.update(earlier_rows=len(earlier), selected_future_rows=future.tolist())
        if len(earlier) < 32 or not len(future):
            record["status"] = "insufficient_prespecified_episode_samples"
            continue
        state, prediction = frozen_state(
            model, sequence.features, config.get("research_features", "none"), device
        )
        try:
            gain = realised_gain(sequence.targets[earlier], prediction[earlier])
        except ValueError as error:
            record.update(status="undefined_earlier_gain", reason=str(error))
            continue
        record["status"] = "retained"
        values = dict(
            # A row view would retain the entire20k-row state array per sequence.
            state=state[anchor].copy(),
            summary=input_summary(sequence.features, anchor),
            gain=gain,
            anchor=anchor,
            seq_id=sequence.seq_ix,
            seq_component=components[("valid", group)],
        )
        for key, value in values.items():
            sequence_arrays.setdefault(key, []).append(value)
        rows = dict(
            pred=prediction[future],
            y=sequence.targets[future],
            row=future,
            sequence=np.full(len(future), sequence.seq_ix),
            component=np.full(len(future), components[("valid", group)]),
        )
        for key, value in rows.items():
            row_arrays.setdefault(key, []).append(value)
    if not sequence_arrays:
        raise ValueError("No sequences retained by the complete fixed gain-episode protocol")
    sequence_data = {key: np.stack(value) for key, value in sequence_arrays.items()}
    row_data = {key: np.concatenate(value) for key, value in row_arrays.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **sequence_data, **row_data)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": 1,
                "sha256": digest_file(output),
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "component_graph_sha256": digest_file(split_path),
                "fit_components": sorted(fitted),
                "outer_fold": split.fold,
                "selection_manifest_sha256": digest_file(manifest_path),
                "rows": len(row_data["row"]),
                "sequences": len(sequence_data["seq_id"]),
                "sampling_report": sampling,
                "label_episode": [LABEL_START, LABEL_END],
                "maturity_events": MATURITY_EVENTS,
                "event_definition": "a6 exact lower endpoint",
                "future_gap_rows": FUTURE_GAP,
                "maturity_scope": "prespecified48-event buffer from archived30–48-event estimate, not a new maturity proof",
                "label_policy": "earlier disjoint targets are supervision/diagnostic only; never predictor inputs",
            },
            indent=2,
        )
        + "\n"
    )


def load_cache(path: Path) -> tuple[dict, dict]:
    """Validate variable-length future rows against their sequence-level causal anchors."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["version"] != 1 or metadata["sha256"] != digest_file(path):
        raise ValueError("Gain cache hash/version mismatch")
    if (
        metadata.get("label_episode") != [LABEL_START, LABEL_END]
        or metadata.get("maturity_events") != MATURITY_EVENTS
        or metadata.get("future_gap_rows") != FUTURE_GAP
    ):
        raise ValueError("Gain cache episode protocol mismatch")
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    count = metadata["sequences"]
    if not 4 <= count <= 256 or len(np.unique(data["seq_id"])) != count:
        raise ValueError("Gain cache sequence count invalid")
    for key in ("state", "summary", "gain", "anchor", "seq_id", "seq_component"):
        if len(data[key]) != count or not np.isfinite(data[key]).all():
            raise ValueError("Invalid gain-cache sequence arrays")
    for key in ("pred", "y", "row", "sequence", "component"):
        if len(data[key]) != metadata["rows"] or not np.isfinite(data[key]).all():
            raise ValueError("Invalid gain-cache future rows")
    if (
        data["gain"].shape != (count, 2)
        or data["summary"].shape != (count, 336)
        or data["pred"].shape != (metadata["rows"], 2)
        or data["y"].shape != data["pred"].shape
        or data["state"].ndim != 2
        or data["state"].shape[1] > 1024
        or np.any((data["gain"] < 0.25) | (data["gain"] > 4))
    ):
        raise ValueError("Invalid gain target/summary shape")
    if set(map(int, data["seq_component"])) & set(metadata["fit_components"]):
        raise ValueError("Gain cache backbone fitted a protected component")
    if set(data["sequence"].tolist()) != set(data["seq_id"].tolist()):
        raise ValueError("Future row sequence identity differs from anchor identities")
    for index, sequence in enumerate(data["seq_id"]):
        selected = data["sequence"] == sequence
        if (
            np.any(data["row"][selected] < data["anchor"][index] + FUTURE_GAP)
            or np.any(data["component"][selected] != data["seq_component"][index])
            or data["anchor"][index] < LABEL_END
        ):
            raise ValueError("Overlapping episodes or component identity mismatch")
    return data, metadata


def gain_heads(fit: dict, evaluation: dict, backend: str) -> tuple[dict, dict, dict]:
    """Fit equal-capacity state/summary probes and fixed shuffled supervision controls."""
    targets = fit["gain"].astype(np.float64)
    mean_gain = targets.mean(0)
    predictions = {"constant": np.broadcast_to(mean_gain, evaluation["gain"].shape).copy()}
    coefficients, transforms = {}, {"fit_mean_gain": mean_gain}
    order = np.random.default_rng(271831).permutation(len(targets))
    transforms["shuffled_fit_labels"] = order
    for name, key, labels in (
        ("candidate", "state", targets),
        ("input_summary", "summary", targets),
        ("shuffled_gain", "state", targets[order]),
    ):
        parameters = {}
        train, test = projection_pair(fit[key], evaluation[key], 64, 271829, parameters)
        train, test = (np.column_stack((np.ones(len(value)), value)) for value in (train, test))
        beta = solve(
            train.T @ train / len(train) + 0.01 * np.eye(train.shape[1]),
            train.T @ (labels - mean_gain) / len(train),
            backend,
        )
        predictions[name] = np.clip(mean_gain + test @ beta, 0.25, 4.0)
        coefficients[name] = beta
        transforms.update({f"{name}__{key}": value for key, value in parameters.items()})
    return predictions, coefficients, transforms


def run(
    fit_path: Path, eval_path: Path, output: Path, backend: str = "numpy", bootstrap: int = 2000
) -> None:
    """Assess gain recoverability and later prediction benefit without evaluation selection."""
    if output.exists() or bootstrap < 100:
        raise ValueError("New output directory and at least100 bootstrap draws required")
    fit, fit_meta = load_cache(fit_path)
    evaluation, eval_meta = load_cache(eval_path)
    if (
        fit_meta["checkpoint_sha256"] != eval_meta["checkpoint_sha256"]
        or fit_meta["component_graph_sha256"] != eval_meta["component_graph_sha256"]
        or fit_meta["outer_fold"] != eval_meta["outer_fold"]
        or set(fit["seq_component"].tolist()) & set(evaluation["seq_component"].tolist())
        or len(np.unique(fit["seq_component"])) < 4
    ):
        raise ValueError("Gain fit/evaluation backbone, component or fold separation failed")
    for meta in (fit_meta, eval_meta):
        if set(meta["fit_components"]) & set(evaluation["seq_component"].tolist()):
            raise ValueError("Gain learner ancestry fitted evaluation components")
    gains, coefficients, transforms = gain_heads(fit, evaluation, backend)
    # Oracle is a non-deployable stability diagnostic. It never enters gain-head fitting.
    gains["earlier_label_oracle"] = evaluation["gain"]
    row_index = {int(sequence): index for index, sequence in enumerate(evaluation["seq_id"])}
    mapping = np.array([row_index[int(sequence)] for sequence in evaluation["sequence"]])
    predictions = {"unchanged": evaluation["pred"]}
    predictions.update({name: evaluation["pred"] * gain[mapping] for name, gain in gains.items()})
    reports = compare(evaluation, predictions, bootstrap)
    oracle_comparison = compare(
        evaluation,
        {"unchanged": evaluation["pred"], "candidate": predictions["earlier_label_oracle"]},
        bootstrap,
    )
    reports["earlier_label_oracle_minus_unchanged"] = oracle_comparison["candidate_minus_unchanged"]
    reference_error = np.mean((evaluation["gain"] - fit["gain"].mean(0)) ** 2, axis=0)
    recoverability = {}
    for name, value in gains.items():
        if name == "earlier_label_oracle":
            continue
        mse = np.mean((evaluation["gain"] - value) ** 2, axis=0)
        recoverability[name] = {
            "mse_per_target": mse.tolist(),
            "r2_vs_fit_mean_per_target": [
                float(1 - error / reference) if reference > 1e-12 else None
                for error, reference in zip(mse, reference_error, strict=True)
            ],
        }
    output.mkdir(parents=True)
    np.savez(
        output / "predictions.npz",
        **predictions,
        sequence=evaluation["sequence"],
        component=evaluation["component"],
        row=evaluation["row"],
    )
    np.savez(
        output / "gain_predictions.npz",
        **gains,
        seq_id=evaluation["seq_id"],
        seq_component=evaluation["seq_component"],
        target=evaluation["gain"],
    )
    np.savez(output / "coefficients.npz", **coefficients)
    np.savez(output / "transforms.npz", **transforms)
    (output / "report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": backend,
                "fit_cache_sha256": fit_meta["sha256"],
                "evaluation_cache_sha256": eval_meta["sha256"],
                "reports": reports,
                "recoverability": recoverability,
                "head_width": 65,
                "ridge": 0.01,
                "label_episode": [LABEL_START, LABEL_END],
                "maturity_events": MATURITY_EVENTS,
                "future_gap_rows": FUTURE_GAP,
                "bootstrap": bootstrap,
                "oracle_scope": "earlier labels unavailable at deployment; diagnostic of cross-episode stability only",
                "selection": "none; component-crossfit stageA; does not implement or establish benefit from auxiliary training",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Prepare gain episodes or run the fixed recoverability/stability screen."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--root", type=Path, default=Path.cwd())
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cache.add_argument("--output", type=Path, required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--fit", type=Path, required=True)
    screen.add_argument("--evaluation", type=Path, required=True)
    screen.add_argument("--backend", choices=("numpy", "torch", "jax"), default="numpy")
    screen.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "cache":
        extract(args.root, args.manifest, args.output, args.device)
    else:
        run(args.fit, args.evaluation, args.output, args.backend)


if __name__ == "__main__":
    main()
