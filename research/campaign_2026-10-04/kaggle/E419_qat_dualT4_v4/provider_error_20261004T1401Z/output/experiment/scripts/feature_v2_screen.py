"""Prespecified exact-mechanism screens on one component-excluded strong backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from feature_screen import assert_outer_separation, digest_file, load_cache, projection_pair, solve

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import reference_global_wp

FAMILIES = (
    "regime_heads",
    "regime_affine",
    "target_clocks",
    "activity_interactions",
    "event_age",
    "i1_innovations",
    "transitions",
    "cycle8",
)


def load_histories(path: Path, base: dict, base_meta: dict) -> tuple[dict, dict]:
    """Require exact base row identity and content-bound, excluded TRAIN routing."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (
        metadata["version"] != 2
        or metadata["sha256"] != digest_file(path)
        or metadata["base_cache_sha256"] != base_meta["sha256"]
        or metadata["component_graph_sha256"] != base_meta["component_graph_sha256"]
    ):
        raise ValueError("History/base content identity mismatch")
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    for key in ("row", "sequence", "component"):
        if not np.array_equal(base[key], data[key]):
            raise ValueError("History/base row alignment mismatch")
    for key, value in data.items():
        if len(value) != len(base["row"]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid history array: {key}")
    if (
        data["regime"].shape != (len(base["row"]), 1)
        or data["regime"].dtype.kind not in "iu"
        or not np.isin(data["regime"], range(4)).all()
    ):
        raise ValueError("Invalid four-class routing")
    return data, metadata


def hashed_regime(sequences: np.ndarray) -> np.ndarray:
    """Fixed sequence routing placebo independent of inputs and targets."""
    assignment = {
        int(seq): int.from_bytes(
            hashlib.sha256(f"feature-regime-placebo:{seq}:271828".encode()).digest()[:8], "little"
        )
        % 4
        for seq in np.unique(sequences)
    }
    return np.array([assignment[int(seq)] for seq in sequences])


def innovation_pair(
    fit: dict, evaluation: dict, instrument: int, transforms: dict
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit a conditional raw-input predictor only on the residual fitting components."""
    other = 1 - instrument
    source = np.concatenate((fit[f"i{other}_lag0"], fit[f"i{instrument}_lag1"]), axis=1)
    mean, scale = (
        source.mean(0, dtype=np.float64),
        np.maximum(source.std(0, dtype=np.float64), 1e-4),
    )

    def design(data: dict, offset: int) -> np.ndarray:
        values = np.concatenate(
            (data[f"i{other}_lag{offset}"], data[f"i{instrument}_lag{offset + 1}"]), axis=1
        )
        return np.column_stack((np.ones(len(values)), np.clip((values - mean) / scale, -8, 8)))

    matrix = design(fit, 0)
    beta = np.linalg.solve(
        matrix.T @ matrix / len(matrix) + 0.01 * np.eye(matrix.shape[1]),
        matrix.T @ fit[f"i{instrument}_lag0"] / len(matrix),
    )
    transforms.update(
        {
            f"innovation_i{instrument}__mean": mean,
            f"innovation_i{instrument}__scale": scale,
            f"innovation_i{instrument}__beta": beta,
        }
    )
    return {
        str(offset): tuple(
            data[f"i{instrument}_lag{offset}"] - design(data, offset) @ beta
            for data in (fit, evaluation)
        )
        for offset in (0, 8, 136)
    }


def build_designs(
    fit: dict, evaluation: dict, fit_history: dict, eval_history: dict, family: str
) -> tuple[dict, dict]:
    """Prepare fixed, equally sized candidate/placebo/capacity maps before fitting targets."""
    if family not in FAMILIES:
        raise ValueError("Unknown exact feature family")
    bases, histories = (fit, evaluation), (fit_history, eval_history)
    transforms: dict[str, np.ndarray] = {}

    def project(name: str, sources: tuple, width: int = 32) -> tuple:
        parameters: dict[str, np.ndarray] = {}
        pair = projection_pair(*sources, width, 271830, parameters)
        transforms.update({f"{name}__{key}": value for key, value in parameters.items()})
        return pair

    def feature(name: str) -> tuple:
        return project(name, tuple(data[name] for data in histories))

    baseline = project("state_baseline", tuple(data["state"] for data in bases), 64)

    def common(extra: tuple) -> list[list[np.ndarray]]:
        matrices = [
            np.column_stack((np.ones(len(data["row"])), data["pred"], state, addition))
            for data, state, addition in zip(bases, baseline, extra, strict=True)
        ]
        return [matrices, matrices]

    capacity = project("state_capacity", tuple(data["state"] for data in bases))
    designs = {"capacity": common(capacity)}
    if family == "regime_heads":
        state = project("routed_state", tuple(data["state"] for data in bases))
        for name, routes in {
            "candidate": tuple(data["regime"][:, 0] for data in histories),
            "hash_route": tuple(hashed_regime(data["sequence"]) for data in bases),
        }.items():
            matrices = [
                np.column_stack(
                    (
                        np.ones(len(data["row"])),
                        data["pred"],
                        np.concatenate(
                            [values * (routing == index)[:, None] for index in range(4)], axis=1
                        ),
                    )
                )
                for data, values, routing in zip(bases, state, routes, strict=True)
            ]
            designs[name] = [matrices, matrices]
        universal = project("universal_state", tuple(data["state"] for data in bases), 128)
        matrices = [
            np.column_stack((np.ones(len(data["row"])), data["pred"], values))
            for data, values in zip(bases, universal, strict=True)
        ]
        designs["capacity"] = [matrices, matrices]
    elif family == "regime_affine":
        designs = {}
        for name, routes in {
            "candidate": tuple(data["regime"][:, 0] for data in histories),
            "hash_route": tuple(hashed_regime(data["sequence"]) for data in bases),
        }.items():
            designs[name] = [
                [
                    np.column_stack(
                        (np.eye(4)[routing], np.eye(4)[routing] * data["pred"][:, target, None])
                    )
                    for routing, data in zip(routes, bases, strict=True)
                ]
                for target in range(2)
            ]
        designs["global_affine"] = [
            [
                np.column_stack((np.ones(len(data["row"])), data["pred"][:, target]))
                for data in bases
            ]
            for target in range(2)
        ]
        extra = project("affine_capacity", tuple(data["state"] for data in bases), 6)
        designs["capacity"] = [
            [
                np.column_stack((np.ones(len(data["row"])), data["pred"][:, target], values))
                for data, values in zip(bases, extra, strict=True)
            ]
            for target in range(2)
        ]
    elif family == "target_clocks":
        row, event, stale = (
            feature(name) for name in ("clock_row", "clock_event", "clock_stale_event")
        )
        joint = project(
            "clock_common",
            tuple(
                np.concatenate((data["clock_row"], data["clock_event"]), 1) for data in histories
            ),
        )
        designs.update(
            candidate=[common(row)[0], common(event)[1]],
            swapped=[common(event)[0], common(row)[1]],
            stale_event=[common(row)[0], common(stale)[1]],
            common_clock=common(joint),
            row_both=common(row),
            event_both=common(event),
        )
    elif family in ("activity_interactions", "event_age"):
        name = "activity" if family == "activity_interactions" else "event_age"
        history, gate, stale = feature("clock_row"), feature(name), feature(name + "_stale")
        order = np.random.default_rng(271831).permutation(len(gate[0]))
        transforms["shuffled_fit_gate__order"] = order
        shuffled = (gate[0][order], gate[1])
        designs.update(
            candidate=common(tuple(a * b for a, b in zip(history, gate, strict=True))),
            additive=common(
                tuple((a + b) / np.sqrt(2) for a, b in zip(history, gate, strict=True))
            ),
            stale_gate=common(tuple(a * b for a, b in zip(history, stale, strict=True))),
            shuffled_fit_gate=common(tuple(a * b for a, b in zip(history, shuffled, strict=True))),
            history_only=common(history),
        )
    elif family == "i1_innovations":
        i1 = innovation_pair(*histories, 1, transforms)
        i0 = innovation_pair(*histories, 0, transforms)
        for name, sources in {
            "candidate": i1["8"],
            "current_i1": i1["0"],
            "shifted_i1": i1["136"],
            "own_i0": i0["8"],
        }.items():
            designs[name] = common(project(name, sources))
    elif family == "transitions":
        designs.update(
            candidate=common(feature("transition")),
            stale_transition=common(feature("transition_stale")),
            current_inputs=common(feature("raw")),
        )
    elif family == "cycle8":
        for name, key in {
            "candidate": "cycle8",
            "row_matched": "row_matched",
            "cycle7_matched": "cycle7_matched",
            "row_original": "row_original",
            "phase8": "phase8",
            "phase7": "phase7",
        }.items():
            designs[name] = common(feature(key))
    designs["global_affine"] = [
        [np.column_stack((np.ones(len(data["row"])), data["pred"][:, target])) for data in bases]
        for target in range(2)
    ]
    return designs, transforms


def compare(data: dict, predictions: dict, bootstrap: int) -> dict:
    """Report official WP and paired component intervals while retaining sequence counts."""
    if len(np.unique(data["component"])) < 4:
        raise ValueError("At least four evaluation components required")
    reports = {
        name: {"wp": reference_global_wp(data["y"], values, np.ones(len(values), bool))}
        for name, values in predictions.items()
    }
    sequences = np.unique(data["sequence"])
    moments: dict[str, np.ndarray] = {}
    components = []
    for name, prediction in predictions.items():
        accum = []
        for sequence in sequences:
            rows = data["sequence"] == sequence
            groups = np.unique(data["component"][rows])
            if len(groups) != 1:
                raise ValueError("A sequence belongs to multiple components")
            if name == "unchanged":
                components.append(groups[0])
            y, p = data["y"][rows][None], prediction[rows][None]
            value, _ = paired_component_moments(y, p, p, np.ones(y.shape[:2], bool))
            accum.append(value[0])
        moments[name] = np.stack(accum)
    for reference in predictions:
        if reference != "candidate":
            reports[f"candidate_minus_{reference}"] = paired_component_bootstrap(
                moments["candidate"],
                moments[reference],
                np.asarray(components),
                n_boot=bootstrap,
                seed=271828,
                centered=True,
            )
    return reports


def run(
    fit_path: Path,
    eval_path: Path,
    fit_history_path: Path,
    eval_history_path: Path,
    family: str,
    backend: str,
    output: Path,
    bootstrap: int = 2000,
) -> None:
    """Fit every fixed arm once; evaluation outcomes never choose transformations or arms."""
    if output.exists() or bootstrap < 100:
        raise ValueError("New output directory and at least100 bootstrap draws required")
    fit, fit_meta = load_cache(fit_path)
    evaluation, eval_meta = load_cache(eval_path)
    assert_outer_separation(fit, evaluation, fit_meta, eval_meta)
    if len(np.unique(fit["component"])) < 4:
        raise ValueError("At least four fitting components required")
    fit_history, fit_history_meta = load_histories(fit_history_path, fit, fit_meta)
    eval_history, eval_history_meta = load_histories(eval_history_path, evaluation, eval_meta)
    if fit_history_meta["router_sha256"] != eval_history_meta["router_sha256"] or any(
        set(meta["router_fit_components"]) & set(map(int, data["component"]))
        for meta in (fit_history_meta, eval_history_meta)
        for data in (fit, evaluation)
    ):
        raise ValueError("Router fit/evaluation component leakage or mismatched router")
    designs, transforms = build_designs(fit, evaluation, fit_history, eval_history, family)
    predictions, coefficients = {"unchanged": evaluation["pred"]}, {}
    widths = {}
    for name, targets in designs.items():
        prediction = evaluation["pred"].astype(np.float64).copy()
        widths[name] = [pair[0].shape[1] for pair in targets]
        for target, (train, test) in enumerate(targets):
            y = fit["y"][:, target].clip(-2, 2)
            weight = fit["fit_weight"] * np.abs(y)
            if weight.sum() <= 0:
                raise ValueError("Zero fitting target weight")
            weight = weight / weight.sum()
            matrix = train.T @ (train * weight[:, None]) + 0.01 * np.eye(train.shape[1])
            rhs = train.T @ (weight * (y - fit["pred"][:, target]))
            beta = solve(matrix, rhs, backend)
            coefficients[f"{name}__t{target}"] = beta
            prediction[:, target] += test @ beta
        predictions[name] = prediction
    reports = compare(evaluation, predictions, bootstrap)
    output.mkdir(parents=True)
    np.savez(
        output / "predictions.npz",
        **predictions,
        sequence=evaluation["sequence"],
        row=evaluation["row"],
        component=evaluation["component"],
    )
    np.savez(output / "coefficients.npz", **coefficients)
    np.savez(output / "transforms.npz", **transforms)
    (output / "report.json").write_text(
        json.dumps(
            {
                "version": 2,
                "family": family,
                "ridge": 0.01,
                "backend": backend,
                "fit_cache_sha256": fit_meta["sha256"],
                "eval_cache_sha256": eval_meta["sha256"],
                "fit_histories_sha256": fit_history_meta["sha256"],
                "eval_histories_sha256": eval_history_meta["sha256"],
                "router_sha256": fit_history_meta["router_sha256"],
                "head_widths_per_target": widths,
                "fit_components": len(np.unique(fit["component"])),
                "evaluation_components": len(np.unique(evaluation["component"])),
                "evaluation_sequences": len(np.unique(evaluation["sequence"])),
                "selection": "none; all arms fixed and reported; independent confirmation required",
                "bootstrap": bootstrap,
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Run a sealed exact-mechanism contrast on CPU, GPU, or separately admitted TPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--fit-histories", type=Path, required=True)
    parser.add_argument("--eval-histories", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--backend", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.fit,
        args.evaluation,
        args.fit_histories,
        args.eval_histories,
        args.family,
        args.backend,
        args.output,
    )


if __name__ == "__main__":
    main()
