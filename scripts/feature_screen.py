from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from feature_cues import CUE_NAMES

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import reference_global_wp


def digest_file(path: Path) -> str:
    """Hash cached arrays before fitting a fixed experiment."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Require component/model provenance; reject non-OOF state rows."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"] != digest_file(path):
        raise ValueError("Cache hash mismatch")
    if not 0 < metadata["rows"] <= 40_000 or not 0 < metadata["state_width"] <= 1024:
        raise ValueError("Cache exceeds the bounded screening shape")
    with np.load(path, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    required = {"state", "pred", "y", "fit_weight", "component", "sequence", "row", "model_id"}
    if not required.issubset(data) or len(data["component"]) != metadata["rows"]:
        raise ValueError("Cache schema or row count mismatch")
    count = metadata["rows"]
    for name, value in data.items():
        if len(value) != count or value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Invalid cache array: {name}")
    if (
        data["pred"].shape != (count, 2)
        or data["y"].shape != (count, 2)
        or np.any(data["row"] < 99)
    ):
        raise ValueError("Invalid target/prediction/warmup rows")
    if data["state"].shape != (count, metadata["state_width"]):
        raise ValueError("Hidden-state width mismatch")
    if data["fit_weight"].shape != (count,) or np.any(data["fit_weight"] < 0):
        raise ValueError("Invalid fixed fitting weights")
    if len(np.unique(data["model_id"])) != 1:
        raise ValueError("Do not pool unaligned hidden coordinates from different backbones")
    for model_id in np.unique(data["model_id"]):
        binding = metadata["members"][str(model_id)]
        rows = data["model_id"] == model_id
        if set(map(int, data["component"][rows])) & set(binding["fit_components"]):
            raise ValueError("Frozen backbone fitted a cached row's component")
        if str(model_id) != binding["checkpoint_sha256"]:
            raise ValueError("Backbone checkpoint binding mismatch")
    return data, metadata


def assert_outer_separation(fit: dict, evaluation: dict, fit_meta: dict, eval_meta: dict) -> None:
    """Outer evaluation components may not occur in any inner backbone training."""
    fit_components = set(map(int, fit["component"]))
    eval_components = set(map(int, evaluation["component"]))
    if fit_components & eval_components:
        raise ValueError("Fit/evaluation component overlap")
    if fit_meta["role"] != "fit" or eval_meta["role"] != "evaluation":
        raise ValueError("Cache roles disagree with the prespecified head contrast")
    if fit_meta["outer_fold"] != eval_meta["outer_fold"]:
        raise ValueError("Cache outer folds disagree")
    if fit_meta["component_graph_sha256"] != eval_meta["component_graph_sha256"] or set(
        fit_meta["members"]
    ) != set(eval_meta["members"]):
        raise ValueError("State coordinates or component graph differ between caches")
    for meta in (fit_meta, eval_meta):
        for binding in meta["members"].values():
            if eval_components & set(binding["fit_components"]):
                raise ValueError("An inner backbone fitted protected outer components")


def projection_pair(
    fit: np.ndarray,
    evaluation: np.ndarray,
    width: int,
    seed: int,
    parameters: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit normalization only on fitting rows; use a fixed label-free nonlinear map."""
    mean = fit.mean(0, dtype=np.float64)
    scale = np.maximum(fit.std(0, dtype=np.float64), 1e-4)
    projection = np.random.default_rng(seed).normal(size=(fit.shape[1], width)) / np.sqrt(
        fit.shape[1]
    )
    if parameters is not None:
        parameters.update(mean=mean, scale=scale, projection=projection)
    return tuple(
        np.tanh(np.clip((array - mean) / scale, -8, 8) @ projection) for array in (fit, evaluation)
    )


def solve(matrix: np.ndarray, rhs: np.ndarray, backend: str) -> np.ndarray:
    """Use the requested solver explicitly; no silent accelerator fallback."""
    if backend == "numpy":
        return np.linalg.solve(matrix, rhs)
    if backend == "torch":
        import torch

        from wnn.resources import configure_torch

        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return (
            torch.linalg.solve(torch.from_numpy(matrix).cuda(), torch.from_numpy(rhs).cuda())
            .cpu()
            .numpy()
        )
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.devices() if device.platform == "tpu"]
        if not devices:
            raise RuntimeError("TPU requested but unavailable")
        # FP32 is explicit for TPU; a matched CPU FP32 numerical pilot is required.
        with jax.default_device(devices[0]):
            return np.asarray(
                jnp.linalg.solve(
                    jnp.asarray(matrix, dtype=jnp.float32), jnp.asarray(rhs, dtype=jnp.float32)
                )
            )
    raise ValueError(f"Unsupported solver: {backend}")


def fit_residual(
    features: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    base_weight: np.ndarray,
    ridge: float,
    backend: str,
) -> np.ndarray:
    """Fit one fixed ridge residual, using clipped target-magnitude weights."""
    coefficients = []
    for index in range(2):
        y = target[:, index].clip(-2, 2)
        weights = base_weight * np.abs(y)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Zero fitting target weight")
        gram = features.T @ (features * weights[:, None]) / total
        gram += ridge * np.eye(features.shape[1])
        rhs = features.T @ (weights * (y - prediction[:, index])) / total
        coefficients.append(solve(gram, rhs, backend))
    return np.stack(coefficients, -1)


def compare_components(data: dict, predictions: dict[str, np.ndarray], bootstrap: int) -> dict:
    """Use identical connected-component draws for all fixed arm contrasts."""
    components = np.unique(data["component"])
    if len(components) < 4:
        raise ValueError("At least four evaluation components required")
    reports = {
        name: {
            "wp": reference_global_wp(data["y"], prediction, np.ones(len(prediction), dtype=bool))
        }
        for name, prediction in predictions.items()
    }
    for name in ("candidate", "placebo", "capacity"):
        for reference in ("unchanged", "capacity"):
            if name == reference:
                continue
            candidate_moments, reference_moments = [], []
            for component in components:
                selected = data["component"] == component
                y = data["y"][selected][None]
                a, b = paired_component_moments(
                    y,
                    predictions[name][selected][None],
                    predictions[reference][selected][None],
                    np.ones(y.shape[:2], dtype=bool),
                )
                candidate_moments.append(a[0])
                reference_moments.append(b[0])
            reports[f"{name}_minus_{reference}"] = paired_component_bootstrap(
                np.stack(candidate_moments),
                np.stack(reference_moments),
                components,
                n_boot=bootstrap,
                seed=271828,
                centered=True,
            )
    return reports


def run(
    fit_path: Path,
    eval_path: Path,
    cue: str,
    backend: str,
    ridge: float,
    output: Path,
    bootstrap: int = 2000,
) -> None:
    """Fit all prespecified matched heads once; report every arm without selecting one."""
    if output.exists() or ridge <= 0 or bootstrap < 100:
        raise ValueError("New output directory, positive fixed ridge and >=100 draws required")
    fit, fit_meta = load_cache(fit_path)
    evaluation, eval_meta = load_cache(eval_path)
    assert_outer_separation(fit, evaluation, fit_meta, eval_meta)
    if cue not in CUE_NAMES:
        raise ValueError("Cue must be a predefined family")
    baseline_parameters = {}
    baseline = projection_pair(fit["state"], evaluation["state"], 64, 271829, baseline_parameters)
    transforms = {f"baseline__{key}": value for key, value in baseline_parameters.items()}
    sources = {
        "candidate": (fit[f"cue__{cue}"], evaluation[f"cue__{cue}"]),
        "placebo": (fit[f"placebo__{cue}"], evaluation[f"placebo__{cue}"]),
        "capacity": (fit["state"], evaluation["state"]),
    }
    predictions = {"unchanged": evaluation["pred"]}
    output.mkdir(parents=True)
    coefficients = {}
    for name, source in sources.items():
        parameters = {}
        extra = projection_pair(*source, 32, 271830, parameters)
        transforms.update({f"{name}__{key}": value for key, value in parameters.items()})
        if cue == "regime_heads":
            extra = tuple(a * b[:, :32] for a, b in zip(extra, baseline, strict=True))
        design = [
            np.concatenate((np.ones((len(values), 1)), cache["pred"], values, addition), -1)
            for values, addition, cache in zip(baseline, extra, (fit, evaluation), strict=True)
        ]
        beta = fit_residual(design[0], fit["y"], fit["pred"], fit["fit_weight"], ridge, backend)
        coefficients[name] = beta
        predictions[name] = evaluation["pred"] + design[1] @ beta
    reports = compare_components(evaluation, predictions, bootstrap)
    np.savez(
        output / "predictions.npz",
        **predictions,
        component=evaluation["component"],
        row=evaluation["row"],
        sequence=evaluation["sequence"],
    )
    np.savez(output / "coefficients.npz", **coefficients)
    np.savez(output / "transforms.npz", **transforms)
    report = {
        "cue": cue,
        "ridge": ridge,
        "backend": backend,
        "fit_cache_sha256": fit_meta["sha256"],
        "eval_cache_sha256": eval_meta["sha256"],
        "reports": reports,
        "bootstrap": bootstrap,
        "selection": "none; all arms reported; independent outer confirmation required",
        "map": {
            "baseline_width": 64,
            "extra_width": 32,
            "baseline_seed": 271829,
            "extra_seed": 271830,
            "normalization": "fit-only mean/std, floor1e-4, clip8",
            "regime_heads": "matched multiplicative state interaction",
        },
        "scope": "exact pooled score on frozen selected rows; not full-validation, end-to-end representation or public gain",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Run a controller-sealed CPU/GPU/TPU fixed-feature experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cue", choices=CUE_NAMES, required=True)
    parser.add_argument("--backend", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.evaluation, args.cue, args.backend, args.ridge, args.output)


if __name__ == "__main__":
    main()
