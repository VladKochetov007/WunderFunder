"""Score the immutable E470 sixteen-arm export with paired component bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import METRIC_CLIP, METRIC_ID, _centered_moments, global_wp

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/sixteen_v2"
DEFAULT_OUTPUT = HERE / "eda/sixteen_v2"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
FIT_SPEC_PATH = HERE / "e470_activation_sixteen_fit_v2.sealed.json"
EXPECTED_COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 271_828
ACTIVATIONS = (
    "tanh",
    "gelu",
    "relu",
    "leaky_relu_001",
    "gelu_sine_01",
    "gelu_sine_cosine_01",
    "gelu_sinc_05",
    "split_gelu_leaky_relu",
)
OPTIMIZERS = ("adamw", "muon_adamw")
ARMS = tuple(f"{activation}__{optimizer}" for activation in ACTIVATIONS for optimizer in OPTIMIZERS)
PRIMARY_CONTRASTS = tuple(
    (f"{activation}__{optimizer}", f"tanh__{optimizer}", "activation_minus_tanh")
    for activation in ACTIVATIONS
    if activation != "tanh"
    for optimizer in OPTIMIZERS
) + tuple(
    (f"{activation}__muon_adamw", f"{activation}__adamw", "muon_minus_adamw_by_activation")
    for activation in ACTIVATIONS
)
SECONDARY_ACTIVATIONS = (
    "gelu_sine_01",
    "gelu_sine_cosine_01",
    "gelu_sinc_05",
    "split_gelu_leaky_relu",
)
SECONDARY_CONTRASTS = tuple(
    (f"{activation}__{optimizer}", f"gelu__{optimizer}", "secondary_activation_minus_gelu")
    for activation in SECONDARY_ACTIVATIONS
    for optimizer in OPTIMIZERS
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float32(values: np.ndarray, name: str) -> np.ndarray:
    """Convert an exported array and reject nonfinite raw values before clipping."""
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite raw values")
    return array


def sealed_data_identity() -> dict[str, int | str]:
    """Read the data artifact identity from the controller-sealed fit spec."""
    spec = json.loads(FIT_SPEC_PATH.read_text())
    matches = [item for item in spec["inputs"] if item["path"] == "datasets/valid.parquet"]
    if len(matches) != 1:
        raise ValueError("sealed fit spec must bind exactly one validation parquet input")
    identity = matches[0]
    if not DATA_PATH.is_file() or DATA_PATH.stat().st_size != identity["size"]:
        raise ValueError(
            "validation parquet is missing or differs in size from the sealed fit input"
        )
    return {"path": identity["path"], "sha256": identity["sha256"], "size": identity["size"]}


def load_nodes(run: Path, report: dict) -> tuple[list[dict], np.ndarray]:
    """Check source and exported cohort identities, then return ordered test nodes."""
    source_hash = sha256_file(COHORT_PATH)
    if source_hash != EXPECTED_COHORT_SHA256 or report.get("cohort_sha256") != source_hash:
        raise ValueError("run cohort identity differs from the frozen source cohort")
    copied = json.loads((run / "cohort.json").read_text())
    copied_nodes = copied.get("cohort")
    source_nodes = json.loads(COHORT_PATH.read_text())["cohort"]
    if copied_nodes != source_nodes:
        raise ValueError("run cohort nodes differ from the immutable source cohort")
    fit = [node for node in copied_nodes if node["role"] == "fit"]
    nodes = [node for node in copied_nodes if node["role"] == "test"]
    if len(fit) != 24 or len(nodes) != 16:
        raise ValueError("expected the fixed 24-fit/16-test exposed cohort")
    components = np.asarray([node["component"] for node in nodes])
    if len(np.unique(components)) != len(nodes):
        raise ValueError("test cohort must have one distinct component per sequence")
    if {node["component"] for node in fit} & set(components.tolist()):
        raise ValueError("fit and test component identities overlap")
    row_groups = [node["row_group"] for node in nodes]
    if len(set(row_groups)) != len(row_groups):
        raise ValueError("duplicate test row-group identity")
    return nodes, components


def verify_target_export(targets: np.ndarray, nodes: list[dict], counts: list[int]) -> None:
    """Verify selected targets and sequence row counts against the sealed parquet."""
    parquet = pq.ParquetFile(DATA_PATH)
    cursor = 0
    for node, count in zip(nodes, counts, strict=True):
        table = parquet.read_row_group(
            node["row_group"],
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")])
        try:
            raw = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("source targets are not numeric") from error
        mask = table["need_prediction"].to_numpy().astype(bool)
        mask &= table["is_scored"].to_numpy().astype(bool)
        if not np.isfinite(raw[mask]).all():
            raise ValueError(
                f"source selected labels are nonfinite in row group {node['row_group']}"
            )
        selected = np.clip(raw[mask], -METRIC_CLIP, METRIC_CLIP)
        if len(selected) != count or not np.array_equal(targets[cursor : cursor + count], selected):
            raise ValueError(
                f"target export does not align to source row group {node['row_group']}"
            )
        cursor += count
    if cursor != len(targets):
        raise ValueError("selected target count does not match test sequence segments")


def sequence_moments(targets: np.ndarray, predictions: np.ndarray, counts: list[int]) -> np.ndarray:
    """Build two-target centered moments while preserving one slot per component."""
    result = np.zeros((len(counts), 2, 6), dtype=np.float64)
    offset = 0
    for index, count in enumerate(counts):
        end = offset + count
        y = targets[offset:end]
        p = predictions[offset:end]
        # The raw arrays were checked before this clip; match the official metric.
        y = np.clip(y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(p, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        for target_index in range(2):
            result[index, target_index] = _centered_moments(y[:, target_index], p[:, target_index])
        offset = end
    if offset != len(targets):
        raise ValueError("sequence moment segments do not cover selected predictions")
    return result


def load_exports(
    run: Path, report: dict, nodes: list[dict]
) -> tuple[dict[str, dict], np.ndarray, dict[str, str]]:
    """Load all sixteen arms, checking run metadata and common selected targets."""
    if report.get("kind") != "fixed activation-by-optimizer screen":
        raise ValueError("unexpected run kind")
    if report.get("seed") != 271_828 or report.get("metric") != METRIC_ID:
        raise ValueError("run seed or metric differs from the fixed experiment")
    if len(report.get("arms", [])) != len(ARMS):
        raise ValueError("all sixteen fixed arms are required; partial result refused")
    sealed_spec = json.loads(FIT_SPEC_PATH.read_text())
    sealed_code = {entry["path"]: entry["sha256"] for entry in sealed_spec["code"]}
    expected_sources = {
        "fit_activation_optimizer.py": sealed_code[
            "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py"
        ],
        "tests.py": sealed_code["research/campaign_2026-10-05/activation_optimizer/tests.py"],
    }
    if report.get("source_sha256") != expected_sources:
        raise ValueError("run source hashes differ from the sealed sixteen-arm fit")
    local_sources = {name: sha256_file(HERE / name) for name in expected_sources}
    if local_sources != expected_sources:
        raise ValueError("local fit source no longer matches the sealed sixteen-arm run")

    by_name = {item.get("arm"): item for item in report["arms"]}
    if len(by_name) != len(ARMS) or set(by_name) != set(ARMS):
        raise ValueError("arm list has duplicates, omissions, or unregistered variants")
    node_groups = [node["row_group"] for node in nodes]
    loaded: dict[str, dict] = {}
    common_targets = None
    common_counts = None
    export_hashes = {}
    for arm in ARMS:
        item = by_name[arm]
        if item.get("parameters") != 8_235 or item.get("steps") != 474:
            raise ValueError(f"unexpected parameter count or update count for {arm}")
        sequences = item.get("sequences", [])
        if len(sequences) != len(nodes):
            raise ValueError(f"wrong sequence count for {arm}")
        if [sequence.get("row_group") for sequence in sequences] != node_groups:
            raise ValueError(f"sequence order differs from the cohort for {arm}")
        counts = [int(sequence.get("scored_rows", -1)) for sequence in sequences]
        if any(count <= 0 for count in counts) or sum(counts) != item.get("scored_rows"):
            raise ValueError(f"invalid selected-row counts for {arm}")
        if common_counts is None:
            common_counts = counts
        elif counts != common_counts:
            raise ValueError(f"selected-row segmentation differs across arms: {arm}")
        path = run / f"{arm}_selected_predictions.npz"
        if not path.is_file():
            raise ValueError(f"missing selected prediction export for {arm}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"targets", "predictions"}:
                raise ValueError(f"unexpected prediction archive schema for {arm}")
            targets = finite_float32(data["targets"], f"{arm} targets")
            predictions = finite_float32(data["predictions"], f"{arm} predictions")
        expected_shape = (sum(counts), 2)
        if targets.shape != expected_shape or predictions.shape != expected_shape:
            raise ValueError(f"wrong selected export shape for {arm}")
        if common_targets is None:
            common_targets = targets.copy()
            verify_target_export(common_targets, nodes, counts)
        elif not np.array_equal(targets, common_targets):
            raise ValueError(f"selected targets do not align across arms: {arm}")
        official = global_wp(
            targets[None, ...], predictions[None, ...], np.ones((1, len(targets)), dtype=bool)
        )
        if not np.isclose(official["wp"], item["wp"], atol=1e-7, rtol=0):
            raise ValueError(f"recorded WP differs from recomputed official metric for {arm}")
        if not np.allclose([official["t0"], official["t1"]], item["target_wp"], atol=1e-7, rtol=0):
            raise ValueError(f"recorded target WP differs from recomputed metric for {arm}")
        loaded[arm] = {
            "summary": item,
            "targets": targets,
            "predictions": predictions,
            "counts": counts,
            "moments": sequence_moments(targets, predictions, counts),
            "official": official,
            "row_groups": node_groups,
        }
        export_hashes[path.name] = sha256_file(path)
    if common_targets is None:
        raise ValueError("no target export loaded")
    return loaded, common_targets, export_hashes


def build_contrasts(loaded: dict[str, dict], components: np.ndarray) -> dict[str, dict]:
    """Compute all predeclared primary and outcome-motivated secondary contrasts."""
    contrasts = {}
    for candidate, reference, family in PRIMARY_CONTRASTS + SECONDARY_CONTRASTS:
        key = f"{candidate}_minus_{reference}"
        contrasts[key] = {
            "family": family,
            "candidate": candidate,
            "reference": reference,
            **paired_component_bootstrap(
                loaded[candidate]["moments"],
                loaded[reference]["moments"],
                components,
                n_boot=BOOTSTRAP_COUNT,
                seed=BOOTSTRAP_SEED,
                centered=True,
            ),
        }
    if len(PRIMARY_CONTRASTS) != 22 or len(SECONDARY_CONTRASTS) != 8 or len(contrasts) != 30:
        raise AssertionError("the fixed 22 primary and 8 secondary contrasts changed")
    return contrasts


def write_plots(loaded: dict[str, dict], contrasts: dict[str, dict], output: Path) -> None:
    """Write the full score heatmap and paired optimizer/activation contrast plots."""
    values = np.asarray(
        [
            [loaded[f"{activation}__{optimizer}"]["official"]["wp"] for optimizer in OPTIMIZERS]
            for activation in ACTIVATIONS
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(OPTIMIZERS)), labels=OPTIMIZERS)
    axis.set_yticks(np.arange(len(ACTIVATIONS)), labels=ACTIVATIONS)
    axis.set_title(
        "E470 fixed 16-arm screen: pooled clipped WP\n16 exposed test components; one fit seed"
    )
    axis.set_xlabel("Optimizer recipe (fixed LR 0.001; no family tuning claim)")
    for row in range(len(ACTIVATIONS)):
        for column in range(len(OPTIMIZERS)):
            axis.text(
                column, row, f"{values[row, column]:.4f}", ha="center", va="center", color="white"
            )
    figure.colorbar(image, ax=axis, label=METRIC_ID)
    figure.tight_layout()
    figure.savefig(output / "activation_optimizer_wp_heatmap.png", dpi=180)
    plt.close(figure)

    optimizer_keys = [
        f"{activation}__muon_adamw_minus_{activation}__adamw" for activation in ACTIVATIONS
    ]
    optimizer_effects = [contrasts[key] for key in optimizer_keys]
    centers = np.asarray([item["mean_diff"] for item in optimizer_effects])
    bounds = np.asarray([item["ci95"] for item in optimizer_effects])
    figure, axis = plt.subplots(figsize=(8, 4.8))
    y_positions = np.arange(len(ACTIVATIONS))
    for center, (lower, upper), y_position in zip(centers, bounds, y_positions, strict=True):
        axis.hlines(y_position, lower, upper, color="#335f9e", linewidth=1.5)
        axis.plot(center, y_position, "o", color="#335f9e")
    axis.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    axis.set_yticks(y_positions, labels=ACTIVATIONS)
    axis.invert_yaxis()
    axis.set_xlabel("MuonAdamW − AdamW pooled WP (paired component bootstrap 95% CI)")
    axis.set_title("Optimizer contrast by activation\nconditional on fixed LR 0.001 and one seed")
    figure.tight_layout()
    figure.savefig(output / "optimizer_contrasts_by_activation.png", dpi=180)
    plt.close(figure)

    activation_keys = [
        f"{activation}__{optimizer}_minus_tanh__{optimizer}"
        for activation in ACTIVATIONS
        if activation != "tanh"
        for optimizer in OPTIMIZERS
    ] + [
        f"{activation}__{optimizer}_minus_gelu__{optimizer}"
        for activation in SECONDARY_ACTIVATIONS
        for optimizer in OPTIMIZERS
    ]
    effects = [contrasts[key] for key in activation_keys]
    centers = np.asarray([item["mean_diff"] for item in effects])
    bounds = np.asarray([item["ci95"] for item in effects])
    labels = [key.replace("__", " / ").replace("_minus_", " − ") for key in activation_keys]
    figure, axis = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(len(labels))
    colors = ["#3973ac"] * 14 + ["#d47b32"] * 8
    for center, (lower, upper), y_position, color in zip(
        centers, bounds, y_positions, colors, strict=True
    ):
        axis.hlines(y_position, lower, upper, color=color, linewidth=1.3)
        axis.plot(center, y_position, "o", color=color)
    axis.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    axis.set_yticks(y_positions, labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("Paired component-bootstrap difference in pooled WP (95% CI)")
    axis.set_title(
        "Primary activation contrasts (blue) and secondary GELU-family controls (orange)"
    )
    figure.tight_layout()
    figure.savefig(output / "activation_contrasts.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Validate all sealed exports, recompute scores and write paired contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {output}")
    report = json.loads((run / "result.json").read_text())
    nodes, components = load_nodes(run, report)
    data_identity = sealed_data_identity()
    loaded, targets, export_hashes = load_exports(run, report, nodes)
    contrasts = build_contrasts(loaded, components)
    output.mkdir(parents=True)
    result = {
        "metric": METRIC_ID,
        "seed": report["seed"],
        "scope": (
            "one fixed fit seed and 16 historically exposed diagnostic components; "
            "conditional exploratory contrasts, not fresh validation, training-seed "
            "uncertainty, or public-score evidence"
        ),
        "fit_recipe": "all 16 registered activation/optimizer arms; fixed LR 0.001; 474 updates each",
        "component_ids": components.tolist(),
        "row_groups": [node["row_group"] for node in nodes],
        "scored_rows": int(len(targets)),
        "arm_scores": {
            arm: {
                "wp": loaded[arm]["official"]["wp"],
                "t0": loaded[arm]["official"]["t0"],
                "t1": loaded[arm]["official"]["t1"],
                "scored_rows": loaded[arm]["official"]["selected_rows"],
                "parameters": loaded[arm]["summary"]["parameters"],
                "steps": loaded[arm]["summary"]["steps"],
            }
            for arm in ARMS
        },
        "primary_contrast_count": len(PRIMARY_CONTRASTS),
        "primary_contrasts": {
            key: value
            for key, value in contrasts.items()
            if value["family"] != "secondary_activation_minus_gelu"
        },
        "secondary_contrast_count": len(SECONDARY_CONTRASTS),
        "secondary_contrasts": {
            key: value
            for key, value in contrasts.items()
            if value["family"] == "secondary_activation_minus_gelu"
        },
        "bootstrap": {
            "method": "wnn.evaluation.paired_component_bootstrap(centered=True)",
            "components_resampled_as_units": True,
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "multiple_comparisons": "30 exploratory contrasts reported without pruning or multiplicity correction",
        },
        "data_input_identity_from_sealed_fit_spec": data_identity,
        "cohort_sha256": report["cohort_sha256"],
        "fit_source_sha256": report["source_sha256"],
        "fit_result_sha256": sha256_file(run / "result.json"),
        "prediction_export_sha256": export_hashes,
        "scorer_sha256": sha256_file(Path(__file__)),
    }
    (output / "paired_components.json").write_text(json.dumps(result, indent=2) + "\n")
    write_plots(loaded, contrasts, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "arms": len(loaded),
                "primary": 22,
                "secondary": 8,
                "rows": len(targets),
            }
        )
    )


if __name__ == "__main__":
    main()
