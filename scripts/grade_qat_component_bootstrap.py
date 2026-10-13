from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

EXPECTED_MANIFEST_SHA256 = "8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556"
OFFICIAL_METRIC_ID = "global_wp_clip2_v1"
OFFICIAL_METRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BATCH = 128
EXPECTED_GROUPS = 468
EXPECTED_COMPONENTS = 141
MAX_JSON_BYTES = 2_000_000
MAX_NPZ_UNCOMPRESSED_BYTES = 2_000_000
SCORE_TOLERANCE = 1e-10
TARGET_STAT_COLUMNS = (0, 1, 3)  # W, WY, WYY depend only on labels and selection.

# JSON records and NumPy archive members are dynamic until validated at the boundary.
JsonObject = dict[str, Any]
ScoreFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One validated deployment output and its component-level metric sums."""

    path: Path
    result: JsonObject
    component_ids: np.ndarray
    sums: np.ndarray
    result_sha256: str
    component_sums_sha256: str
    manifest_path: Path
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so endpoint validation has bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    """Read a size-bounded JSON object from a frozen endpoint or manifest."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected regular JSON file: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON file exceeds {MAX_JSON_BYTES} bytes: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_metric(project_root: Path) -> tuple[ScoreFunction, str]:
    """Load the pinned official metric implementation after checking its bytes."""
    metric_path = project_root / "src/wnn/gmetric.py"
    actual_sha256 = sha256_file(metric_path)
    if actual_sha256 != OFFICIAL_METRIC_SHA256:
        raise ValueError("official metric source hash differs from the reviewed version")
    spec = importlib.util.spec_from_file_location("qat_review_gmetric", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official metric module from {metric_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "METRIC_ID", None) != OFFICIAL_METRIC_ID:
        raise ValueError("official metric identifier differs")
    function = getattr(module, "wp_from_sums", None)
    if not callable(function):
        raise AttributeError("official metric lacks wp_from_sums")
    return function, actual_sha256


def manifest_components(manifest: JsonObject) -> tuple[list[int], list[int]]:
    """Return the frozen evaluation order and its sorted component identities."""
    groups_raw = manifest.get("evaluation_groups")
    component_map = manifest.get("components")
    if not isinstance(groups_raw, list) or not isinstance(component_map, dict):
        raise ValueError("manifest lacks evaluation groups or component mapping")
    groups = [int(group) for group in groups_raw]
    if len(groups) != EXPECTED_GROUPS or len(set(groups)) != EXPECTED_GROUPS:
        raise ValueError("manifest is not the expected unique 468-group fold")
    try:
        component_ids = sorted({int(component_map[str(group)]) for group in groups})
    except KeyError as error:
        raise ValueError(f"manifest lacks component mapping for group {error}") from error
    if len(component_ids) != EXPECTED_COMPONENTS:
        raise ValueError("manifest does not yield the expected 141 components")
    return groups, component_ids


def load_manifest(path: Path, expected_sha256: str) -> JsonObject:
    """Read a hash-pinned manifest and require the fixed F0 metric and fold."""
    manifest = read_json(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"deployment manifest hash differs: {path}")
    if manifest.get("metric") != OFFICIAL_METRIC_ID or manifest.get("fold") != 0:
        raise ValueError("manifest metric or fold differs from the fixed F0 grade")
    manifest_components(manifest)
    return manifest


def verify_manifest_sources(manifest: JsonObject) -> None:
    """Check every small frozen source file named by this manifest."""
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("manifest has no frozen source-file hashes")
    for raw_path, expected in source_files.items():
        path = Path(raw_path)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"frozen evaluator source hash differs: {path}")


def component_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only bounded, numeric component IDs and official six-sum arrays."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected regular component-sums archive: {path}")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    if len(entries) != 2 or {entry.filename for entry in entries} != {
        "components.npy",
        "sums.npy",
    }:
        raise ValueError("component archive must contain only components.npy and sums.npy")
    if sum(entry.file_size for entry in entries) > MAX_NPZ_UNCOMPRESSED_BYTES:
        raise ValueError("component archive exceeds the uncompressed size bound")
    with np.load(path, allow_pickle=False) as archive:
        component_ids = np.asarray(archive["components"])
        sums = np.asarray(archive["sums"])
    if component_ids.dtype.kind not in "iu" or component_ids.ndim != 1:
        raise ValueError("component identities must be a one-dimensional integer array")
    if sums.dtype != np.float64 or sums.ndim != 3 or sums.shape[1:] != (2, 6):
        raise ValueError("component sums must be float64 with shape (N, 2, 6)")
    if not np.isfinite(sums).all():
        raise ValueError("component sums contain nonfinite values")
    return component_ids.astype(np.int64, copy=False), sums


def endpoint_manifest_path(endpoint_path: Path, result: JsonObject, fallback: Path) -> Path:
    """Resolve the endpoint's adjacent manifest or use the pinned baseline one."""
    adjacent = endpoint_path.parent / "manifest.json"
    if adjacent.is_file():
        return adjacent
    if result.get("manifest_sha256") == sha256_file(fallback):
        return fallback
    raise FileNotFoundError(f"endpoint has no adjacent manifest: {endpoint_path}")


def validate_endpoint(
    endpoint_path: Path,
    expected_groups: list[int],
    expected_components: list[int],
    anchor: Endpoint | None = None,
    fallback_manifest: Path | None = None,
) -> Endpoint:
    """Validate one complete endpoint against its own and the anchor's identity."""
    result_path = endpoint_path / "result.json"
    sums_path = endpoint_path / "component_sums.npz"
    result = read_json(result_path)
    if result.get("profile_only") is not False:
        raise ValueError(f"endpoint is not a full-fold grade: {endpoint_path}")
    if result.get("full_sequence_replay_exact") is not True:
        raise ValueError(f"endpoint lacks exact sequence replay: {endpoint_path}")
    if result.get("scope") != "exposed development fold, exact parent ancestry audited separately":
        raise ValueError(f"endpoint scope differs: {endpoint_path}")
    if result.get("groups") != expected_groups:
        raise ValueError(f"endpoint evaluation group order differs: {endpoint_path}")
    manifest_sha256 = result.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError(f"endpoint manifest hash is missing: {endpoint_path}")
    if fallback_manifest is None:
        raise ValueError("a fallback manifest is required to validate the baseline")
    manifest_path = endpoint_manifest_path(endpoint_path, result, fallback_manifest)
    manifest = load_manifest(manifest_path, manifest_sha256)
    groups, component_ids_expected = manifest_components(manifest)
    if groups != expected_groups or component_ids_expected != expected_components:
        raise ValueError(f"endpoint manifest fold membership differs: {endpoint_path}")
    verify_manifest_sources(manifest)
    component_ids, sums = component_archive(sums_path)
    if not np.array_equal(component_ids, np.asarray(expected_components, dtype=np.int64)):
        raise ValueError(f"endpoint component IDs differ: {endpoint_path}")
    if sums.shape != (EXPECTED_COMPONENTS, 2, 6):
        raise ValueError(f"endpoint component count differs: {endpoint_path}")
    if result.get("components") != EXPECTED_COMPONENTS:
        raise ValueError(f"endpoint result component count differs: {endpoint_path}")
    score = result.get("score")
    if not isinstance(score, dict) or score.get("metric_id") != OFFICIAL_METRIC_ID:
        raise ValueError(f"endpoint score does not use the official metric: {endpoint_path}")
    selected_rows = score.get("selected_rows")
    if not isinstance(selected_rows, int) or selected_rows <= 0:
        raise ValueError(f"endpoint selected-row count is invalid: {endpoint_path}")
    if anchor is not None:
        anchor_score = anchor.result["score"]
        if selected_rows != anchor_score["selected_rows"]:
            raise ValueError(f"endpoint selected-row mask count differs: {endpoint_path}")
        target_stats = np.take(sums, TARGET_STAT_COLUMNS, axis=2)
        anchor_target_stats = np.take(anchor.sums, TARGET_STAT_COLUMNS, axis=2)
        if not np.array_equal(target_stats, anchor_target_stats):
            raise ValueError(f"endpoint component target/mask signatures differ: {endpoint_path}")
    return Endpoint(
        path=endpoint_path,
        result=result,
        component_ids=component_ids,
        sums=sums,
        result_sha256=sha256_file(result_path),
        component_sums_sha256=sha256_file(sums_path),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def score_sums(sums: np.ndarray, wp_from_sums: ScoreFunction) -> np.ndarray:
    """Compute t0, t1, and their official equal-weight mean from raw sums."""
    rho = np.asarray(wp_from_sums(sums), dtype=np.float64)
    if rho.shape[-1] != 2:
        raise ValueError("official metric returned an unexpected target-score shape")
    return np.stack((rho[..., 0], rho[..., 1], rho.mean(axis=-1)), axis=-1)


def endpoint_score(endpoint: Endpoint, wp_from_sums: ScoreFunction) -> np.ndarray:
    """Score all components pooled, then compare with the stored result receipt."""
    observed = score_sums(endpoint.sums.sum(axis=0), wp_from_sums)
    stored = endpoint.result["score"]
    expected = np.asarray([stored["t0"], stored["t1"], stored["wp"]], dtype=np.float64)
    if not np.allclose(observed, expected, rtol=0.0, atol=SCORE_TOLERANCE):
        raise ValueError(
            f"official component sums disagree with result.json for {endpoint.path}: "
            f"computed={observed.tolist()} stored={expected.tolist()}"
        )
    if abs(float(observed[2]) - 0.5 * (float(observed[0]) + float(observed[1]))) > 1e-15:
        raise ValueError("pooled WP is not the equal mean of t0 and t1")
    return observed


def paired_bootstrap_delta(
    reference: Endpoint,
    candidate: Endpoint,
    wp_from_sums: ScoreFunction,
) -> np.ndarray:
    """Resample shared components with replacement and return paired score deltas."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(reference.component_ids)
    deltas = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH):
        stop = min(start + BOOTSTRAP_BATCH, BOOTSTRAP_REPLICATES)
        draws = generator.integers(0, count, size=(stop - start, count))
        reference_sums = np.take(reference.sums, draws, axis=0).sum(axis=1)
        candidate_sums = np.take(candidate.sums, draws, axis=0).sum(axis=1)
        deltas[start:stop] = score_sums(candidate_sums, wp_from_sums) - score_sums(
            reference_sums, wp_from_sums
        )
    if not np.isfinite(deltas).all():
        raise ValueError("bootstrap produced nonfinite paired metric deltas")
    return deltas


def summarize_pair(
    name: str,
    reference: Endpoint,
    candidate: Endpoint,
    wp_from_sums: ScoreFunction,
) -> JsonObject:
    """Return full-fold paired point deltas and deterministic bootstrap intervals."""
    if (
        reference.result_sha256 == candidate.result_sha256
        and reference.component_sums_sha256 == candidate.component_sums_sha256
    ):
        raise ValueError(f"paired endpoints are byte-identical: {name}")
    reference_score = endpoint_score(reference, wp_from_sums)
    candidate_score = endpoint_score(candidate, wp_from_sums)
    deltas = paired_bootstrap_delta(reference, candidate, wp_from_sums)
    metrics: JsonObject = {}
    for index, metric in enumerate(("t0", "t1", "wp")):
        low, median, high = np.quantile(deltas[:, index], (0.025, 0.5, 0.975))
        metrics[metric] = {
            "reference": float(reference_score[index]),
            "candidate": float(candidate_score[index]),
            "candidate_minus_reference": float(candidate_score[index] - reference_score[index]),
            "bootstrap_q025": float(low),
            "bootstrap_median": float(median),
            "bootstrap_q975": float(high),
        }
    return {
        "name": name,
        "reference_endpoint": str(reference.path),
        "reference_case": reference.result["case"],
        "reference_result_sha256": reference.result_sha256,
        "reference_component_sums_sha256": reference.component_sums_sha256,
        "candidate_endpoint": str(candidate.path),
        "candidate_case": candidate.result["case"],
        "candidate_result_sha256": candidate.result_sha256,
        "candidate_component_sums_sha256": candidate.component_sums_sha256,
        "reference_manifest_sha256": reference.manifest_sha256,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "paired_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse the pinned baseline and optional endpoint-pair inputs."""
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/qat_renewable_r1/deployment_frozen_r2/manifest.json"),
    )
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/qat_renewable_r1/deployment_frozen_r2/native_q8"),
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "REFERENCE_DIR", "CANDIDATE_DIR"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    """Resolve a CLI path under the declared repository root when relative."""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    """Validate the native-Q8 baseline and grade any matched endpoint pairs."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve(repo_root, args.manifest)
    baseline_path = _resolve(repo_root, args.baseline)
    output_path = _resolve(repo_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite grade receipt: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"receipt directory does not exist: {output_path.parent}")
    manifest = load_manifest(manifest_path, args.expected_manifest_sha256)
    verify_manifest_sources(manifest)
    groups, component_ids = manifest_components(manifest)
    baseline = validate_endpoint(
        baseline_path,
        groups,
        component_ids,
        fallback_manifest=manifest_path,
    )
    if baseline.manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("native-Q8 anchor does not use the admitted R2 manifest")
    if baseline.result.get("case") != "native_q8":
        raise ValueError("fixed anchor endpoint is not native_q8")
    if baseline.result["score"].get("selected_rows") != 808_863:
        raise ValueError("native-Q8 anchor selected-row count differs from the R2 receipt")
    stored_score = baseline.result["score"]
    wp_from_sums, metric_sha256 = load_metric(repo_root)
    official_score = endpoint_score(baseline, wp_from_sums)
    pairs: list[JsonObject] = []
    seen_names: set[str] = set()
    for name, reference_arg, candidate_arg in args.pair:
        if not name or name in seen_names:
            raise ValueError(f"paired grade name is empty or duplicated: {name!r}")
        seen_names.add(name)
        reference_path = _resolve(repo_root, Path(reference_arg))
        candidate_path = _resolve(repo_root, Path(candidate_arg))
        reference = validate_endpoint(
            reference_path,
            groups,
            component_ids,
            anchor=baseline,
            fallback_manifest=manifest_path,
        )
        candidate = validate_endpoint(
            candidate_path,
            groups,
            component_ids,
            anchor=baseline,
            fallback_manifest=manifest_path,
        )
        pairs.append(summarize_pair(name, reference, candidate, wp_from_sums))
    receipt = {
        "schema": "wnn.qat-component-bootstrap-grade.v1",
        "status": "baseline_verified" if not pairs else "graded",
        "grader_script_sha256": sha256_file(Path(__file__)),
        "metric_id": OFFICIAL_METRIC_ID,
        "metric_source_sha256": metric_sha256,
        "deployment_manifest_sha256": baseline.manifest_sha256,
        "manifest_path": str(manifest_path),
        "fold": 0,
        "groups": EXPECTED_GROUPS,
        "components": EXPECTED_COMPONENTS,
        "selected_rows": int(stored_score["selected_rows"]),
        "baseline_case": "native_q8",
        "baseline_result_sha256": baseline.result_sha256,
        "baseline_component_sums_sha256": baseline.component_sums_sha256,
        "baseline_score_stored": {key: float(stored_score[key]) for key in ("t0", "t1", "wp")},
        "baseline_score_recomputed_from_component_sums": {
            key: float(official_score[index]) for index, key in enumerate(("t0", "t1", "wp"))
        },
        "bootstrap": {
            "unit": "source-qualified connected component, equal component sampling",
            "paired": True,
            "sum_scoring_function": "wnn.gmetric.wp_from_sums",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "quantiles": [0.025, 0.5, 0.975],
            "quantile_method": "linear",
            "batch_size": BOOTSTRAP_BATCH,
            "interval_interpretation": "descriptive raw-six-sum uncertainty; not an automatic stop or kill rule",
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "mask_validation": "full ordered groups and components plus exact per-component target-only sums and total selected rows",
        "pairs": pairs,
        "limitations": [
            "component_sums.npz stores six legacy raw sums, not the centered moments used by GlobalAccumulator.result; bootstrap intervals therefore apply wp_from_sums to resampled raw sums and are an approximation to the exact centered-moment bootstrap",
            "F0 is an exposed development fold, not a pristine holdout.",
            "All endpoints use one fitting seed; this is not cross-seed confirmation.",
            "No public score or transfer forecast is implied.",
            "Component sums preserve mask/target fingerprints but not row-level mask hashes.",
        ],
    }
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
