"""Grade three quota ZIP replays only after their cross-run gate is frozen."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any  # JSON metadata and lazy NumPy arrays have runtime-defined shapes.

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FEASIBILITY = ROOT / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout"
ONLINE = ROOT / "research/experiment_drafts/online-relationships-20261022"
SELECTION_SHA256 = "48c017d13884d8294cdb21f19cef6be1df4663fa4fc54eb592e651a21bcaff07"
REPLAY_PLAN_SHA256 = "1f49103e708ea13695a55c760083b61283a12ff65ca2cab4cdbd36787079bb6d"
EXPECTED_CANDIDATES = ("baseline", "s3", "s3_s4")
PREDICTION_SHAPE = [64, 20_000, 2]
PREDICTION_DTYPE = "<f4"
ROWS = 20_000
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271_828


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load bounded JSON metadata and require an object root."""
    if path.stat().st_size > 5_000_000:
        raise ValueError(f"metadata file exceeds 5 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve_path(value: str) -> Path:
    """Resolve absolute and repository-relative recipe paths."""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def npy_header(path: Path) -> dict[str, Any]:
    """Read only an NPY header and verify exact byte coverage without loading values."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"invalid NPY magic: {path}")
        version = (prefix[6], prefix[7])
        if version == (1, 0):
            length_bytes = stream.read(2)
            length_format = "<H"
            header_encoding = "latin1"
        elif version in ((2, 0), (3, 0)):
            length_bytes = stream.read(4)
            length_format = "<I"
            header_encoding = "utf8" if version == (3, 0) else "latin1"
        else:
            raise ValueError(f"unsupported NPY version {version}: {path}")
        if len(length_bytes) != struct.calcsize(length_format):
            raise ValueError(f"truncated NPY header length: {path}")
        header_length = struct.unpack(length_format, length_bytes)[0]
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"truncated NPY header: {path}")
        header_bytes = stream.tell()
    metadata = ast.literal_eval(header.decode(header_encoding).strip())
    shape = metadata.get("shape")
    dtype = metadata.get("descr")
    if not isinstance(shape, tuple) or not isinstance(dtype, str):
        raise ValueError(f"malformed NPY header: {path}")
    if metadata.get("fortran_order") is not False:
        raise ValueError(f"expected C-order NPY data: {path}")
    dtype_sizes = {"<f4": 4, "=f4": 4, "|u1": 1, "|b1": 1}
    if dtype not in dtype_sizes:
        raise ValueError(f"unsupported NPY dtype {dtype}: {path}")
    payload_bytes = math.prod(shape) * dtype_sizes[dtype]
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + payload_bytes:
        raise ValueError(f"NPY file length differs from header coverage: {path}")
    return {
        "shape": list(shape),
        "dtype": dtype,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "file_bytes": file_bytes,
    }


def verify_source_pins(recipe: dict[str, Any]) -> dict[str, str]:
    """Verify frozen scoring, metric, extraction, and replay implementation sources."""
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in recipe["source_sha256"].items():
        path = resolve_path(relative_path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"grade source changed: {relative_path}")
        observed[relative_path] = actual_sha256
    return dict(sorted(observed.items()))


def verify_plan_and_selection(recipe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify exact selection, admitted replay plan, split pins, and source hashes."""
    selection_record, plan_record = recipe["selection"], recipe["replay_plan"]
    selection_path = resolve_path(selection_record["path"])
    plan_path = resolve_path(plan_record["path"])
    if (
        sha256_file(selection_path) != selection_record["sha256"]
        or selection_record["sha256"] != SELECTION_SHA256
    ):
        raise ValueError("selection source differs from its frozen SHA-256")
    if (
        sha256_file(plan_path) != plan_record["sha256"]
        or plan_record["sha256"] != REPLAY_PLAN_SHA256
    ):
        raise ValueError("replay plan differs from its admitted SHA-256")
    selection = read_json(selection_path)
    plan = read_json(plan_path)
    if resolve_path(plan.get("selection_path", selection_record["path"])) != selection_path:
        raise ValueError("admitted replay plan points to a different selection file")
    if plan.get("schema") != "wnn.quota-native-replay-plan.v1" or plan.get("status") != "admitted":
        raise ValueError("quota replay plan is not admitted")
    if plan.get("selection_sha256") != selection_record["sha256"]:
        raise ValueError("replay plan does not bind the frozen selection")
    if selection.get("status") != "frozen_before_any_candidate_replay_or_quality_read":
        raise ValueError("quota selection does not carry its frozen-before-replay status")
    if selection.get("split_manifests") != plan.get("split_manifests"):
        raise ValueError("selection and replay plan use different fold manifests")
    if (
        selection.get("metric") != recipe["metric"]["id"]
        or plan.get("metric") != recipe["metric"]["id"]
    ):
        raise ValueError("selection, plan, and recipe do not bind the same official metric")
    if selection.get("target_cache_sha256") != recipe["target_cache"]["sha256"]:
        raise ValueError("selection and recipe do not bind the same target cache")
    if plan.get("target_cache_sha256") != recipe["target_cache"]["sha256"]:
        raise ValueError("plan and recipe do not bind the same target cache")
    if selection.get("primary_contrasts") != recipe["contrasts"]["primary"]:
        raise ValueError("recipe primary contrasts differ from the frozen selection")
    if selection.get("secondary_contrasts") != recipe["contrasts"]["secondary"]:
        raise ValueError("recipe secondary contrasts differ from the frozen selection")
    expected_order = [
        {"name": "s3-baseline", "candidate": "s3", "reference": "baseline", "role": "primary"},
        {
            "name": "s3_s4-baseline",
            "candidate": "s3_s4",
            "reference": "baseline",
            "role": "primary",
        },
        {"name": "s3_s4-s3", "candidate": "s3_s4", "reference": "s3", "role": "secondary"},
    ]
    if recipe["contrasts"].get("ordered") != expected_order:
        raise ValueError("recipe ordered contrasts differ from the frozen quota comparison")
    if plan.get("members") != selection.get("members") or len(plan["members"]) != 64:
        raise ValueError("selection and plan do not bind the same 64 ordered members")
    if set(plan.get("candidates", {})) != set(EXPECTED_CANDIDATES):
        raise ValueError("replay plan candidate roster changed")
    for split in plan["split_manifests"]:
        if sha256_file(resolve_path(split["path"])) != split["sha256"]:
            raise ValueError("replay split manifest hash changed")
        split_data = read_json(resolve_path(split["path"]))
        if split_data.get("fold") != split["fold"]:
            raise ValueError("replay split fold identity changed")
        valid_rows = {int(row["row_group"]): row for row in split_data["sequences"]["valid"]}
        valid_eval = {int(group) for group in split_data["evaluation"]["valid"]}
        fold_members = [member for member in plan["members"] if member["fold"] == split["fold"]]
        if len(fold_members) != 32 or len({item["component"] for item in fold_members}) != 32:
            raise ValueError(f"fold {split['fold']} must contribute 32 distinct components")
        for member in fold_members:
            row = valid_rows.get(int(member["row_group"]))
            if row is None or int(member["row_group"]) not in valid_eval:
                raise ValueError("selected group is absent from source-qualified valid evaluation")
            if int(row["component"]) != int(member["component"]) or int(row["seq_ix"]) != int(
                member["seq_ix"]
            ):
                raise ValueError("selected component or sequence identity differs from its split")
    if len({member["row_group"] for member in plan["members"]}) != 64:
        raise ValueError("selected validation row groups are not unique")
    if len({member["seq_ix"] for member in plan["members"]}) != 64:
        raise ValueError("selected callback sequence IDs are not unique")
    for relative_path, expected_sha256 in plan["source_sha256"].items():
        if sha256_file(resolve_path(relative_path)) != expected_sha256:
            raise ValueError(f"replay source changed: {relative_path}")
    return selection, plan


def verify_package(candidate: str, package: dict[str, Any]) -> dict[str, Any]:
    """Hash an exact candidate ZIP and each embedded native library without loading it."""
    path = resolve_path(package["path"])
    if path.stat().st_size != package["bytes"] or sha256_file(path) != package["sha256"]:
        raise ValueError(f"{candidate} ZIP differs from the replay plan")
    native_hashes: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"{candidate} ZIP has duplicate entries")
        member_map = json.loads(archive.read("members.json"))
        if member_map != package["members"]:
            raise ValueError(f"{candidate} ZIP members.json differs from the plan")
        for name, expected_sha256 in package["native_sha256"].items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            native_hashes[name] = digest.hexdigest()
            if native_hashes[name] != expected_sha256:
                raise ValueError(f"{candidate} packaged native library changed: {name}")
    if sha256_file(path) != package["sha256"]:
        raise ValueError(f"{candidate} ZIP changed during package verification")
    return {"path": str(path), "sha256": package["sha256"], "native_sha256": native_hashes}


def verify_replays(
    recipe: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Require three exact replay receipts and matching ordered feature-only inputs."""
    expected_members = plan["members"]
    paths: dict[str, Path] = {}
    prediction_hashes: dict[str, str] = {}
    receipts: dict[str, Any] = {}
    signatures: dict[str, list[tuple[int, int, int, int, str, str]]] = {}
    for candidate in EXPECTED_CANDIDATES:
        pin = recipe["replay_receipts"][candidate]
        receipt_sha = pin.get("receipt_sha256")
        prediction_sha = pin.get("prediction_sha256")
        if not isinstance(receipt_sha, str) or not isinstance(prediction_sha, str):
            raise ValueError(
                "root must freeze all three receipt and prediction hashes before grading"
            )
        receipt_path = resolve_path(pin["receipt_path"])
        if sha256_file(receipt_path) != receipt_sha:
            raise ValueError(f"{candidate} replay receipt hash differs from the frozen recipe")
        receipt = read_json(receipt_path)
        package = plan["candidates"][candidate]
        if (
            receipt.get("status") != "COMPLETE_FEATURE_ONLY_NATIVE_REPLAY"
            or receipt.get("candidate") != candidate
            or receipt.get("plan_sha256") != recipe["replay_plan"]["sha256"]
            or receipt.get("zip_sha256") != package["sha256"]
            or receipt.get("reset_aba_bitwise") is not True
            or receipt.get("target_values_read") is not False
            or receipt.get("scoring_masks_read") is not False
            or receipt.get("quality_score_computed") is not False
        ):
            raise ValueError(
                f"{candidate} replay receipt does not match the admitted feature-only replay"
            )
        native_name = receipt.get("native_library")
        if native_name not in package["native_sha256"]:
            raise ValueError(f"{candidate} replay loaded an unpinned native library")
        if receipt.get("native_sha256") != package["native_sha256"][native_name]:
            raise ValueError(f"{candidate} replay native hash differs from its ZIP pin")
        if (
            receipt.get("abi", {}).get("passed") is not True
            or receipt.get("abi", {}).get("artifact_sha256") != package["sha256"]
        ):
            raise ValueError(f"{candidate} replay did not pass its packaged ABI check")
        rows = receipt.get("members")
        if not isinstance(rows, list) or len(rows) != 64:
            raise ValueError(f"{candidate} replay receipt does not contain 64 ordered rows")
        signature: list[tuple[int, int, int, int, str, str]] = []
        for member, record in zip(expected_members, rows, strict=True):
            identity = {
                key: int(record[key]) for key in ("fold", "component", "row_group", "seq_ix")
            }
            expected_identity = {key: int(member[key]) for key in identity}
            if identity != expected_identity:
                raise ValueError(f"{candidate} replay member order differs from selection")
            feature_sha = record.get("feature_sha256")
            need_sha = record.get("need_prediction_sha256")
            if not _is_sha256(feature_sha) or not _is_sha256(need_sha):
                raise ValueError(f"{candidate} replay member lacks valid feature/need hashes")
            if record.get("warmup_rows") != 99 or record.get("predicted_rows") != ROWS - 99:
                raise ValueError(
                    f"{candidate} replay member is not a complete post-warmup sequence"
                )
            signature.append(
                (
                    identity["fold"],
                    identity["component"],
                    identity["row_group"],
                    identity["seq_ix"],
                    feature_sha,
                    need_sha,
                )
            )
        signatures[candidate] = signature
        prediction_path = resolve_path(pin["prediction_path"])
        if resolve_path(receipt.get("prediction_path", "")) != prediction_path:
            raise ValueError(f"{candidate} replay prediction path differs from the frozen recipe")
        coverage = npy_header(prediction_path)
        if coverage["shape"] != PREDICTION_SHAPE or coverage["dtype"] != PREDICTION_DTYPE:
            raise ValueError(
                f"{candidate} replay prediction header differs from (64, 20000, 2) FP32"
            )
        actual_prediction_sha = sha256_file(prediction_path)
        if (
            actual_prediction_sha != prediction_sha
            or receipt.get("prediction_sha256") != prediction_sha
            or receipt.get("shape") != PREDICTION_SHAPE
            or receipt.get("dtype") != PREDICTION_DTYPE
        ):
            raise ValueError(f"{candidate} prediction payload differs from its replay receipt pin")
        paths[candidate] = prediction_path
        prediction_hashes[candidate] = actual_prediction_sha
        receipts[candidate] = {
            "path": str(receipt_path),
            "sha256": receipt_sha,
            "prediction_path": str(prediction_path),
            "prediction_sha256": actual_prediction_sha,
            "native_library": native_name,
            "native_sha256": receipt["native_sha256"],
            "members": rows,
            "feature_need_signature": signature,
        }
    reference = signatures["baseline"]
    if any(signatures[candidate] != reference for candidate in EXPECTED_CANDIDATES[1:]):
        raise ValueError(
            "the three replay receipts do not share identical ordered feature/need hashes"
        )
    return paths, prediction_hashes, receipts


def _is_sha256(value: Any) -> bool:
    """Return whether a JSON value is a lowercase 64-character SHA-256."""
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def verify_admission(
    recipe: dict[str, Any], recipe_sha256: str, expected_admission_sha256: str
) -> dict[str, Any]:
    """Require a separate root grade admission bound to the completed replay pins."""
    pin = recipe["grade_admission"]
    expected_sha = expected_admission_sha256
    path = resolve_path(pin["path"])
    if sha256_file(path) != expected_sha:
        raise ValueError("root grade admission file hash differs from the frozen recipe")
    admission = read_json(path)
    expected = {
        "schema": "wnn.quota-package-grade-admission.v1",
        "status": "admitted",
        "recipe_sha256": recipe_sha256,
        "selection_sha256": recipe["selection"]["sha256"],
        "replay_plan_sha256": recipe["replay_plan"]["sha256"],
        "target_cache_sha256": recipe["target_cache"]["sha256"],
        "metric_id": recipe["metric"]["id"],
        "target_quality_reads_authorized": True,
        "output_path": str(resolve_path(recipe["output_path"])),
        "replay_receipt_sha256": {
            name: recipe["replay_receipts"][name]["receipt_sha256"] for name in EXPECTED_CANDIDATES
        },
        "prediction_sha256": {
            name: recipe["replay_receipts"][name]["prediction_sha256"]
            for name in EXPECTED_CANDIDATES
        },
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError("grade admission does not bind the exact recipe, replays, and metric")
    resources = admission.get("resources", {})
    if resources.get("memory_max_bytes") != 400_000_000:
        raise ValueError("grade admission must cap worker memory at 400,000,000 bytes")
    if resources.get("cpu_cores") != 1 or resources.get("max_runtime_seconds") != 120:
        raise ValueError("grade admission must use one CPU core and a 120-second bound")
    if (
        resources.get("swap_bytes") != 0
        or resources.get("scratch") != recipe["resources"]["scratch"]
    ):
        raise ValueError("grade admission must bind zero swap and physical scratch")
    return {"path": str(path), "sha256": expected_sha, "record": admission}


def prelabel_gate(
    recipe_path: Path, expected_recipe_sha256: str, expected_admission_sha256: str
) -> dict[str, Any]:
    """Verify every source, replay, prediction, target, and root-admission pin before labels."""
    recipe_path = recipe_path.resolve(strict=True)
    recipe_sha256 = sha256_file(recipe_path)
    if recipe_sha256 != expected_recipe_sha256:
        raise ValueError("grade recipe SHA-256 differs from the requested identity")
    recipe = read_json(recipe_path)
    if recipe.get("status") not in {"ready_for_grade", "frozen_for_grade"}:
        raise ValueError(
            "draft grade recipe is waiting for root to fill replay hashes and admission"
        )
    if sys.version_info[:2] != tuple(recipe["python"]["version"]):
        raise RuntimeError("grade interpreter version differs from the frozen recipe")
    if Path(sys.executable).resolve() != resolve_path(recipe["python"]["executable"]):
        raise RuntimeError("grade executable differs from the frozen recipe")
    expected_metric = {
        "clip": [-2.0, 2.0],
        "id": "global_wp_clip2_v1",
        "row_mask": "official scored mask and row index >= 99",
        "source": "src/wnn/gmetric.py",
        "weight": "absolute clipped target magnitude",
    }
    if recipe.get("metric") != expected_metric:
        raise ValueError("grade recipe differs from the frozen official metric contract")
    if recipe.get("bootstrap") != {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "source-qualified connected component within fold",
        "scope": "conditional on the three fixed replay predictions",
    }:
        raise ValueError("bootstrap protocol differs from the frozen component resampling plan")
    if recipe.get("score_scope") != {
        "folds": [0, 2],
        "endpoint_scores": ["baseline", "s3", "s3_s4"],
        "score_mask": "official target mask AND row index >= 99",
        "component_count_per_fold": 32,
        "row_group_count": 64,
        "cross_fold_summary": "equal-weight mean of F0 and F2 point deltas; descriptive only",
    }:
        raise ValueError("score scope differs from the frozen F0/F2 sampling panel")
    if recipe.get("resources") != {
        "memory_max_bytes": 400_000_000,
        "cpu_cores": 1,
        "max_runtime_seconds": 120,
        "swap_bytes": 0,
        "scratch": "/home/vlad/development/wunder/research_tmp",
    }:
        raise ValueError("grade worker resources differ from the bounded admission plan")
    if (
        recipe.get("new_fit_count") != 0
        or recipe.get("coefficient_search") is not False
        or recipe.get("competition_upload") is not False
    ):
        raise ValueError(
            "grade recipe may not add fits, search coefficients, or upload submissions"
        )
    foundation_path = resolve_path(recipe["scoring_basis"]["path"])
    if sha256_file(foundation_path) != recipe["scoring_basis"]["sha256"]:
        raise ValueError("F0 scoring basis recipe hash changed")
    foundation = read_json(foundation_path)
    if foundation.get("metric") != recipe["metric"]:
        raise ValueError("grade metric differs from its frozen F0 scoring basis")
    for key in ("path", "file_bytes", "sha256"):
        if foundation.get("target_cache", {}).get(key) != recipe["target_cache"].get(key):
            raise ValueError("grade target cache differs from its frozen F0 scoring basis")
    source_hashes = verify_source_pins(recipe)
    selection, plan = verify_plan_and_selection(recipe)
    package_hashes = {
        candidate: verify_package(candidate, plan["candidates"][candidate])
        for candidate in EXPECTED_CANDIDATES
    }
    prediction_paths, prediction_hashes, receipts = verify_replays(recipe, plan)
    admission = verify_admission(recipe, recipe_sha256, expected_admission_sha256)
    target_record = recipe["target_cache"]
    target_path = resolve_path(target_record["path"])
    if target_path.stat().st_size != target_record["file_bytes"]:
        raise ValueError("official target archive byte size differs from the frozen source")
    target_sha256 = sha256_file(target_path)
    if target_sha256 != target_record["sha256"]:
        raise ValueError("official target archive hash differs from the frozen source")
    output_path = resolve_path(recipe["output_path"])
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"create-only grade output already exists: {output_path}")
    scratch = resolve_path(recipe["resources"]["scratch"])
    if not scratch.is_dir():
        raise ValueError("physical scratch root is unavailable")
    return {
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "selection": selection,
        "plan": plan,
        "source_sha256": source_hashes,
        "package_sha256": package_hashes,
        "prediction_paths": prediction_paths,
        "prediction_sha256": prediction_hashes,
        "replay_receipts": receipts,
        "target_path": target_path,
        "target_sha256_before": target_sha256,
        "admission": admission,
        "output_path": output_path,
        "scratch": scratch,
    }


def _load_targets(
    target_path: Path,
    scratch_dir: Path,
    expected_group_count: int,
    expected_row_groups: set[int],
) -> tuple[Any, Any, dict[int, int], list[Path]]:
    """Extract targets and official masks to physical scratch, then memory-map them."""
    import numpy as np

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(FEASIBILITY), str(ONLINE)]
    from f0_grade_scoring import _extract_npy_member

    with np.load(target_path, allow_pickle=False) as archive:
        if not {"row_group", "y", "mask"}.issubset(archive.files):
            raise ValueError("official target cache lacks row_group, y, or mask")
        row_groups = np.asarray(archive["row_group"])
    if row_groups.ndim != 1 or row_groups.dtype.kind not in "iu":
        raise ValueError("official target row-group vector is malformed")
    if len(row_groups) != expected_group_count or len(np.unique(row_groups)) != len(row_groups):
        raise ValueError("official target row-group vector is incomplete or duplicated")
    positions = {int(group): index for index, group in enumerate(row_groups)}
    missing = expected_row_groups.difference(positions)
    if missing:
        raise ValueError(f"official target cache omits selected row groups: {sorted(missing)[:5]}")
    y_path, mask_path = scratch_dir / "valid_targets_y.npy", scratch_dir / "valid_targets_mask.npy"
    y_coverage = _extract_npy_member(target_path, "y.npy", y_path)
    mask_coverage = _extract_npy_member(target_path, "mask.npy", mask_path)
    if y_coverage["shape"] != [expected_group_count, ROWS, 2] or y_coverage["dtype"] != "<f4":
        raise ValueError("official target member shape or dtype differs from the source contract")
    if mask_coverage["shape"] != [expected_group_count, ROWS] or mask_coverage["dtype"] != "|b1":
        raise ValueError("official score mask shape or dtype differs from the source contract")
    return (
        np.load(y_path, mmap_mode="r", allow_pickle=False),
        np.load(mask_path, mmap_mode="r", allow_pickle=False),
        positions,
        [y_path, mask_path],
    )


def _score_fold(
    fold: int,
    members: list[dict[str, Any]],
    target_maps: tuple[Any, Any, dict[int, int], list[Path]],
    prediction_maps: dict[str, Any],
    metric: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score three fixed endpoints and centered per-group moments for one fold."""
    import numpy as np

    from wnn.evaluation import paired_component_moments
    from wnn.gmetric import GlobalAccumulator

    targets, masks, positions, _ = target_maps
    fold_members = [member for member in members if int(member["fold"]) == fold]
    if len(fold_members) != 32:
        raise ValueError(f"F{fold} does not contain exactly32 selected groups")
    components = np.asarray([int(member["component"]) for member in fold_members], dtype=np.int64)
    if len(np.unique(components)) != 32:
        raise ValueError(f"F{fold} component bootstrap units are not unique")
    names = list(EXPECTED_CANDIDATES)
    accumulators = {name: GlobalAccumulator() for name in names}
    moments = {name: np.zeros((len(fold_members), 2, 6), dtype=np.float64) for name in names}
    rows = np.arange(ROWS)
    fold_entries = [
        (prediction_index, member)
        for prediction_index, member in enumerate(members)
        if int(member["fold"]) == fold
    ]
    for index, (prediction_index, member) in enumerate(fold_entries):
        group = int(member["row_group"])
        if group not in positions:
            raise ValueError(f"official target cache omits F{fold} row group {group}")
        target_row = targets[positions[group]]
        selected = np.asarray(masks[positions[group]], dtype=bool) & (rows >= 99)
        if not selected.any():
            raise ValueError(f"F{fold} row group {group} has no official scored rows")
        for name in names:
            prediction_row = prediction_maps[name][prediction_index]
            accumulators[name].add(target_row, prediction_row, selected)
            centered, _ = paired_component_moments(
                target_row[None, ...],
                prediction_row[None, ...],
                prediction_row[None, ...],
                selected[None, ...],
            )
            moments[name][index] = centered[0]
    scores = {name: accumulator.result() for name, accumulator in accumulators.items()}
    if any(score["metric_id"] != metric["id"] for score in scores.values()):
        raise ValueError("official accumulator returned an unexpected metric ID")
    return scores, {"components": components, "moments": moments, "members": fold_members}


def _contrasts_for_fold(
    fold: int, scores: dict[str, Any], stats: dict[str, Any], recipe: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compute fixed point deltas and conditional paired component intervals."""
    from wnn.evaluation import paired_component_bootstrap

    results: list[dict[str, Any]] = []
    for contrast in recipe["contrasts"]["ordered"]:
        name = contrast["name"]
        candidate = contrast["candidate"]
        reference = contrast["reference"]
        role = contrast["role"]
        candidate_score, reference_score = scores[candidate], scores[reference]
        if candidate_score["selected_rows"] != reference_score["selected_rows"]:
            raise ValueError(f"F{fold} contrast {name} uses mismatched scored rows")
        interval = paired_component_bootstrap(
            stats["moments"][candidate],
            stats["moments"][reference],
            stats["components"],
            n_boot=recipe["bootstrap"]["draws"],
            seed=recipe["bootstrap"]["seed"],
            centered=True,
        )
        results.append(
            {
                "name": name,
                "role": role,
                "candidate": candidate,
                "reference": reference,
                "point_deltas": {
                    metric: float(candidate_score[metric] - reference_score[metric])
                    for metric in ("wp", "t0", "t1")
                },
                "selected_rows": candidate_score["selected_rows"],
                "component_bootstrap": interval,
                "uncertainty_scope": "conditional on the fixed replay predictions",
            }
        )
    return results


def score_prepared(prepared: dict[str, Any]) -> dict[str, Any]:
    """Stream the target cache, grade F0/F2 separately, and verify post-score hashes."""
    import numpy as np

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(FEASIBILITY), str(ONLINE)]
    from f0_grade_scoring import verify_target_hashes

    recipe = prepared["recipe"]
    predictions: dict[str, Any] = {}
    for candidate in EXPECTED_CANDIDATES:
        prediction_path = prepared["prediction_paths"][candidate]
        predictions[candidate] = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    results: dict[str, Any] = {}
    target_identity: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(
        prefix="quota-package-grade-", dir=prepared["scratch"]
    ) as temp_name:
        target_maps = _load_targets(
            prepared["target_path"],
            Path(temp_name),
            recipe["target_cache"]["row_group_count"],
            {int(member["row_group"]) for member in prepared["plan"]["members"]},
        )
        for fold in (0, 2):
            scores, stats = _score_fold(
                fold,
                prepared["plan"]["members"],
                target_maps,
                predictions,
                recipe["metric"],
            )
            results[str(fold)] = {
                "endpoint_scores": scores,
                "contrasts": _contrasts_for_fold(fold, scores, stats, recipe),
                "component_count": int(len(stats["components"])),
                "ordered_groups": [int(member["row_group"]) for member in stats["members"]],
            }
        target_sha_after = sha256_file(prepared["target_path"])
        target_identity = verify_target_hashes(
            recipe["target_cache"]["sha256"],
            prepared["target_sha256_before"],
            target_sha_after,
        )
        del target_maps
    del predictions
    prediction_after = {
        candidate: sha256_file(prepared["prediction_paths"][candidate])
        for candidate in EXPECTED_CANDIDATES
    }
    if prediction_after != prepared["prediction_sha256"]:
        raise ValueError("a replay prediction payload changed during grading")
    summary: dict[str, Any] = {}
    for contrast_index, contrast in enumerate(recipe["contrasts"]["ordered"]):
        name = contrast["name"]
        f0_delta = results["0"]["contrasts"][contrast_index]["point_deltas"]["wp"]
        f2_delta = results["2"]["contrasts"][contrast_index]["point_deltas"]["wp"]
        summary[name] = {
            "equal_weight_F0_F2_mean_wp_delta_descriptive": 0.5 * (f0_delta + f2_delta)
        }
    return {
        "schema": "wnn.quota-package-fixed-replay-grade.v1",
        "status": "graded_exploratory_exposed_sample",
        "recipe_sha256": prepared["recipe_sha256"],
        "selection_sha256": recipe["selection"]["sha256"],
        "replay_plan_sha256": recipe["replay_plan"]["sha256"],
        "grade_admission": prepared["admission"],
        "metric": recipe["metric"],
        "score_mask": "official target mask AND row index >= 99",
        "target_cache_identity": {"path": str(prepared["target_path"]), **target_identity},
        "replay_receipts": prepared["replay_receipts"],
        "source_sha256": prepared["source_sha256"],
        "packages": prepared["package_sha256"],
        "prediction_sha256_after_score": prediction_after,
        "folds": results,
        "cross_fold_descriptive_summary": summary,
        "bootstrap": recipe["bootstrap"],
        "new_fit_count": 0,
        "exposure": recipe["exposure"],
        "interpretation_limits": recipe["interpretation_limits"],
        "competition_upload": False,
        "expected_public_gain": None,
    }


def main() -> None:
    """Grade only after complete replay receipts, exact hashes, and root admission."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    prepared = prelabel_gate(args.recipe, args.recipe_sha256, args.admission_sha256)
    result = score_prepared(prepared)
    with prepared["output_path"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
