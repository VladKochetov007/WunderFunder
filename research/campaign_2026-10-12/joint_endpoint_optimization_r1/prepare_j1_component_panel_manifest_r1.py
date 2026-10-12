"""Prepare a frozen J1 panel manifest without reading prediction payloads."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PANEL_SALT = "j1-clean-t3-f0-component-panel-r1"
MANIFEST_NAME = "J1_COMPONENT_PANEL_MANIFEST_R2.json"
VALIDATION_NAME = "J1_STATIC_PREPARATION_VALIDATION_R2.json"
RESOURCE_PROFILE_NAME = "J1_RESOURCE_PROFILE_R1.json"
MAX_NPY_HEADER_BYTES = 65_536
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_PROPENSITY_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
SEEDS = (3, 4, 5)
CAL_COMPONENTS = 32
EVAL_COMPONENTS = 16


def sha256_bytes(raw: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a small metadata/index file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy_header(path: Path) -> dict[str, Any]:
    """Read and validate only an NPY header, with a strict size bound."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"invalid NPY magic in {path}")
        version = tuple(prefix[6:8])
        if version[0] == 1:
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"truncated NPY header length in {path}")
            header_length = struct.unpack("<H", length_bytes)[0]
            prefix_bytes = 10
        elif version[0] in (2, 3):
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError(f"truncated NPY header length in {path}")
            header_length = struct.unpack("<I", length_bytes)[0]
            prefix_bytes = 12
        else:
            raise ValueError(f"unsupported NPY version {version} in {path}")
        if not 0 < header_length <= MAX_NPY_HEADER_BYTES:
            raise ValueError(f"NPY header exceeds bound in {path}: {header_length}")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated NPY header in {path}")

    encoding = "utf-8" if version[0] == 3 else "latin1"
    header = ast.literal_eval(header_bytes.decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError(f"NPY header is not a dictionary in {path}")
    shape = tuple(int(value) for value in header["shape"])
    dtype = np.dtype(header["descr"])
    expected_bytes = prefix_bytes + header_length + math.prod(shape) * dtype.itemsize
    actual_bytes = path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(f"NPY file size disagrees with header in {path}")
    return {
        "version": f"{version[0]}.{version[1]}",
        "header_length_bytes": header_length,
        "header_offset_bytes": prefix_bytes + header_length,
        "dtype": dtype.str,
        "shape": list(shape),
        "fortran_order": bool(header["fortran_order"]),
        "payload_bytes": math.prod(shape) * dtype.itemsize,
        "file_bytes": actual_bytes,
    }


def stable_membership_hash(group_ids: list[int]) -> str:
    """Hash sorted group IDs in canonical little-endian int64 representation."""
    values = np.asarray(sorted(group_ids), dtype="<i8")
    return sha256_bytes(values.tobytes())


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read a bounded JSON metadata file and return its parsed object and bytes."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value, raw


def verify_component_partition(
    project_root: Path, split: dict[str, Any]
) -> dict[tuple[str, int], int]:
    """Check the split manifest's component partition against the small graph cache."""
    # Import the repository's canonical graph helper without reading any dataset.
    source_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(source_root / "src"))
    from wnn.splits import connected_components

    cache_path = project_root / "cache/chains.npz"
    graph = np.load(cache_path, allow_pickle=False)
    derived = connected_components(graph["next"])
    graph_component = {
        (str(source_split), int(group)): int(component)
        for source_split, group, component in zip(
            graph["split"], graph["row_group"], derived, strict=True
        )
    }
    manifest_component = {
        (source_split, int(node["row_group"])): int(node["component"])
        for source_split, nodes in split["sequences"].items()
        for node in nodes
    }
    if graph_component.keys() != manifest_component.keys():
        raise ValueError("component graph nodes disagree with the split manifest")

    graph_to_manifest: dict[int, int] = {}
    manifest_to_graph: dict[int, int] = {}
    for key, graph_id in graph_component.items():
        manifest_id = manifest_component[key]
        if graph_id in graph_to_manifest and graph_to_manifest[graph_id] != manifest_id:
            raise ValueError("manifest splits a connected component")
        if manifest_id in manifest_to_graph and manifest_to_graph[manifest_id] != graph_id:
            raise ValueError("manifest merges distinct connected components")
        graph_to_manifest[graph_id] = manifest_id
        manifest_to_graph[manifest_id] = graph_id
    return manifest_component


def build(project_root: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build static source pins, component membership, and safe header receipts."""
    split_path = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split, split_raw = load_json(split_path)
    split_sha = sha256_bytes(split_raw)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("fold0 split manifest hash changed")

    source_records = split["sources"]
    for relative in ("cache/chain_folds.npz", "cache/chains.npz"):
        path = project_root / relative
        identity = source_records[relative]
        if path.stat().st_size != identity["bytes"]:
            raise ValueError(f"small graph cache size changed: {relative}")
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"small graph cache hash changed: {relative}")
    valid_path = project_root / "datasets/valid.parquet"
    valid_identity = source_records["datasets/valid.parquet"]
    valid_stat = valid_path.stat()
    if (valid_stat.st_size, valid_stat.st_mtime_ns) != (
        valid_identity["bytes"],
        valid_identity["mtime_ns"],
    ):
        raise ValueError("valid.parquet stat identity changed")

    component_of = verify_component_partition(project_root, split)
    split_eval = set(map(int, split["evaluation"]["valid"]))
    split_fit_valid = set(map(int, split["fit"]["valid"]))
    split_fit_train = set(map(int, split["fit"]["train"]))
    valid_nodes = {int(node["row_group"]): node for node in split["sequences"]["valid"]}
    train_nodes = {int(node["row_group"]): node for node in split["sequences"]["train"]}
    eval_components = {int(valid_nodes[group]["component"]) for group in split_eval}
    if len(split_eval) != 468 or len(eval_components) != 141:
        raise ValueError("fold0 evaluation membership counts changed")

    chain_folds = np.load(project_root / "cache/chain_folds.npz", allow_pickle=False)
    fold_zero = {
        int(group)
        for source_split, group, fold in zip(
            chain_folds["split"],
            chain_folds["row_group"],
            chain_folds["fold"],
            strict=True,
        )
        if source_split == "valid" and int(fold) == 0
    }
    if fold_zero != split_eval:
        raise ValueError("fold0 membership disagrees with chain_folds.npz")

    producer_relative = Path(
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
        "/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort"
        "/r6_bootstrap_guard_successor_20261012_r1"
    )
    producer = project_root / producer_relative
    terminal_receipt_relative = producer_relative / "terminal_collection_v2"
    terminal_receipt = (
        project_root / terminal_receipt_relative / "COMPLETE_COLLECTOR_RECEIPT_R1.json"
    )
    terminal_receipt_obj, terminal_receipt_raw = load_json(terminal_receipt)
    if not terminal_receipt_obj.get("terminal_collection", {}).get("collection_complete"):
        raise ValueError("R6 collected output inventory is incomplete")

    ops_relative = Path(
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
        "/ops_continuation_20261011"
    )
    grade_relative = (
        ops_relative
        / "clean_t3_local_equal_trio_grade_r1"
        / ("F0_EQUAL_TEACHER_TRIO_GRADE_R1.json")
    )
    input_validation_relative = (
        ops_relative
        / "clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1"
        / "ACTUAL_INPUT_VALIDATION_R2.json"
    )
    cache_validation_relative = (
        ops_relative
        / "clean_t3_f0_matched_student_r5_cache_r1/cache_adapter_r1"
        / "CACHE_MATERIALIZATION_VALIDATION_R1.json"
    )
    producer_preflight_relative = (
        ops_relative
        / "clean_t3_local_s3_export_r1"
        / "ROOT_S4_S5_ACTUAL_PRODUCER_PREFLIGHT_R1.json"
    )
    raw_valid_review_relative = (
        ops_relative / "clean_t3_local_s3_export_r1" / "ROOT_RAW_VALID_CONTENT_REVIEW_R1.json"
    )
    grade_path = project_root / grade_relative
    input_validation_path = project_root / input_validation_relative
    cache_validation_path = project_root / cache_validation_relative
    producer_preflight_path = project_root / producer_preflight_relative
    raw_valid_review_path = project_root / raw_valid_review_relative
    grade, grade_raw = load_json(grade_path)
    input_validation, input_validation_raw = load_json(input_validation_path)
    cache_validation, cache_validation_raw = load_json(cache_validation_path)
    producer_preflight, producer_preflight_raw = load_json(producer_preflight_path)
    raw_valid_review, raw_valid_review_raw = load_json(raw_valid_review_path)
    if grade.get("status") != "PASS_OFFICIAL_EXPOSED_FOLD0_DISTINCT_SEED_AND_FIXED_TRIO_GRADE":
        raise ValueError("actual exposed-F0 teacher grade is absent or not passing")
    grade_metric = grade.get("metric", {})
    if (
        grade_metric.get("id") != "global_wp_clip2_v1"
        or grade_metric.get("split_sha256") != split_sha
        or grade_metric.get("selected_groups") != 468
        or grade_metric.get("components") != 141
    ):
        raise ValueError("actual grade does not bind the expected F0 metric panel")
    if input_validation.get("status") != "PASS_ALL_TEACHER_INPUTS_READY":
        raise ValueError("full prediction payload SHA validation receipt is absent")
    if cache_validation.get("status") != "PASS_FROZEN_CONSUMER_AND_ACTUAL_OUTPUT_VALIDATION":
        raise ValueError("actual cache consumer validation receipt is absent")
    if cache_validation.get("actual_producer_input_validation_sha256") != sha256_bytes(
        input_validation_raw
    ):
        raise ValueError("cache validation does not bind the exact full-input validation receipt")
    if (
        producer_preflight.get("status")
        != "PASS_ACTUAL_CP_SOURCE_SPLIT_FULL_VALID_SELECTOR_BEFORE_CUDA"
    ):
        raise ValueError("actual s4/s5 producer preflight is absent")
    if grade.get("input_hashes", {}).get("root_s4_s5_producer_preflight_sha256") != sha256_bytes(
        producer_preflight_raw
    ):
        raise ValueError("official teacher grade does not bind the actual s4/s5 source preflight")
    if (
        raw_valid_review.get("status") != "PASS"
        or raw_valid_review.get("full_content_rehashed") is not True
        or raw_valid_review.get("observed_full_sha256")
        != raw_valid_review.get("expected_frozen_producer_sha256")
        or raw_valid_review.get("bytes") != valid_identity["bytes"]
    ):
        raise ValueError("raw valid parquet source has no matching full-content receipt")
    grade_driver_relative = grade_relative.with_name("grade_equal_trio_r1.py")
    grade_driver_path = project_root / grade_driver_relative
    grade_driver_sha = sha256_file(grade_driver_path)
    if grade.get("execution", {}).get("driver_sha256") != grade_driver_sha:
        raise ValueError("grade receipt does not bind its present scoring driver")

    # The official grade checks every scored row in the complete 468-group
    # evaluation universe for finite predictions. The stored full-file hashes
    # bind every byte; unscored/prefix prediction values are not claimed finite.
    grade_endpoints = grade.get("endpoints", {}).get("teachers", {})
    input_teachers = {
        record["provenance"]["run"]: record for record in input_validation["teacher_records"]
    }

    candidate_records: list[dict[str, Any]] = []
    common_fit_train: set[int] | None = None
    common_fit_valid: set[int] | None = None
    common_eval: set[int] | None = None
    common_run_source: str | None = None
    common_propensity: str | None = None
    common_recipe_source: str | None = None
    for seed in SEEDS:
        run_name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r5"
        run_relative = Path(
            "terminal_collection_v2/provider_outputs/attempt_001/outputs/experiment"
            f"/artifacts/runs/{run_name}"
        )
        provenance_path = producer / run_relative / "provenance.json"
        provenance, provenance_raw = load_json(provenance_path)
        prediction_name = f"pred_{run_name}_local_full_valid_r1_valid"
        prediction_path = project_root / "cache" / f"{prediction_name}.npy"
        sidecar_path = project_root / "cache" / f"{prediction_name}.json"
        sidecar, sidecar_raw = load_json(sidecar_path)
        fit_train = set(map(int, provenance["fit_row_groups"]["train"]))
        fit_valid = set(map(int, provenance["fit_row_groups"]["valid"]))
        eval_groups = set(map(int, provenance["evaluation_row_groups"]))
        args = provenance["args"]
        if provenance["split_manifest_sha256"] != split_sha:
            raise ValueError(f"split manifest changed in {run_name}")
        if (
            eval_groups != split_eval
            or fit_train != split_fit_train
            or fit_valid != split_fit_valid
        ):
            raise ValueError(f"fit/evaluation membership changed in {run_name}")
        if eval_groups & fit_valid:
            raise ValueError(f"valid fit/evaluation group overlap in {run_name}")
        fit_components = {
            component_of[(source_split, group)]
            for source_split, groups in (("train", fit_train), ("valid", fit_valid))
            for group in groups
        }
        if fit_components & eval_components:
            raise ValueError(f"fit/evaluation component overlap in {run_name}")
        expected_args = {
            "seed": seed,
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "train_source": "train+valid",
            "arch": "gru",
            "hidden": 384,
            "layers": 2,
            "optim": "muon",
            "lr": 0.001,
            "wd": 0.01,
            "batch": 32,
            "epochs": 3,
            "since_weight": 32.0,
            "propensity": "c20260929_prop_f0",
        }
        if any(args.get(key) != value for key, value in expected_args.items()):
            raise ValueError(f"recipe seed/fold changed in {run_name}")
        if args["init"] or args["distill"]:
            raise ValueError(f"unexpected model initializer/teacher in {run_name}")
        if args["precision"] != "fp16" or args["arch"] != "gru":
            raise ValueError(f"training source recipe changed in {run_name}")
        if provenance["selected_epoch"] != 2 or sidecar["epoch"] != 2:
            raise ValueError(f"selected epoch changed in {run_name}")
        checkpoint_sha = provenance["checkpoint_sha256"]["model_ep2.pt"]
        if sidecar["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError(f"prediction checkpoint does not bind to epoch 2 in {run_name}")
        if sidecar["run"] != run_name or sidecar["split"] != "valid":
            raise ValueError(f"prediction sidecar run/split changed in {run_name}")
        if sidecar["split_manifest_sha256"] != split_sha:
            raise ValueError(f"prediction sidecar split hash changed in {run_name}")
        if sidecar["campaign_split_manifest_sha256"] != split_sha:
            raise ValueError(f"prediction sidecar campaign split hash changed in {run_name}")
        if sidecar["campaign_fit_scope"] != "full-valid-feature-only":
            raise ValueError(f"unexpected prediction scope in {run_name}")
        if sidecar["inference_precision"] != "fp32" or sidecar["quantization_preset"] is not None:
            raise ValueError(f"unexpected prediction precision in {run_name}")
        if sidecar["row_groups"] != list(range(1873)):
            raise ValueError(f"prediction row-group index changed in {run_name}")
        header = read_npy_header(prediction_path)
        if header["shape"] != [1873, 20_000, 2] or header["dtype"] != "<f4":
            raise ValueError(f"prediction array layout changed in {run_name}")
        if header["fortran_order"]:
            raise ValueError(f"unexpected Fortran NPY storage in {run_name}")

        key = f"s{seed}"
        grade_endpoint = grade_endpoints.get(key)
        input_record = input_teachers.get(run_name)
        if not isinstance(grade_endpoint, dict) or not isinstance(input_record, dict):
            raise ValueError(f"existing full prediction validation is missing for {run_name}")
        input_valid = input_record["predictions"]["valid"]
        input_provenance = input_record["provenance"]
        input_array = input_valid.get("array", {})
        actual_stat = prediction_path.stat()
        grade_input_hashes = grade.get("input_hashes", {})
        if (
            grade_endpoint.get("run") != run_name
            or grade_endpoint.get("seed") != seed
            or grade_endpoint.get("checkpoint_epoch") != 2
            or grade_endpoint.get("checkpoint_sha256") != checkpoint_sha
            or grade_endpoint.get("provenance_sha256") != sha256_bytes(provenance_raw)
            or grade_endpoint.get("prediction_sha256") != sidecar["prediction_sha256"]
            or grade_endpoint.get("sidecar_sha256") != sha256_bytes(sidecar_raw)
            or grade_endpoint.get("inference_precision") != "fp32"
            or grade_endpoint.get("direct_valid_evaluation_overlap") != 0
            or grade_input_hashes.get("teacher_prediction_sha256", {}).get(key)
            != sidecar["prediction_sha256"]
            or grade_input_hashes.get("teacher_checkpoint_sha256", {}).get(key) != checkpoint_sha
            or grade_input_hashes.get("teacher_provenance_sha256", {}).get(key)
            != sha256_bytes(provenance_raw)
            or grade_input_hashes.get("teacher_sidecar_sha256", {}).get(key)
            != sha256_bytes(sidecar_raw)
        ):
            raise ValueError(f"actual official grade does not bind {run_name} source/output")
        if (
            input_provenance.get("provenance_sha256") != sha256_bytes(provenance_raw)
            or input_provenance.get("checkpoint_sha256") != checkpoint_sha
            or input_valid.get("sidecar_sha256") != sha256_bytes(sidecar_raw)
            or input_valid.get("prediction_sha256") != sidecar["prediction_sha256"]
            or input_array.get("sha256") != sidecar["prediction_sha256"]
            or input_array.get("hash_recomputed") is not True
            or input_array.get("shape") != [1873, 20_000, 2]
            or input_array.get("dtype") != "float32"
            or input_array.get("bytes") != actual_stat.st_size
            or input_array.get("mtime_ns") != actual_stat.st_mtime_ns
            or input_array.get("inode") != actual_stat.st_ino
        ):
            raise ValueError(f"actual full-payload hash receipt/stat does not bind {run_name}")

        for slot, values, message in (
            ("fit_train", fit_train, "train fit"),
            ("fit_valid", fit_valid, "valid fit"),
            ("eval", eval_groups, "evaluation"),
        ):
            prior = {
                "fit_train": common_fit_train,
                "fit_valid": common_fit_valid,
                "eval": common_eval,
            }[slot]
            if prior is not None and prior != values:
                raise ValueError(f"seed members have different {message} membership")
            if slot == "fit_train":
                common_fit_train = values
            elif slot == "fit_valid":
                common_fit_valid = values
            else:
                common_eval = values
        run_source = provenance["run_identity_source_sha256"]
        propensity_sha = provenance["propensity_checkpoint_sha256"]
        recipe_source = provenance["args"]["split_manifest_sha256"]
        if common_run_source is not None and common_run_source != run_source:
            raise ValueError("seed members have different run identity source")
        if common_propensity is not None and common_propensity != propensity_sha:
            raise ValueError("seed members have different propensity sources")
        if common_recipe_source is not None and common_recipe_source != recipe_source:
            raise ValueError("seed members have different split source")
        common_run_source = run_source
        common_propensity = propensity_sha
        common_recipe_source = recipe_source
        candidate_records.append(
            {
                "seed": seed,
                "run_name": run_name,
                "training_recipe": {
                    "architecture": args["arch"],
                    "hidden": args["hidden"],
                    "layers": args["layers"],
                    "optimizer": args["optim"],
                    "training_precision": args["precision"],
                    "epochs": args["epochs"],
                    "selected_epoch": provenance["selected_epoch"],
                    "initializer": args["init"],
                    "distillation_source": args["distill"],
                    "propensity_source": args["propensity"],
                    "propensity_checkpoint_sha256": propensity_sha,
                    "run_identity_source_sha256": run_source,
                },
                "checkpoint": {
                    "path": (producer_relative / run_relative / "model_ep2.pt").as_posix(),
                    "sha256_from_provenance": checkpoint_sha,
                },
                "prediction": {
                    "path": f"cache/{prediction_name}.npy",
                    "producer_sidecar_path": f"cache/{prediction_name}.json",
                    "producer_prediction_sha256": sidecar["prediction_sha256"],
                    "producer_sidecar_sha256": sha256_bytes(sidecar_raw),
                    "official_grade_prediction_sha256": grade_endpoint["prediction_sha256"],
                    "official_grade_sidecar_sha256": grade_endpoint["sidecar_sha256"],
                    "full_payload_sha256_recomputed_by_validation": input_array["sha256"],
                    "full_payload_validation_stat": {
                        "bytes": input_array["bytes"],
                        "mtime_ns": input_array["mtime_ns"],
                        "inode": input_array["inode"],
                    },
                    "finite_scope_from_official_grade": (
                        "all official scored rows in all 468 evaluation groups; the grade's "
                        "fixed mask is is_scored AND row >= 99; unscored/prefix bytes are "
                        "hash-bound but not claimed finite"
                    ),
                    "provenance_path": (
                        producer_relative / run_relative / "provenance.json"
                    ).as_posix(),
                    "provenance_sha256": sha256_bytes(provenance_raw),
                    "npy_header_only": header,
                    "hash_status": (
                        "full file SHA recomputed in existing actual input validation and "
                        "bound in the actual official grade; current stat/header checked "
                        "without rereading payload in prep"
                    ),
                },
                "fit_membership": {
                    "train_groups": len(fit_train),
                    "train_groups_sha256_sorted_int64le": stable_membership_hash(list(fit_train)),
                    "valid_groups": len(fit_valid),
                    "valid_groups_sha256_sorted_int64le": stable_membership_hash(list(fit_valid)),
                    "evaluation_groups": len(eval_groups),
                    "evaluation_groups_sha256_sorted_int64le": stable_membership_hash(
                        list(eval_groups)
                    ),
                    "fit_eval_group_overlap": 0,
                    "fit_eval_component_overlap": 0,
                },
            }
        )

    propensity_path = project_root / "artifacts/runs/c20260929_prop_f0/provenance.json"
    propensity, propensity_raw = load_json(propensity_path)
    if propensity["checkpoint_sha256"]["model.pt"] != EXPECTED_PROPENSITY_SHA256:
        raise ValueError("shared propensity checkpoint provenance changed")
    if propensity["selected_epoch"] != 3 or propensity["args"]["checkpoint_epoch"] != 3:
        raise ValueError("shared propensity selected epoch changed")
    if set(map(int, propensity["fit_row_groups"]["valid"])) != split_fit_valid:
        raise ValueError("propensity fit membership is not the declared valid fit split")
    if set(map(int, propensity["evaluation_row_groups"])) != split_eval:
        raise ValueError("propensity evaluation membership changed")
    if propensity["split_manifest_sha256"] != split_sha:
        raise ValueError("propensity split manifest hash changed")
    propensity_fit_components = {
        component_of[("valid", group)] for group in map(int, propensity["fit_row_groups"]["valid"])
    }
    if propensity_fit_components & eval_components:
        raise ValueError("propensity fit/evaluation component overlap")

    target_archive_path = project_root / "cache/valid_targets.npz"
    target_archive_stat = target_archive_path.stat()
    target_sha = grade_metric.get("valid_targets_sha256")
    if not isinstance(target_sha, str) or len(target_sha) != 64:
        raise ValueError("official grade does not bind the exact valid target cache")
    with np.load(target_archive_path, allow_pickle=False) as archive:
        if not {"row_group", "seq_ix"}.issubset(archive.files):
            raise ValueError("valid target cache lacks source-qualified row indices")
        target_row_groups = np.asarray(archive["row_group"], dtype=np.int64)
        target_seq_ix = np.asarray(archive["seq_ix"], dtype=np.int64)
    expected_target_groups = np.arange(1873, dtype=np.int64)
    expected_target_seq_ix = np.asarray(
        [int(valid_nodes[group]["seq_ix"]) for group in range(1873)], dtype=np.int64
    )
    if not np.array_equal(target_row_groups, expected_target_groups):
        raise ValueError("valid target cache row-group order differs from the split manifest")
    if not np.array_equal(target_seq_ix, expected_target_seq_ix):
        raise ValueError("valid target cache sequence identities differ from the split manifest")

    component_to_groups: dict[int, list[int]] = {}
    for group in split_eval:
        component_to_groups.setdefault(int(valid_nodes[group]["component"]), []).append(group)
    ordered_components = sorted(
        component_to_groups,
        key=lambda component: (
            sha256_bytes(f"{PANEL_SALT}|valid|fold=0|component={component}".encode("ascii")),
            component,
        ),
    )
    selected = ordered_components[: CAL_COMPONENTS + EVAL_COMPONENTS]
    calibration_components = selected[:CAL_COMPONENTS]
    evaluation_components = selected[CAL_COMPONENTS:]
    if len(selected) != CAL_COMPONENTS + EVAL_COMPONENTS:
        raise ValueError("not enough source-qualified components for the frozen panel")

    def panel_groups(component_ids: list[int]) -> list[dict[str, int | str]]:
        selected_groups = sorted(
            group for component in component_ids for group in component_to_groups[component]
        )
        return [
            {
                "source": "datasets/valid.parquet",
                "row_group": group,
                "seq_ix": int(valid_nodes[group]["seq_ix"]),
                "component": int(valid_nodes[group]["component"]),
            }
            for group in selected_groups
        ]

    calibration_groups = panel_groups(calibration_components)
    evaluation_groups = panel_groups(evaluation_components)
    selected_component_set = set(selected)
    excluded_train_groups = sorted(
        group
        for group, node in train_nodes.items()
        if int(node["component"]) in selected_component_set
    )
    all_train_fit = split_fit_train
    if set(excluded_train_groups) & all_train_fit:
        raise ValueError("selected evaluation components overlap fitted train groups")

    primitive_path = project_root / "src/wnn/blend_optimization.py"
    metric_path = project_root / "src/wnn/gmetric.py"
    evaluator_path = project_root / "scripts/evaluate_bandit_continuation.py"
    receipt_rel = (terminal_receipt_relative / "COMPLETE_COLLECTOR_RECEIPT_R1.json").as_posix()
    runner_relative = Path(
        "research/campaign_2026-10-12/joint_endpoint_optimization_r1/j1_panel_runner_r1.py"
    )
    runner_path = Path(__file__).resolve().with_name("j1_panel_runner_r1.py")
    if not runner_path.is_file():
        raise ValueError("bounded J1 runner is absent")
    manifest: dict[str, Any] = {
        "schema": "wnn.j1-component-panel-manifest.v2",
        "prepared_date": date.today().isoformat(),
        "status": "prepared_pending_root_review_and_admission",
        "experiment": {
            "id": "J1-clean-t3-f0-three-seed-blend-panel-r1",
            "claim_scope": "bounded exposed F0 fixed-prediction blend screen only",
            "not_claimed": [
                "fresh holdout confirmation",
                "training-seed uncertainty",
                "public transfer or public score forecast",
                "global optimality or exact official-score gradient",
                "serving or quantized endpoint performance",
            ],
            "no_model_fit_or_prediction_generation": True,
        },
        "source_identity": {
            "split_manifest": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "split_manifest_sha256": split_sha,
            "valid_parquet": {
                "path": "datasets/valid.parquet",
                "bytes": valid_identity["bytes"],
                "mtime_ns": valid_identity["mtime_ns"],
                "footer_bytes": valid_identity["footer_bytes"],
                "footer_sha256_from_split_manifest": valid_identity["footer_sha256"],
                "footer_rehashed_in_prep": False,
            },
            "raw_valid_content_review": {
                "path": raw_valid_review_relative.as_posix(),
                "sha256": sha256_bytes(raw_valid_review_raw),
                "status": raw_valid_review["status"],
                "full_content_rehashed": raw_valid_review["full_content_rehashed"],
                "full_sha256": raw_valid_review["observed_full_sha256"],
                "note": "Historical full-content source receipt; prep itself only checks stat/footer metadata.",
            },
            "actual_prediction_grade": {
                "path": grade_relative.as_posix(),
                "sha256": sha256_bytes(grade_raw),
                "status": grade["status"],
                "metric_id": grade_metric["id"],
                "scored_groups": grade_metric["selected_groups"],
                "components": grade_metric["components"],
                "source_preflight_sha256": grade["input_hashes"][
                    "root_s4_s5_producer_preflight_sha256"
                ],
                "grade_driver_path": grade_driver_relative.as_posix(),
                "grade_driver_sha256": grade_driver_sha,
                "finite_scope": (
                    "official metric validates finite target/prediction values on its fixed "
                    "scored mask in all 468 evaluation groups; no all-byte finite claim "
                    "for unscored/prefix rows"
                ),
            },
            "actual_full_prediction_validation": {
                "path": input_validation_relative.as_posix(),
                "sha256": sha256_bytes(input_validation_raw),
                "status": input_validation["status"],
                "array_hashing": input_validation.get("array_hashing"),
                "cache_consumer_validation_path": cache_validation_relative.as_posix(),
                "cache_consumer_validation_sha256": sha256_bytes(cache_validation_raw),
                "cache_consumer_validation_status": cache_validation["status"],
                "note": (
                    "Full prediction NPY payload SHA values were recomputed by this existing "
                    "receipt and cross-bound to current stat identities; finite assurance for "
                    "the J1 metric domain is from the official grade's selected-row validation."
                ),
            },
            "actual_source_preflight": {
                "path": producer_preflight_relative.as_posix(),
                "sha256": sha256_bytes(producer_preflight_raw),
                "status": producer_preflight["status"],
            },
            "runner": {
                "path": runner_relative.as_posix(),
                "sha256": sha256_file(runner_path),
                "entrypoint": "scripts/limited python j1_panel_runner_r1.py",
            },
            "valid_target_cache": {
                "path": "cache/valid_targets.npz",
                "bytes": target_archive_stat.st_size,
                "mtime_ns": target_archive_stat.st_mtime_ns,
                "inode": target_archive_stat.st_ino,
                "sha256_from_official_grade": target_sha,
                "row_group_order_sha256": stable_membership_hash(target_row_groups.tolist()),
                "seq_ix_sha256_sorted_int64le": sha256_bytes(
                    target_seq_ix.astype("<i8", copy=False).tobytes()
                ),
                "index_members_checked_in_prep": ["row_group.npy", "seq_ix.npy"],
                "target_and_mask_members_read_in_prep": False,
            },
            "component_graph_sources": {
                relative: {
                    "bytes": source_records[relative]["bytes"],
                    "sha256": source_records[relative]["sha256"],
                    "component_partition_matches_split_manifest": True,
                }
                for relative in ("cache/chain_folds.npz", "cache/chains.npz")
            },
            "terminal_collection_receipt": {
                "path": receipt_rel,
                "sha256": sha256_bytes(terminal_receipt_raw),
                "terminal_status": terminal_receipt_obj["terminal_status"],
                "output_exit_code": terminal_receipt_obj["attempt_result"]["output"]["exit_code"],
                "collection_complete": terminal_receipt_obj["terminal_collection"][
                    "collection_complete"
                ],
                "provider_status_note": f"Recorded provider terminal status {terminal_receipt_obj['terminal_status']} coexists with collector output exit code {terminal_receipt_obj['attempt_result']['output']['exit_code']} and collection_complete={terminal_receipt_obj['terminal_collection']['collection_complete']}; this prep does not infer fit or score success from provider status.",
            },
            "scoring_sources": {
                "metric_id": "global_wp_clip2_v1",
                "metric_source": "src/wnn/gmetric.py",
                "metric_source_sha256": sha256_file(metric_path),
                "blend_primitive": "src/wnn/blend_optimization.py",
                "blend_primitive_sha256": sha256_file(primitive_path),
                "fixed_mask_source": "scripts/evaluate_bandit_continuation.py",
                "fixed_mask_source_sha256": sha256_file(evaluator_path),
                "official_mask_column": "is_scored",
                "need_prediction_column": "need_prediction",
                "fixed_mask_rule": "need_prediction AND is_scored",
                "need_prediction_rule": "step_in_seq >= 99",
                "warmup_rows": 99,
                "target_columns": ["t0", "t1"],
                "fixed_mask_clip": 2.0,
                "weight": "abs(clip(target, -2, 2))",
            },
        },
        "members": candidate_records,
        "shared_propensity_ancestry": {
            "run_name": "c20260929_prop_f0",
            "provenance_path": "artifacts/runs/c20260929_prop_f0/provenance.json",
            "provenance_sha256": sha256_bytes(propensity_raw),
            "checkpoint_sha256": EXPECTED_PROPENSITY_SHA256,
            "fit_valid_groups": len(propensity["fit_row_groups"]["valid"]),
            "fit_valid_groups_sha256_sorted_int64le": stable_membership_hash(
                list(map(int, propensity["fit_row_groups"]["valid"]))
            ),
            "evaluation_groups": len(propensity["evaluation_row_groups"]),
            "evaluation_groups_sha256_sorted_int64le": stable_membership_hash(
                list(map(int, propensity["evaluation_row_groups"]))
            ),
            "fit_evaluation_group_overlap": 0,
            "fit_evaluation_component_overlap": 0,
            "note": "Shared mask-propensity ancestor; not a model teacher or blend member.",
        },
        "panel": {
            "unit": "source-qualified connected component within fold0 evaluation valid row groups",
            "source_qualified_id": ["datasets/valid.parquet", "row_group", "seq_ix", "component"],
            "component_method": split["component_method"],
            "evaluation_universe": {
                "components": len(eval_components),
                "valid_row_groups": len(split_eval),
                "groups_sha256_sorted_int64le": stable_membership_hash(list(split_eval)),
                "all_three_member_eval_membership_identical": True,
            },
            "selection": {
                "salt": PANEL_SALT,
                "order": "ascending SHA256(salt|valid|fold=0|component_id), tie by integer component id",
                "selected_components": selected,
                "selected_components_sha256_sorted_int64le": stable_membership_hash(selected),
                "calibration_component_ids": calibration_components,
                "calibration_component_ids_sha256_sorted_int64le": stable_membership_hash(
                    calibration_components
                ),
                "evaluation_component_ids": evaluation_components,
                "evaluation_component_ids_sha256_sorted_int64le": stable_membership_hash(
                    evaluation_components
                ),
                "calibration_groups": calibration_groups,
                "calibration_groups_sha256_sorted_int64le": stable_membership_hash(
                    [int(record["row_group"]) for record in calibration_groups]
                ),
                "evaluation_groups": evaluation_groups,
                "evaluation_groups_sha256_sorted_int64le": stable_membership_hash(
                    [int(record["row_group"]) for record in evaluation_groups]
                ),
                "calibration_evaluation_component_overlap": 0,
                "calibration_evaluation_group_overlap": 0,
                "linked_train_groups_excluded_by_campaign_split": excluded_train_groups,
                "linked_train_groups_excluded_count": len(excluded_train_groups),
            },
            "exposure": "F0 validation was already exposed to model development; this component split is a bounded internal blend screen, not a pristine holdout.",
        },
        "frozen_comparisons": {
            "max_unique_weight_vectors": 11,
            "maximum_scored_candidate_vectors": {
                "calibration": 11,
                "evaluation": 8,
            },
            "evaluation_policy_comparisons": 8,
            "fixed_policies": [
                {"name": "seed3_single", "weights": [1.0, 0.0, 0.0]},
                {"name": "seed4_single", "weights": [0.0, 1.0, 0.0]},
                {"name": "seed5_single", "weights": [0.0, 0.0, 1.0]},
                {"name": "seed3_seed4_equal", "weights": [0.5, 0.5, 0.0]},
                {"name": "seed3_seed5_equal", "weights": [0.5, 0.0, 0.5]},
                {"name": "seed4_seed5_equal", "weights": [0.0, 0.5, 0.5]},
                {"name": "all_three_equal", "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]},
            ],
            "bounded_solver_policy": {
                "name": "shared_two_target_clip2_local_slsqp",
                "starts": ["equal", "seed3_seed4_equal", "regularized_analytic_initializer"],
                "start_count": 3,
                "ridge": 0.1,
                "shrink_to_equal": 0.0,
                "max_iterations_per_start": 50,
                "simplex_constraints": "nonnegative weights summing to one",
                "optimize_on": "calibration components only",
                "candidate_ranking": "calibration global_wp_clip2_v1 official float32 path",
                "gradient": "continuous float64 clip2 surrogate; not the derivative of official float32 quantization",
                "reported_on_evaluation": "only the calibration-selected policy, with weights frozen before evaluation read",
                "candidate_accounting": (
                    "at most 11 unique calibration weight vectors: seven fixed policies, "
                    "the analytic start if distinct, and three solver endpoints if distinct; "
                    "equal/current starts duplicate fixed policies and are deduplicated"
                ),
                "global_optimality_claim": False,
            },
            "evaluation_rule": "Report all seven fixed policies plus the calibration-selected solver policy; do not retune, select, or change weights from evaluation results.",
        },
        "future_execution_contract": {
            "root_admission_required": True,
            "not_executed_in_preparation": True,
            "max_runtime_seconds_per_limited_invocation": 120,
            "static_preparation_memory_max_bytes": 681574400,
            "cpu_quota": "100% of one CPU",
            "swap_bytes": 0,
            "no_gpu_or_provider": True,
            "threads": 1,
            "memory_strategy": "mmap full prediction members and the stored target-cache members; stage only complete calibration groups, then release all stage arrays before a separate evaluation invocation",
            "point_of_use_binding": {
                "required_root_admission": True,
                "prediction_files": "check current device/inode/size/mtime, NPY magic/version/bounded header/shape/dtype, and full SHA against the existing grade/input-validation receipts before mapping; validate all staged fixed-mask values finite",
                "target_cache": "check current device/inode/size/mtime and SHA against the existing official-grade receipt; map only stored y/mask NPY members, validate row_group/seq_ix identity and selected values finite",
                "metric_mask": "is_scored from valid_targets.npz AND row index >= 99, matching the official grade and WARMUP=99",
                "optimizer": "calibration only; exact official float32 scorer ranks candidates; smooth float64 clip-2 gradient is a local search surrogate",
                "evaluation": "separate invocation; fixed weights from the calibration receipt, no solver call and no weight changes",
            },
            "conditional_future_resource_profile": {
                "memory_max_bytes": 1_500_000_000,
                "runtime_max_seconds": 120,
                "cpu_quota": "100% of one CPU",
                "swap_bytes": 0,
                "admission": "pending root review, fresh aggregate-memory check, and explicit root release",
            },
            "calibration_rows": len(calibration_groups) * 20_000,
            "calibration_prediction_stage_bytes": len(calibration_groups) * 20_000 * 3 * 2 * 4,
            "calibration_target_stage_bytes": len(calibration_groups) * 20_000 * 2 * 4,
            "calibration_mask_stage_bytes": len(calibration_groups) * 20_000,
            "evaluation_rows": len(evaluation_groups) * 20_000,
            "evaluation_prediction_stage_bytes": len(evaluation_groups) * 20_000 * 3 * 2 * 4,
            "evaluation_target_stage_bytes": len(evaluation_groups) * 20_000 * 2 * 4,
            "evaluation_mask_stage_bytes": len(evaluation_groups) * 20_000,
            "target_and_mask_columns": [
                "seq_ix",
                "t0",
                "t1",
                "need_prediction",
                "is_scored",
            ],
            "prediction_payload_bytes_loaded_in_preparation": 0,
            "target_payload_bytes_loaded_in_preparation": 0,
            "data_pages_read_in_preparation": 0,
            "resource_profile": {
                "full_sha_recheck_bytes_per_phase": (3 * 299_680_128 + target_archive_stat.st_size),
                "calibration_known_stage_bytes": (
                    len(calibration_groups) * 20_000 * (3 * 2 * 4 + 2 * 4 + 1)
                ),
                "evaluation_known_stage_bytes": (
                    len(evaluation_groups) * 20_000 * (3 * 2 * 4 + 2 * 4 + 1)
                ),
                "calibration_stage_plus_prepared_problem_bytes": (
                    len(calibration_groups)
                    * 20_000
                    * (3 * 2 * 4 + 2 * 4 + 1 + 2 * 8 + 2 * 4 + 3 * 2 * 4 + 2 * 8)
                ),
                "optimizer_workspace_bytes": "not statically bounded; requires admitted execution profile",
                "memory_cap_bytes": 1_500_000_000,
                "wall_cap_seconds": 120,
                "execution_profile_status": "pending root admission; no real arrays read in prep",
            },
        },
    }
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    resource_profile = {
        "schema": "wnn.j1-resource-profile.v1",
        "status": "PREPARED_STATIC_ESTIMATE_PENDING_ROOT_EXECUTION_PROFILE",
        "manifest_sha256": sha256_bytes(manifest_raw),
        "runner_path": runner_relative.as_posix(),
        "runner_sha256": sha256_file(runner_path),
        "phase_memory_estimates": {
            "calibration_groups": len(calibration_groups),
            "calibration_components": len(calibration_components),
            "calibration_stage_bytes": manifest["future_execution_contract"][
                "calibration_prediction_stage_bytes"
            ]
            + manifest["future_execution_contract"]["calibration_target_stage_bytes"]
            + manifest["future_execution_contract"]["calibration_mask_stage_bytes"],
            "calibration_known_stage_plus_prepared_problem_bytes": manifest[
                "future_execution_contract"
            ]["resource_profile"]["calibration_stage_plus_prepared_problem_bytes"],
            "evaluation_groups": len(evaluation_groups),
            "evaluation_components": len(evaluation_components),
            "evaluation_stage_bytes": manifest["future_execution_contract"][
                "evaluation_prediction_stage_bytes"
            ]
            + manifest["future_execution_contract"]["evaluation_target_stage_bytes"]
            + manifest["future_execution_contract"]["evaluation_mask_stage_bytes"],
            "optimizer_workspace_bytes": "unmeasured; requires root-admitted run",
        },
        "per_phase_full_hash_read_bytes": manifest["future_execution_contract"]["resource_profile"][
            "full_sha_recheck_bytes_per_phase"
        ],
        "resource_ceiling": {
            "cpu_quota": "100%",
            "threads": 1,
            "memory_max_bytes": 1_500_000_000,
            "runtime_max_seconds": 120,
            "swap_bytes": 0,
        },
        "command_templates": {
            "calibration": (
                "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 "
                "WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited "
                ".venv/bin/python research/campaign_2026-10-12/joint_endpoint_optimization_r1/"
                "j1_panel_runner_r1.py --phase calibration --manifest "
                "research/campaign-2026-10-12/joint_endpoint_optimization_r1/"
                "J1_COMPONENT_PANEL_MANIFEST_R2.json --root-admission <ROOT_RELEASE> "
                "--output <NEW_CALIBRATION_RECEIPT>"
            ),
            "evaluation": (
                "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 "
                "WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited "
                ".venv/bin/python research/campaign_2026-10-12/joint_endpoint_optimization_r1/"
                "j1_panel_runner_r1.py --phase evaluation --manifest "
                "research/campaign-2026-10-12/joint_endpoint_optimization_r1/"
                "J1_COMPONENT_PANEL_MANIFEST_R2.json --root-admission <ROOT_RELEASE> "
                "--calibration-receipt <FROZEN_CALIBRATION_RECEIPT> "
                "--output <NEW_EVALUATION_RECEIPT>"
            ),
        },
        "validation_scope": {
            "static_source_and_membership_checks": True,
            "actual_optimizer_profile_run": False,
            "real_prediction_or_target_payload_read": False,
            "model_inference_or_fit": False,
            "provider_or_gpu": False,
        },
    }
    resource_raw = (json.dumps(resource_profile, indent=2, sort_keys=True) + "\n").encode()
    validation = {
        "schema": "wnn.j1-static-preparation-validation.v2",
        "status": "PASS_STATIC_METADATA_ONLY",
        "manifest_sha256": sha256_bytes(manifest_raw),
        "resource_profile_sha256": sha256_bytes(resource_raw),
        "checks": {
            "split_manifest_hash_and_source_stats": "PASS",
            "small_component_cache_hashes": "PASS",
            "manifest_graph_component_partition": "PASS",
            "fold0_group_membership": "PASS",
            "candidate_fit_eval_group_and_component_disjointness": "PASS",
            "propensity_fit_eval_group_and_component_disjointness": "PASS",
            "candidate_checkpoint_sidecar_source_bindings": "PASS",
            "actual_grade_source_precision_epoch2_bindings": "PASS",
            "full_prediction_sha_and_current_file_stat_bindings": "PASS",
            "official_grade_fixed_mask_finite_scope_recorded": "PASS",
            "valid_target_cache_index_source_qualified_order": "PASS",
            "bounded_npy_header_and_file_size_checks": "PASS",
        },
        "payload_access": {
            "prediction_payload_read": False,
            "target_payload_read": False,
            "valid_parquet_data_page_read": False,
            "model_checkpoint_loaded": False,
            "blend_fit_or_scoring_run": False,
        },
        "historical_oom_context": {
            "receipt": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/monitor_20261012_1044_r21/NPY_HEADER_OOM_DIAGNOSIS_R1.json",
            "journal": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/monitor_20261012_1044_r21/NPY_HEADER_OOM_JOURNAL_R1.txt",
            "retries": 0,
            "reader": "This builder checks the six-byte NPY magic, supported version, and a 65,536-byte header ceiling before reading the declared header; it never reads array payloads.",
        },
        "npy_header_max_bytes": MAX_NPY_HEADER_BYTES,
        "manifest_bytes": len(manifest_raw),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_exclusive(output_dir / MANIFEST_NAME, manifest_raw)
    write_exclusive(output_dir / RESOURCE_PROFILE_NAME, resource_raw)
    validation_raw = (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive(output_dir / VALIDATION_NAME, validation_raw)
    return manifest, validation


def write_exclusive(path: Path, raw: bytes) -> None:
    """Write a new artifact without replacing an earlier frozen receipt."""
    with path.open("xb") as stream:
        stream.write(raw)


def main() -> None:
    """Build the J1 static panel artifacts from existing metadata only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, validation = build(args.project_root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": validation["status"],
                "manifest_sha256": validation["manifest_sha256"],
                "calibration_components": len(
                    manifest["panel"]["selection"]["calibration_component_ids"]
                ),
                "calibration_groups": len(manifest["panel"]["selection"]["calibration_groups"]),
                "evaluation_components": len(
                    manifest["panel"]["selection"]["evaluation_component_ids"]
                ),
                "evaluation_groups": len(manifest["panel"]["selection"]["evaluation_groups"]),
                "prediction_payload_read": False,
                "target_payload_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
