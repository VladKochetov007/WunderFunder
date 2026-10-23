"""Freeze source identity and run a wider causal Kalman input replay without labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
PROBE = ROOT / "research/campaign_2026-10-18/online_features_r1/kalman_filter_probe"
RECIPE = PACKET / "recipe_r1.json"
PREFLIGHT = PACKET / "preflight_r1.json"
OLD_PREFLIGHT = PROBE / "preflight_receipt_r2.json"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
DATA = ROOT / "datasets/valid.parquet"
CHECKPOINT = ROOT / "artifacts/runs/warm_mechanism_f0_s3_c_r3/model_ep1.pt"
NORMALIZATION = (
    ROOT / "research/campaign_2026-10-18/online_features_r1/kalman_parent_normalization.json"
)
SCRATCH = Path(
    "/home/vlad/development/wunder/research_tmp/online_features_r1/kalman_input_panel_r2"
)
EXECUTION = PACKET / "execution_r1.json"
GRADER = PACKET / "grade_panel.py"
ARMS = ("raw", "local_trend_posterior")
RAW_NAMES = ("i0_p0", "i0_p3", "i1_p0", "i1_p3")
INFERENCE_SOURCES = (
    "src/wnn/gmetric.py",
    "src/wnn/evaluation.py",
    "src/wnn/data.py",
    "src/wnn/features.py",
    "src/wnn/training/environment.py",
    "src/wnn/training/evaluation.py",
    "src/wnn/training/forward.py",
    "src/wnn/training/inputs.py",
    "src/wnn/training/models.py",
    "src/wnn/nn.py",
)
LOCAL_SOURCES = (
    "frozen_gru_probe.py",
    "probe_contract.py",
    "kalman_features.py",
    "kalman_normalization.py",
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to keep peak memory independent of file size."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ints(values: list[int]) -> str:
    """Hash an ordered integer list as compact JSON."""
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def current_cgroup() -> Path:
    """Return this process's unified cgroup directory."""
    entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((line.split("::", 1)[1] for line in entries if line.startswith("0::")), None)
    if unified is None:
        raise RuntimeError("unified cgroup membership is unavailable")
    return Path("/sys/fs/cgroup") / unified.lstrip("/")


def cgroup_values(path: Path) -> dict[str, Any]:
    """Read resource counters while the task scope is still alive."""
    names = (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.swap.current",
        "cpu.max",
    )
    values: dict[str, Any] = {}
    for name in names:
        try:
            value = (path / name).read_text(encoding="utf-8").strip()
        except OSError as error:
            values[name] = {"unavailable": str(error)}
        else:
            values[name] = (
                dict(line.split() for line in value.splitlines())
                if name == "memory.events"
                else value
            )
    return values


def campaign_cgroup() -> tuple[Path | None, dict[str, Any] | None]:
    """Find the campaign parent cgroup for aggregate memory and OOM accounting."""
    directory = current_cgroup()
    for candidate in (directory, *directory.parents):
        if candidate.name == "wnn-campaign.slice":
            return candidate, cgroup_values(candidate)
    return None, None


def source_hashes(recipe: dict[str, Any]) -> dict[str, str]:
    """Pin the actor, grader, reused replay, metric and model input path."""
    sources = {
        "replay_panel.py": Path(__file__).resolve(),
        "grade_panel.py": GRADER,
        "recipe_r1.json": RECIPE,
        "preflight_receipt_r2.json": OLD_PREFLIGHT,
    }
    sources.update({name: PROBE / name for name in LOCAL_SOURCES})
    sources.update({name: ROOT / name for name in INFERENCE_SOURCES})
    pins = recipe["source_pins"]
    sources["resource_profile.json"] = ROOT / pins["resource_profile_path"]
    sources["resource_profile_contract.json"] = ROOT / pins["resource_profile_contract_path"]
    return {name: sha256_file(path) for name, path in sources.items()}


def verify_resource_profile(recipe: dict[str, Any]) -> dict[str, Any]:
    """Bind the RAM/VRAM reservation to a completed representative attempt."""
    pins = recipe["source_pins"]
    profile_path = (ROOT / pins["resource_profile_path"]).resolve(strict=True)
    contract_path = (ROOT / pins["resource_profile_contract_path"]).resolve(strict=True)
    profile_path.relative_to(ROOT.resolve())
    contract_path.relative_to(ROOT.resolve())
    profile_stat = profile_path.stat()
    contract_stat = contract_path.stat()
    if profile_stat.st_size > 1_000_000 or contract_stat.st_size > 1_000_000:
        raise ValueError("resource profile or its recipe exceeds the bounded metadata size")
    profile_sha256 = sha256_file(profile_path)
    contract_sha256 = sha256_file(contract_path)
    if profile_sha256 != pins["resource_profile_sha256"]:
        raise ValueError("measured resource profile differs from the frozen SHA-256 pin")
    if contract_sha256 != pins["resource_profile_contract_sha256"]:
        raise ValueError("resource profile recipe differs from the frozen SHA-256 pin")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    ram_cap = int(recipe["resources"]["task_ram_cap_bytes"])
    vram_cap = int(recipe["resources"]["observed_vram_cap_bytes"])
    if (
        profile.get("id") != "kalman-gru-replay-20261018-r3"
        or profile.get("kind") != "gpu"
        or profile.get("status") != "finished"
        or profile.get("phase") != "terminal"
        or profile.get("exit_status") != 0
        or profile.get("failure") is not None
        or profile.get("contract_sha256") != contract_sha256
        or Path(str(profile.get("contract", ""))).resolve() != contract_path
        or profile.get("peak_ram_bytes", ram_cap + 1) > ram_cap
        or profile.get("peak_observed_vram_bytes", vram_cap + 1) > vram_cap
    ):
        raise ValueError("resource profile is not a successful, in-cap representative replay")
    command = profile.get("command", [])
    has_execute_mode = isinstance(command, list) and any(
        command[index : index + 2] == ["--mode", "execute"]
        for index in range(max(len(command) - 1, 0))
    )
    has_frozen_replay = isinstance(command, list) and any(
        isinstance(argument, str) and Path(argument).name == "frozen_gru_probe.py"
        for argument in command
    )
    if not has_frozen_replay or not has_execute_mode:
        raise ValueError("resource profile command is not the frozen GRU prediction replay")
    return {
        "path": str(profile_path),
        "sha256": profile_sha256,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "attempt_id": profile["id"],
        "status": profile["status"],
        "exit_status": profile["exit_status"],
        "peak_ram_bytes": profile["peak_ram_bytes"],
        "peak_observed_vram_bytes": profile["peak_observed_vram_bytes"],
        "elapsed_seconds": profile["elapsed_seconds"],
        "fits_declared_caps": True,
    }


def selected_membership(recipe: dict[str, Any]) -> dict[str, Any]:
    """Derive and verify the source-qualified prefix without reading data rows."""
    split_bytes = SPLIT.read_bytes()
    split = json.loads(split_bytes)
    if hashlib.sha256(split_bytes).hexdigest() != recipe["membership"]["split_manifest_sha256"]:
        raise ValueError("fold-0 split bytes differ from the frozen recipe")
    ordered = list(map(int, split["evaluation"]["valid"]))
    count = int(recipe["membership"]["groups"])
    groups = ordered[:count]
    if len(ordered) != 468 or len(groups) != count or len(set(groups)) != count:
        raise ValueError("unexpected fold-0 evaluation membership")
    rows = {int(row["row_group"]): row for row in split["sequences"]["valid"]}
    if any(group not in rows for group in groups):
        raise ValueError("fold-0 split lacks source-qualified sequence metadata")
    sequence_ids = [int(rows[group]["seq_ix"]) for group in groups]
    components = [int(rows[group]["component"]) for group in groups]
    checks = (
        ("group_order_sha256", digest_ints(groups)),
        ("sequence_id_order_sha256", digest_ints(sequence_ids)),
        ("component_order_sha256", digest_ints(components)),
    )
    for key, actual in checks:
        if actual != recipe["membership"][key]:
            raise ValueError(f"source-qualified {key} differs from the frozen recipe")
    if len(set(components)) != int(recipe["membership"]["unique_components"]):
        raise ValueError("source component count differs from the frozen recipe")
    return {
        "groups": groups,
        "sequence_ids": sequence_ids,
        "components": components,
        "group_order_sha256": checks[0][1],
        "sequence_id_order_sha256": checks[1][1],
        "component_order_sha256": checks[2][1],
        "unique_components": len(set(components)),
        "selection_rule": recipe["membership"]["selection_rule"],
    }


def verify_identity_and_step_order(
    parquet: pq.ParquetFile, membership: dict[str, Any], rows_per_sequence: int
) -> dict[str, Any]:
    """Check sequence IDs and row order using only the two identity columns."""
    digest = hashlib.sha256()
    expected_steps = np.arange(rows_per_sequence, dtype=np.int64)
    for group, expected_sequence in zip(
        membership["groups"], membership["sequence_ids"], strict=True
    ):
        table = parquet.read_row_group(
            int(group), columns=["seq_ix", "step_in_seq"], use_threads=False
        )
        if table.num_rows != rows_per_sequence:
            raise ValueError(
                f"row group {group} has {table.num_rows} rows, expected {rows_per_sequence}"
            )
        sequence_ids = table["seq_ix"].to_numpy()
        steps = table["step_in_seq"].to_numpy()
        if not np.all(sequence_ids == expected_sequence):
            raise ValueError(f"seq_ix does not match frozen membership at row group {group}")
        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"step_in_seq is not exact 0..{rows_per_sequence - 1} order at group {group}"
            )
        digest.update(np.asarray([group, expected_sequence], dtype="<i8").tobytes())
        digest.update(np.asarray(sequence_ids, dtype="<i8").tobytes())
        digest.update(np.asarray(steps, dtype="<i8").tobytes())
        del table, sequence_ids, steps
    return {
        "columns_read": ["seq_ix", "step_in_seq"],
        "groups_verified": len(membership["groups"]),
        "rows_per_group": rows_per_sequence,
        "rows_verified": len(membership["groups"]) * rows_per_sequence,
        "identity_and_step_order_sha256": digest.hexdigest(),
        "targets_or_masks_read": False,
    }


def verify_parquet_footer(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Verify the frozen footer and trailer using bounded-memory reads."""
    expected_size = int(expected["footer_bytes"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_size = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_size != expected_size:
            raise ValueError(
                "parquet footer size or trailer differs from the frozen split manifest"
            )
        stream.seek(-(footer_size + 8), os.SEEK_END)
        remaining = footer_size + 8
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("valid.parquet footer is truncated")
            digest.update(block)
            remaining -= len(block)
    actual = {"footer_bytes": footer_size, "footer_sha256": digest.hexdigest()}
    if actual["footer_sha256"] != expected["footer_sha256"]:
        raise ValueError("valid.parquet footer digest differs from the frozen split manifest")
    return actual


def static_contract() -> dict[str, Any]:
    """Validate source identity and write a label-blind row-order contract."""
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if recipe.get("schema") != "wnn.kalman_input_panel_recipe.v1":
        raise ValueError("unsupported or missing frozen recipe")
    resource_profile = verify_resource_profile(recipe)
    if sha256_file(OLD_PREFLIGHT) != recipe["source_pins"]["prior_preflight_sha256"]:
        raise ValueError("prior 16-group preflight changed")
    prior = json.loads(OLD_PREFLIGHT.read_text(encoding="utf-8"))
    expected_prior_sources = prior["source_hashes"]
    for name, expected in expected_prior_sources.items():
        source = PROBE / name if (PROBE / name).is_file() else ROOT / name
        if sha256_file(source) != expected:
            raise ValueError(f"previously replayed source changed: {name}")

    membership = selected_membership(recipe)
    source = json.loads(SPLIT.read_text(encoding="utf-8"))["sources"]["datasets/valid.parquet"]
    stat = DATA.stat()
    if stat.st_size != int(source["bytes"]) or stat.st_mtime_ns != int(source["mtime_ns"]):
        raise ValueError("valid.parquet identity differs from the frozen source manifest")
    footer_identity = verify_parquet_footer(DATA, source)
    parquet = pq.ParquetFile(DATA)
    names = parquet.schema_arrow.names
    raw_names = names[4:116]
    if parquet.num_row_groups != 1873 or len(raw_names) != 112:
        raise ValueError("valid.parquet schema or row-group count differs from the source contract")
    columns = [raw_names[index] for index in (0, 3, 52, 55)]
    if columns != list(RAW_NAMES):
        raise ValueError("selected raw feature order differs from the frozen recipe")
    if not {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}.issubset(names):
        raise ValueError("valid.parquet omits required inference or grading columns")
    identity_order = verify_identity_and_step_order(
        parquet, membership, int(recipe["inference"]["all_rows_processed"])
    )

    normalization_bytes = NORMALIZATION.read_bytes()
    normalization = json.loads(normalization_bytes)
    if normalization["source_checkpoint_sha256"] != recipe["parent"]["checkpoint_sha256"]:
        raise ValueError("normalization record names another parent checkpoint")
    if sha256_file(CHECKPOINT) != recipe["parent"]["checkpoint_sha256"]:
        raise ValueError("parent checkpoint hash differs from the frozen recipe")
    old_normalization = prior["normalization"]
    normalization_sha = hashlib.sha256(normalization_bytes).hexdigest()
    if normalization_sha != old_normalization["sha256"]:
        raise ValueError("parent normalization differs from the prior replay preflight")

    sources = source_hashes(recipe)
    campaign_path, campaign = campaign_cgroup()
    return {
        "schema": "wnn.kalman_input_panel_preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass_identity_step_order_targets_masks_blind",
        "recipe_sha256": sha256_file(RECIPE),
        "source_hashes": sources,
        "resource_profile": resource_profile,
        "prior_preflight_sha256": sha256_file(OLD_PREFLIGHT),
        "split_manifest_sha256": recipe["membership"]["split_manifest_sha256"],
        "data": {
            "path": str(DATA),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "row_groups": parquet.num_row_groups,
            "raw_feature_count": len(raw_names),
            "selected_raw_columns": columns,
            "parquet_footer_metadata_read": True,
            **footer_identity,
            "identity_columns_only_read_for_selected_groups": True,
        },
        "parent": {
            **recipe["parent"],
            "normalization_sha256": normalization_sha,
            "loaded": False,
        },
        "membership": membership,
        "identity_and_step_order": identity_order,
        "arms": list(ARMS),
        "targets_or_masks_numeric_values_read": False,
        "torch_imported": False,
        "gpu_used": False,
        "campaign_parent_cgroup": {"path": str(campaign_path), "values": campaign},
    }


def write_preflight() -> dict[str, Any]:
    """Write immutable identity preflight before inference or target/mask reads."""
    payload = static_contract()
    with PREFLIGHT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def verify_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck every source and membership pin before loading the model."""
    frozen = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    fresh = static_contract()
    keys = (
        "recipe_sha256",
        "source_hashes",
        "resource_profile",
        "prior_preflight_sha256",
        "split_manifest_sha256",
        "data",
        "parent",
        "membership",
        "arms",
    )
    if frozen.get("status") != "pass_identity_step_order_targets_masks_blind" or any(
        frozen.get(key) != fresh[key] for key in (*keys, "identity_and_step_order")
    ):
        raise ValueError("frozen preflight no longer matches sources, data metadata, or membership")
    return frozen, json.loads(OLD_PREFLIGHT.read_text(encoding="utf-8"))


def write_cache(
    output_dir: Path, arm: str, values: np.ndarray, preflight: dict[str, Any]
) -> dict[str, Any]:
    """Atomically persist one prediction cache and its identity sidecar."""
    cache = output_dir / f"{arm}.npy"
    temporary = cache.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, cache)
    receipt = {
        "schema": "wnn.kalman_input_panel_prediction.v1",
        "arm": arm,
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "cache_bytes": cache.stat().st_size,
        "prediction_shape": list(values.shape),
        "prediction_dtype": "float32",
        "group_order_sha256": preflight["membership"]["group_order_sha256"],
        "sequence_id_order_sha256": preflight["membership"]["sequence_id_order_sha256"],
        "groups": preflight["membership"]["groups"],
        "sequence_ids": preflight["membership"]["sequence_ids"],
        "checkpoint_sha256": preflight["parent"]["checkpoint_sha256"],
        "normalization_sha256": preflight["parent"]["normalization_sha256"],
        "preflight_sha256": sha256_file(PREFLIGHT),
        "source_hashes": preflight["source_hashes"],
        "targets_or_masks_loaded": False,
        "official_metric_called": False,
    }
    receipt_path = output_dir / f"{arm}.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def execute(output_dir: Path) -> dict[str, Any]:
    """Run the two frozen prediction arms under the admitted one-GPU scope."""
    preflight, prior = verify_preflight()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite prediction artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROBE))
    replay = importlib.import_module("frozen_gru_probe")
    replay.ARMS = ARMS

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    runtime = replay.load_runtime(prior)
    values = {
        arm: np.empty((len(preflight["membership"]["groups"]), 20_000, 2), dtype=np.float32)
        for arm in ARMS
    }
    for index, (group, sequence_id) in enumerate(
        zip(preflight["membership"]["groups"], preflight["membership"]["sequence_ids"], strict=True)
    ):
        predictions = replay.predict_group(runtime, group, sequence_id)
        for arm in ARMS:
            values[arm][index] = predictions[arm]
        if (index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "completed_groups": index + 1,
                        "total_groups": len(preflight["membership"]["groups"]),
                    }
                ),
                flush=True,
            )

    arm_receipts = [write_cache(output_dir, arm, values[arm], preflight) for arm in ARMS]
    torch = runtime["torch"]
    torch.cuda.synchronize(runtime["device"])
    execution = {
        "schema": "wnn.kalman_input_panel_execution.v1",
        "status": "prediction_only_complete",
        "experiment_id": "E-KALMAN-INPUT-EXPANDED-F0-20261018-R1",
        "recipe_sha256": preflight["recipe_sha256"],
        "preflight_sha256": sha256_file(PREFLIGHT),
        "source_hashes": preflight["source_hashes"],
        "parent_checkpoint_sha256": preflight["parent"]["checkpoint_sha256"],
        "groups": len(preflight["membership"]["groups"]),
        "unique_components": preflight["membership"]["unique_components"],
        "group_order_sha256": preflight["membership"]["group_order_sha256"],
        "sequence_id_order_sha256": preflight["membership"]["sequence_id_order_sha256"],
        "component_order_sha256": preflight["membership"]["component_order_sha256"],
        "identity_and_step_order_sha256": preflight["identity_and_step_order"][
            "identity_and_step_order_sha256"
        ],
        "arms": arm_receipts,
        "elapsed_wall_seconds": time.perf_counter() - wall_start,
        "elapsed_cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": runtime["torch_version"],
        "cuda_device": runtime["cuda_device"],
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime["device"])),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime["device"])),
        "task_cgroup": cgroup_values(current_cgroup()),
        "campaign_parent_cgroup": {
            "path": str(campaign_cgroup()[0]),
            "values": campaign_cgroup()[1],
        },
        "targets_or_masks_loaded": False,
        "official_metric_called": False,
    }
    with EXECUTION.open("x", encoding="utf-8") as stream:
        json.dump(execution, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return execution


def main() -> None:
    """Run an identity-only preflight or the explicitly admitted replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--output-dir", type=Path, default=SCRATCH)
    args = parser.parse_args()
    result = write_preflight() if args.mode == "preflight" else execute(args.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
