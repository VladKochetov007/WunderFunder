"""Replay one frozen quota candidate on a fixed, feature-only sequence panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(REPLAY)]
from check_submission_abi import inspect_artifact  # noqa: E402
from native_runtime import extract_safe, load_callback, run_sequence, sha256_file  # noqa: E402

from utils import FEATURE_COLUMNS, DataPoint, validate_sequence  # noqa: E402


def _selection_rank(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def select_members(split_entries: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Choose32 components per pinned fold, then one input sequence per component."""
    selected: list[dict[str, int]] = []
    if {entry["fold"] for entry in split_entries} != {0, 2} or len(split_entries) != 2:
        raise ValueError("quota panel must use exactly F0 and F2")
    for entry in sorted(split_entries, key=lambda item: item["fold"]):
        split_path = ROOT / entry["path"]
        if sha256_file(split_path) != entry["sha256"]:
            raise ValueError("quota split identity changed")
        split = json.loads(split_path.read_text())
        if split["fold"] != entry["fold"]:
            raise ValueError("quota split fold label changed")
        valid = {int(item["row_group"]): item for item in split["sequences"]["valid"]}
        components: dict[int, list[int]] = {}
        for group in split["evaluation"]["valid"]:
            group = int(group)
            components.setdefault(int(valid[group]["component"]), []).append(group)
        chosen = sorted(
            components,
            key=lambda comp: _selection_rank(f"20261024:fold{entry['fold']}:component{comp}"),
        )[:32]
        if len(chosen) != 32:
            raise ValueError("quota fold has fewer than32 components")
        for component in chosen:
            group = min(
                components[component], key=lambda value: _selection_rank(f"20261024:valid:{value}")
            )
            selected.append(
                {
                    "fold": entry["fold"],
                    "component": component,
                    "row_group": group,
                    "seq_ix": int(valid[group]["seq_ix"]),
                }
            )
    if len({item["row_group"] for item in selected}) != 64:
        raise ValueError("quota source-qualified groups overlap across folds")
    if len({item["seq_ix"] for item in selected}) != 64:
        raise ValueError("quota sequence IDs must be distinct for callback reset")
    return selected


def load_plan(path: Path, expected_sha: str, candidate: str) -> dict[str, Any]:
    """Require exact package, source and raw-file identities before feature access."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("the original TL3 native extension requires Python3.11")
    if sha256_file(path) != expected_sha:
        raise ValueError("quota replay plan hash changed")
    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "wnn.quota-native-replay-plan.v1"
        or plan["status"] != "admitted"
        or set(plan["candidates"]) != {"baseline", "s3", "s3_s4"}
        or candidate not in plan["candidates"]
    ):
        raise ValueError("quota replay is not admitted for this candidate")
    for relative, digest in plan["source_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"quota replay source changed: {relative}")
    for split in plan["split_manifests"]:
        if sha256_file(ROOT / split["path"]) != split["sha256"]:
            raise ValueError("quota replay split identity changed")
    entry = plan["candidates"][candidate]
    archive = ROOT / entry["path"]
    if sha256_file(archive) != entry["sha256"] or archive.stat().st_size != entry["bytes"]:
        raise ValueError("quota ZIP identity changed")
    raw = ROOT / plan["raw_source"]["path"]
    stat = raw.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        plan["raw_source"]["bytes"],
        plan["raw_source"]["mtime_ns"],
    ):
        raise ValueError("raw validation file stat identity changed")
    if plan["members"] != select_members(plan["split_manifests"]):
        raise ValueError("quota members differ from the deterministic32+32 component panel")
    return plan


def replay(plan: dict[str, Any], candidate: str, output: Path) -> dict[str, Any]:
    """Run complete sequences, require reset equality, and save immutable predictions."""
    started = time.monotonic()
    entry = plan["candidates"][candidate]
    archive = ROOT / entry["path"]
    abi = inspect_artifact(archive)
    if not abi["passed"]:
        raise ValueError("quota ZIP failed native ABI inspection")
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(64, 20_000, 2)
    )
    parquet = pq.ParquetFile(ROOT / plan["raw_source"]["path"])
    if parquet.num_row_groups != plan["raw_source"]["row_groups"]:
        raise ValueError("validation Parquet row-group count changed")
    records: list[dict[str, Any]] = []
    first: tuple[int, np.ndarray, np.ndarray] | None = None
    with tempfile.TemporaryDirectory(prefix="package-", dir=output) as temporary:
        package = Path(temporary)
        with zipfile.ZipFile(archive) as stream:
            if stream.testzip() is not None:
                raise ValueError("quota ZIP failed CRC verification")
            extract_safe(stream, package)
        _, model, native_path = load_callback(package)
        if model.member_count != entry["member_count"]:
            raise ValueError("native member count differs from frozen endpoint")
        native_sha = sha256_file(native_path)
        relative_native = str(native_path.relative_to(package))
        if native_sha != entry["native_sha256"][relative_native]:
            raise ValueError("loaded native library differs from pinned ZIP entry")
        for index, member in enumerate(plan["members"]):
            table = parquet.read_row_group(
                member["row_group"],
                columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
                use_threads=False,
            )
            sequence, need = validate_sequence(table)
            if sequence != member["seq_ix"]:
                raise ValueError("source-qualified sequence identity changed")
            features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
                np.float32, copy=False
            )
            if features.shape != (20_000, 112) or not np.isfinite(features).all():
                raise ValueError("quota replay requires finite complete sequences")
            values, warmup, predicted, elapsed = run_sequence(
                model, DataPoint, sequence, features, need
            )
            predictions[index] = values
            records.append(
                {
                    **member,
                    "warmup_rows": warmup,
                    "predicted_rows": predicted,
                    "callback_wall_ns": elapsed,
                    "feature_sha256": hashlib.sha256(memoryview(features).cast("B")).hexdigest(),
                    "need_prediction_sha256": hashlib.sha256(need.tobytes()).hexdigest(),
                }
            )
            if index == 0:
                first = (sequence, features.copy(), need.copy())
            if index == 1:
                if first is None:
                    raise RuntimeError("reset check lost its first sequence")
                repeated = run_sequence(model, DataPoint, *first)[0]
                if not np.array_equal(predictions[0].view(np.uint32), repeated.view(np.uint32)):
                    raise ValueError("quota callback failed A-B-A bitwise reset")
            if (index + 1) % 8 == 0:
                print(json.dumps({"candidate": candidate, "sequences": index + 1}), flush=True)
    predictions.flush()
    del predictions
    if sha256_file(archive) != entry["sha256"]:
        raise ValueError("quota ZIP changed during replay")
    return {
        "status": "COMPLETE_FEATURE_ONLY_NATIVE_REPLAY",
        "candidate": candidate,
        "zip_sha256": entry["sha256"],
        "abi": abi,
        "python_version": sys.version,
        "native_library": relative_native,
        "native_sha256": native_sha,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "shape": [64, 20_000, 2],
        "dtype": "<f4",
        "members": records,
        "reset_aba_bitwise": True,
        "target_values_read": False,
        "scoring_masks_read": False,
        "quality_score_computed": False,
        "wall_seconds": time.monotonic() - started,
        "limits": "Exposed source-qualified sample; all-data/unknown ancestry. Local callback timing is not competition runtime. Raw file stat checked; historical full-file digest is not rehashed here.",
    }


def main() -> None:
    """Execute a root-admitted exact-ZIP feature replay with create-only outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan, args.plan_sha256, args.candidate)
    report = replay(plan, args.candidate, args.output)
    report["plan_sha256"] = args.plan_sha256
    with (args.output / "REPLAY_RECEIPT_R1.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "status": report["status"],
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
