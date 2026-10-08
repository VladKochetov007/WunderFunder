#!/usr/bin/env python3
"""Independently bind a served head-merge ZIP to two source checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

# Checkpoint and package metadata have source-defined JSON schemas.
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(
    root: Path,
    run_at: str,
    *,
    core_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], Path, dict[str, Any]]:
    run, marker, epoch = run_at.partition("@ep")
    if not marker or not epoch.isdigit():
        raise ValueError(f"source must be RUN@epN: {run_at}")
    path = root / "artifacts/runs" / run / f"model_ep{epoch}.pt"
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if loaded.get("args", {}).get("readout_activation", "gelu") != "gelu":
        raise ValueError(f"unsupported readout activation in {run_at}")
    if loaded.get("sentinel_preprocessing"):
        raise ValueError(f"unsupported sentinel preprocessing in {run_at}")
    state = {
        name: np.ascontiguousarray(value.detach().cpu().float().numpy())
        for name, value in loaded["model"].items()
    }
    provenance_path = root / "artifacts/runs" / run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checkpoint_hash = sha256(path)
    expected_hash = provenance.get("checkpoint_sha256", {}).get(path.name)
    provenance_binding = "matched"
    if expected_hash is not None and expected_hash != checkpoint_hash:
        derivation = provenance.get("core_derivation")
        if not isinstance(derivation, dict) or core_audit is None:
            raise ValueError(f"{run_at}: provenance checkpoint hash mismatch")
        source_run = derivation.get("source_run")
        source_epoch = derivation.get("source_epoch")
        source_path = root / "artifacts/runs" / str(source_run) / f"model_ep{source_epoch}.pt"
        source_provenance_path = source_path.parent / "provenance.json"
        source_provenance = json.loads(source_provenance_path.read_text())
        audit_matches = (
            core_audit.get("status") == "passed"
            and core_audit.get("destination_run") == run
            and core_audit.get("epoch") == int(epoch)
            and core_audit.get("source_run") == source_run
            and core_audit.get("source_checkpoint_sha256") == sha256(source_path)
            and core_audit.get("destination_checkpoint_sha256") == checkpoint_hash
            and core_audit.get("destination_provenance_sha256") == sha256(provenance_path)
            and expected_hash == source_provenance.get("checkpoint_sha256", {}).get(path.name)
        )
        if not audit_matches:
            raise ValueError(f"{run_at}: stale provenance hash lacks matching independent core audit")
        provenance_binding = "stale_core_source_hash_in_checkpoint_map"
    return (
        state,
        path,
        {
            "run_at": run_at,
            "checkpoint": str(path.relative_to(root)),
            "checkpoint_sha256": checkpoint_hash,
            "provenance_checkpoint_hash_binding": provenance_binding,
            "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
            "provenance": str(provenance_path.relative_to(root)),
            "provenance_sha256": sha256(provenance_path),
            "fit_group_counts": {
                split: len(groups) for split, groups in provenance.get("fit_row_groups", {}).items()
            },
            "split_manifest_sha256": provenance.get("split_manifest_sha256"),
            "args": provenance.get("args", {}),
        },
    )


def merged(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(first) != set(second):
        raise ValueError("source checkpoints have different tensor names")
    changed = [
        name
        for name, value in first.items()
        if not name.startswith("head.") and not np.array_equal(value, second[name])
    ]
    if changed:
        raise ValueError(f"head-only fine-tune changed non-head tensors: {changed[:8]}")
    out = {name: value for name, value in first.items() if not name.startswith("head.")}
    out["head.0.weight"] = np.concatenate([first["head.0.weight"], second["head.0.weight"]], axis=0)
    out["head.0.bias"] = np.concatenate([first["head.0.bias"], second["head.0.bias"]], axis=0)
    out["head.2.weight"] = np.concatenate(
        [first["head.2.weight"][:2], second["head.2.weight"][:2]], axis=1
    ) / np.float32(2.0)
    out["head.2.bias"] = (first["head.2.bias"][:2] + second["head.2.bias"][:2]) / np.float32(2.0)
    return {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="exactly two RUN@epN sources: core first, head-fine-tune second",
    )
    parser.add_argument(
        "--native-reference",
        type=Path,
        required=True,
        help="incumbent package used for the unchanged solution/native files",
    )
    parser.add_argument(
        "--core-audit",
        type=Path,
        required=True,
        help="independent audit receipt for the first source's family-head-free derivation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source) != 2:
        parser.error("exactly two --source values are required")
    root = args.root.resolve()
    archive_path = args.zip.resolve()
    native_reference = args.native_reference.resolve()
    core_audit_path = args.core_audit.resolve()
    core_audit = json.loads(core_audit_path.read_text())
    first, first_path, first_receipt = checkpoint(root, args.source[0], core_audit=core_audit)
    second, second_path, second_receipt = checkpoint(root, args.source[1])
    if second_receipt["args"].get("init") != args.source[0]:
        raise ValueError(
            "head-fine-tune provenance does not name the exact candidate core initializer"
        )
    if second_receipt["initial_checkpoint_sha256"] != first_receipt["checkpoint_sha256"]:
        raise ValueError("head-fine-tune initial checkpoint hash does not bind to the exact core bytes")
    freeze = second_receipt["args"].get("freeze", "")
    if not isinstance(freeze, str) or "inp" not in freeze or "gru" not in freeze:
        raise ValueError("head-fine-tune provenance does not freeze the input projection and GRU")
    if first["head.2.weight"].shape[0] != 23 or second["head.2.weight"].shape[0] != 23:
        raise ValueError("source pair does not use the family-free 23-row primary head")
    expected = merged(first, second)
    if first_path == second_path:
        raise ValueError("candidate sources must be distinct checkpoints")

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        names = set(archive.namelist())
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("ZIP contains an unsafe member path")
        members = json.loads(archive.read("members.json"))
        if members != [
            {
                "weights": "weights_0.npz",
                "coef": 1.0,
                "source": "|".join(args.source),
            }
        ]:
            raise ValueError(f"ZIP member declaration differs from fixed endpoint: {members}")
        if archive.read("precision.txt") != b"i16\n":
            raise ValueError("ZIP does not select the fixed i16 serving precision")
        with np.load(io.BytesIO(archive.read("weights_0.npz")), allow_pickle=False) as stored:
            actual = {name: stored[name] for name in stored.files}
    if set(actual) != set(expected):
        raise ValueError("ZIP tensor names differ from independently merged source tensors")
    mismatches = [name for name in expected if not np.array_equal(actual[name], expected[name])]
    if mismatches:
        raise ValueError(f"ZIP tensors differ from exact equal head merge: {mismatches[:8]}")
    if any(not np.isfinite(value).all() for value in actual.values()):
        raise ValueError("ZIP includes a non-finite weight tensor")

    with zipfile.ZipFile(native_reference) as base, zipfile.ZipFile(archive_path) as candidate:
        native_hashes = {}
        for name in (
            "solution.py",
            "v2/wnn_native.abi3.so",
            "avx2/wnn_native.abi3.so",
            "v3/wnn_native.abi3.so",
        ):
            reference_bytes = base.read(name)
            candidate_bytes = candidate.read(name)
            if candidate_bytes != reference_bytes:
                raise ValueError(f"package changed incumbent serving code: {name}")
            native_hashes[name] = hashlib.sha256(candidate_bytes).hexdigest()
    if archive_path.stat().st_size > 20_000_000:
        raise ValueError("ZIP exceeds the 20 MB submission limit")

    report = {
        "schema": "wnn.independent-package-tensor-audit.v1",
        "status": "passed",
        "zip": str(archive_path),
        "zip_sha256": sha256(archive_path),
        "zip_bytes": archive_path.stat().st_size,
        "sources": [
            {key: value for key, value in first_receipt.items() if key != "args"},
            {key: value for key, value in second_receipt.items() if key != "args"},
        ],
        "head_finetune_initializer_matches_core": True,
        "head_finetune_initial_checkpoint_hash_matches_core": True,
        "source_provenance_checkpoint_hash_binding": first_receipt[
            "provenance_checkpoint_hash_binding"
        ],
        "independent_core_audit_sha256": sha256(core_audit_path),
        "head_finetune_freezes_input_and_recurrent_trunk": True,
        "member_declaration": members,
        "precision": "i16",
        "tensor_count": len(actual),
        "all_exported_tensors_bitwise_equal_to_exact_equal_head_merge": True,
        "all_non_head_tensors_in_source_pair_bitwise_equal": True,
        "source_head_merge_coefficients": [0.5, 0.5],
        "native_code_sha256": native_hashes,
        "zip_crc_passed": True,
        "under_20mb": True,
        "limitations": [
            "This binds ZIP tensors and code to source checkpoints; ABI, Python 3.11 callback, reset, metric, and runtime gates are separate.",
            "Source provenance records fit ancestry but does not establish an unexposed evaluation claim for all-data fits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "zip_sha256": report["zip_sha256"]}))


if __name__ == "__main__":
    main()
