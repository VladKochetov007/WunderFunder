"""Verify the frozen H388 propensity checkpoint, caches, and fold ancestry."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ANCESTRY_ORIGINAL_PATHS = {
    "training_job": "research/campaign_2026-09-29/jobs/c20260929_prop_f1.json",
    "training_log": "research/campaign_2026-09-29/logs/c20260929_prop_f1.log",
}
EXPECTED_FILE_ROLES = {
    "model": "artifacts/runs/c20260929_prop_f1/model.pt",
    "provenance": "artifacts/runs/c20260929_prop_f1/provenance.json",
    "train_cache": "cache/propcache_c20260929_prop_f1_train.npy",
    "train_sidecar": "cache/propcache_c20260929_prop_f1_train.json",
    "valid_cache": "cache/propcache_c20260929_prop_f1_valid.npy",
    "valid_sidecar": "cache/propcache_c20260929_prop_f1_valid.json",
}


def sha256_file(path: Path) -> str:
    """Hash a file with a fixed memory bound."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed on a mismatch in a mounted input or ancestry receipt."""
    if not condition:
        raise ValueError(message)


def read_npy_header(path: Path) -> tuple[tuple[int, ...], str, bool, int]:
    """Read an NPY header without importing NumPy or loading array data."""
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        if version == b"\x01\x00":
            length = int.from_bytes(stream.read(2), "little")
        elif version in (b"\x02\x00", b"\x03\x00"):
            length = int.from_bytes(stream.read(4), "little")
        else:
            raise ValueError(f"unsupported NPY version {version!r}: {path}")
        require(length <= 1_000_000, f"unreasonable NPY header length: {path}")
        raw = stream.read(length)
        require(len(raw) == length, f"truncated NPY header: {path}")
        header = ast.literal_eval(raw.decode("latin1").strip())
        offset = stream.tell()
    return tuple(header["shape"]), str(header["descr"]), bool(header["fortran_order"]), offset


def verify_ancestry(
    project_root: Path, expected: dict[str, Any], files: dict[str, dict[str, Any]]
) -> dict[str, object]:
    """Verify the finished fold-1 propensity job, model provenance, and fit membership."""
    receipts = expected["ancestry_receipts"]
    observed_receipts: dict[str, object] = {}
    for role, receipt in receipts.items():
        if role in EXPECTED_ANCESTRY_ORIGINAL_PATHS:
            require(
                receipt.get("original_path") == EXPECTED_ANCESTRY_ORIGINAL_PATHS[role],
                f"{role} original receipt path differs",
            )
        path = project_root / receipt["path"]
        require(path.is_file(), f"missing fold-1 ancestry receipt {role}: {path}")
        require(path.stat().st_size == receipt["bytes"], f"{role} receipt byte count differs")
        digest = sha256_file(path)
        require(digest == receipt["sha256"], f"{role} receipt digest differs")
        observed_receipts[role] = {
            "path": receipt["path"],
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    split_receipt = receipts["split_manifest"]
    split: dict[str, Any] = json.loads((project_root / split_receipt["path"]).read_text())
    require(
        split.get("fold") == 1 and split.get("n_folds") == 4, "propensity split is not fold 1 of 4"
    )
    job_receipt = receipts["training_job"]
    job: dict[str, Any] = json.loads((project_root / job_receipt["path"]).read_text())
    require(job.get("name") == "c20260929_prop_f1", "propensity training job identity differs")
    require(
        job.get("status") == "finished" and job.get("exit_code") == 0,
        "propensity job did not finish cleanly",
    )
    command = job.get("command", [])
    for flag, value in (
        ("--name", "c20260929_prop_f1"),
        ("--heldout-fold", "1"),
        ("--split-manifest", "research/campaign_2026-09-29/artifacts/splits/fold1.json"),
        ("--checkpoint-policy", "fixed"),
        ("--checkpoint-epoch", "3"),
    ):
        require(
            flag in command and command[command.index(flag) + 1] == value,
            f"propensity job flag differs: {flag}",
        )
    require(
        job.get("source_sha256", {}).get("scripts/train_propensity.py")
        == expected["ancestry"]["train_propensity_source_sha256"],
        "propensity trainer source hash differs",
    )

    provenance: dict[str, Any] = json.loads(
        (project_root / files["provenance"]["path"]).read_text()
    )
    args = provenance.get("args", {})
    split_sha = expected["split_manifest_sha256"]
    require(
        provenance.get("split_manifest_sha256") == split_sha,
        "propensity provenance split hash differs",
    )
    require(args.get("heldout_fold") == 1, "propensity provenance does not hold out fold 1")
    require(
        args.get("name") == "c20260929_prop_f1" and args.get("seed") == 0,
        "propensity provenance identity differs",
    )
    require(
        args.get("epochs") == 4 and args.get("checkpoint_epoch") == 3,
        "propensity epoch recipe differs",
    )
    require(args.get("split_manifest_sha256") == split_sha, "propensity args split hash differs")
    require(
        str(args.get("split_manifest", "")).endswith(
            "research/campaign_2026-09-29/artifacts/splits/fold1.json"
        ),
        "propensity split path differs",
    )
    require(
        provenance.get("checkpoint_sha256", {}).get("model.pt") == files["model"]["sha256"],
        "propensity provenance checkpoint hash differs from model",
    )
    fit_valid = set(provenance.get("fit_row_groups", {}).get("valid", []))
    evaluation_valid = set(provenance.get("evaluation_row_groups", []))
    require(
        fit_valid == set(split["fit"]["valid"]),
        "propensity fit groups differ from fold-1 fit membership",
    )
    require(
        evaluation_valid == set(split["evaluation"]["valid"]),
        "propensity evaluation groups differ from fold-1 evaluation membership",
    )
    require(not fit_valid & evaluation_valid, "propensity fit and evaluation groups overlap")
    return {
        "receipts": observed_receipts,
        "fold": 1,
        "fit_valid_groups": len(fit_valid),
        "evaluation_valid_groups": len(evaluation_valid),
    }


def validate(project_root: Path, manifest_path: Path, manifest_sha256: str) -> dict[str, object]:
    """Check the six frozen propensity files, cache layouts, and fold ancestry."""
    require(manifest_path.is_file(), f"propensity input manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    require(
        hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha256,
        "propensity input manifest hash differs",
    )
    expected: dict[str, Any] = json.loads(manifest_bytes)
    require(
        expected.get("schema") == "wnn.h388-prop-inputs.v1", "unknown propensity manifest schema"
    )
    require(
        expected.get("propensity_id") == "c20260929_prop_f1", "unexpected propensity model identity"
    )
    require(
        expected.get("split_manifest_sha256")
        == "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
        "fold-1 split identity differs",
    )
    files: dict[str, dict[str, Any]] = expected["files"]
    require(
        {entry["path"] for entry in files.values()} == set(EXPECTED_FILE_ROLES.values()),
        "propensity file inventory differs",
    )
    observed_files: dict[str, dict[str, object]] = {}
    for role, relative_path in EXPECTED_FILE_ROLES.items():
        item = files[role]
        require(item["path"] == relative_path, f"{role} input path differs")
        path = project_root / relative_path
        require(path.is_file(), f"required propensity input is missing: {path}")
        require(
            path.stat().st_size == item["bytes"],
            f"propensity input byte count differs: {relative_path}",
        )
        digest = sha256_file(path)
        require(digest == item["sha256"], f"propensity input hash differs: {relative_path}")
        observed_files[role] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    model_sha256 = files["model"]["sha256"]
    for split_name, expected_rows, expected_filled in (
        ("train", 10_607, 10_606),
        ("valid", 1_873, 1_873),
    ):
        sidecar = json.loads((project_root / files[f"{split_name}_sidecar"]["path"]).read_text())
        require(
            sidecar.get("model") == "c20260929_prop_f1", f"{split_name} cache names another model"
        )
        require(
            sidecar.get("model_sha256") == model_sha256,
            f"{split_name} cache is not bound to expected model",
        )
        require(sidecar.get("split") == split_name, f"{split_name} cache sidecar split differs")
        require(sidecar.get("rows") == expected_rows, f"{split_name} cache row count differs")
        require(
            sidecar.get("n_filled") == expected_filled,
            f"{split_name} cache filled-row count differs",
        )
        cache_path = project_root / files[f"{split_name}_cache"]["path"]
        shape, dtype, fortran_order, offset = read_npy_header(cache_path)
        require(
            shape == (expected_rows, 20_000),
            f"{split_name} propensity cache shape differs: {shape}",
        )
        require(
            dtype in ("<f4", "=f4") and not fortran_order,
            f"{split_name} propensity cache layout differs",
        )
        require(
            cache_path.stat().st_size == offset + expected_rows * 20_000 * 4,
            f"{split_name} propensity cache NPY size differs",
        )
    ancestry = verify_ancestry(project_root, expected, files)
    return {
        "schema": "wnn.h388-prop-preflight-receipt.v1",
        "propensity_id": expected["propensity_id"],
        "manifest_sha256": manifest_sha256,
        "files": observed_files,
        "ancestry": ancestry,
        "result": "pass",
    }


def main() -> None:
    """Run a bounded remote input check and print its complete verification receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--self-sha256", required=True)
    args = parser.parse_args()
    require(sha256_file(Path(__file__)) == args.self_sha256, "preflight helper source hash differs")
    print(
        json.dumps(validate(args.project_root, args.manifest, args.manifest_sha256), sort_keys=True)
    )


if __name__ == "__main__":
    main()
