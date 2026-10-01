"""Independently check equal supervision masks and excluded-fold cache contents."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from replay import HERE, ROOT, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    metadata_path = args.root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    manifest_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    manifest = json.loads(manifest_path.read_text())
    if metadata["split_manifest_sha256"] != digest(manifest_path):
        raise ValueError("Cache belongs to another split")
    if metadata["train_lattice_audit_sha256"] != digest(HERE / "lattice_audit_f0.json"):
        raise ValueError("TRAIN lattice audit changed")
    if metadata["auxiliary_class_index_sha256"] != digest(HERE / "aux_class_index_f0.json"):
        raise ValueError("Auxiliary class index changed")
    if metadata["source_script_sha256"] != digest(HERE / "build_aux_cache.py"):
        raise ValueError("Cache builder source changed")
    if metadata["fit_train_groups"] != manifest["fit"]["train"]:
        raise ValueError("TRAIN cache row-group provenance mismatch")
    if metadata["fit_valid_groups"] != manifest["fit"]["valid"]:
        raise ValueError("VALID cache row-group provenance mismatch")
    if metadata["evaluation_valid_groups"] != sorted(manifest["evaluation"]["valid"]):
        raise ValueError("Evaluation row-group provenance mismatch")
    result = {}
    for split in ("train", "valid"):
        control_path = args.root / "code" / f"quote_path_{split}.npy"
        candidate_path = args.root / "quote8" / f"quote_path_{split}.npy"
        for variant, source_path in (("code", control_path), ("quote8", candidate_path)):
            expected = metadata["files"][f"{variant}/{split}"]["sha256"]
            if digest(source_path) != expected:
                raise ValueError(f"{variant}/{split}: array content hash mismatch")
        control = np.load(control_path, mmap_mode="r")
        candidate = np.load(candidate_path, mmap_mode="r")
        if control.shape != candidate.shape or control.dtype != np.float16:
            raise ValueError(f"{split}: cache shape/dtype mismatch")
        counts = {"groups": 0, "finite_horizon8_rows": 0,
                  "changed_horizon8_rows": 0, "finite_horizon24_rows": 0}
        for group in manifest["fit"][split]:
            a = np.asarray(control[group])
            b = np.asarray(candidate[group])
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise ValueError(f"{split}/{group}: finite masks differ")
            if not np.array_equal(a[:, 1], b[:, 1], equal_nan=True):
                raise ValueError(f"{split}/{group}: horizon24 differs")
            counts["groups"] += 1
            counts["finite_horizon8_rows"] += int(np.isfinite(a[:, 0]).sum())
            counts["changed_horizon8_rows"] += int(np.sum(
                np.isfinite(a[:, 0]) & (np.abs(a[:, 0].astype(np.float32) -
                                             b[:, 0].astype(np.float32)) > 0.001)
            ))
            counts["finite_horizon24_rows"] += int(np.isfinite(a[:, 1]).sum())
        if split == "valid":
            for group in manifest["evaluation"]["valid"]:
                if np.any(control[group]) or np.any(candidate[group]):
                    raise ValueError(f"Evaluation group {group} contains auxiliary data")
            counts["zero_evaluation_groups"] = len(manifest["evaluation"]["valid"])
        else:
            if np.any(control[2315]) or np.any(candidate[2315]):
                raise ValueError("TRAIN lattice outlier 2315 has nonzero auxiliary data")
            counts["zero_lattice_outlier_groups"] = 1
        result[split] = counts
    output = {"cache_metadata_sha256": digest(metadata_path),
              "cache_root": str(args.root), "checks": result,
              "resources": {"wall_seconds": time.monotonic() - started,
                            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "exit_code": 0}}
    destination = HERE / "cache_verification_f0_v3.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
