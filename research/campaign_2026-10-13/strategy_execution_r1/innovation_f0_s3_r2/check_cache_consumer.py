"""Check actual T3 metadata and row mapping without re-reading large payloads."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parents[3]
ROOT = Path(os.environ["WNN_PROJECT_ROOT"])
sys.path.insert(0, str(SOURCE / "scripts"))


def main() -> None:
    """Reuse executed byte-validation evidence and check the new actual consumer."""
    import torch
    import train_rnn

    from wnn.distillation_cache import (
        _validate_cleanfold_t3_profile,
        file_sha256,
        row_groups_sha256,
    )

    reader_path = ROOT / "scripts/alias_grade_prediction.py"
    spec = importlib.util.spec_from_file_location("bounded_npy_reader", reader_path)
    assert spec is not None and spec.loader is not None
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    assert train_rnn.SOURCE_ROOT == SOURCE and train_rnn.ROOT == ROOT
    assert not torch.cuda.is_initialized()
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = json.loads(split_path.read_text())
    cache = ROOT / "cache/clean_t3_f0_h390_t3r5_a9_r2"
    metadata = json.loads((cache / "metadata.json").read_text())
    assert metadata["split_manifest_sha256"] == file_sha256(split_path)
    assert metadata["fit_groups"] == split["fit"]
    assert metadata["evaluation_valid_groups"] == split["evaluation"]["valid"]
    _validate_cleanfold_t3_profile(metadata, "homogeneous", False)
    prior_path = ROOT / "artifacts/runs/clean_t3_f0_amplitude1_s3_a9_t3r5_r3_p2/provenance.json"
    prior = json.loads(prior_path.read_text())["distillation_cache"]
    assert prior["metadata_sha256"] == file_sha256(cache / "metadata.json")
    checks = []
    for source in ("train", "valid"):
        target_index_path = ROOT / f"cache/{source}_targets.npz"
        with np.load(target_index_path, allow_pickle=False) as target:
            target_order = list(map(int, target["row_group"]))
        fit = split["fit"][source]
        expected = [g for g in target_order if g in set(fit)]
        key = f"main_homogeneous_{source}"
        entry = metadata["files"][key]
        assert entry["row_groups"] == expected
        assert entry["row_groups_sha256"] == row_groups_sha256(expected)
        assert entry["path"] == f"{key}.npy"
        path = cache / entry["path"]
        shape, dtype, fortran, offset = reader._npy_header(path)
        assert shape == (len(fit), 20_000, 2) and dtype == np.float32 and not fortran
        assert path.stat().st_size == entry["bytes"] == offset + len(fit) * 20_000 * 8
        assert prior["verified_files"][key] == entry["sha256"]
        dataset = train_rnn.Sequences(
            source, fit, distill_cache_root=cache,
            distill_cache_variant="homogeneous", distill_cache_metadata={"metadata": metadata},
        )
        assert dataset._distill_pos == {g: i for i, g in enumerate(expected)}
        assert dataset._file is None and dataset._distill_main is None
        checks.append({"source": source, "groups": len(fit), "shape": shape,
                       "bytes": path.stat().st_size, "metadata_row_order": "pass",
                       "actual_Sequences_row_map": "pass", "npy_header": "pass",
                       "payload_sha256_from_completed_P2_validation": entry["sha256"]})
    record = {
        "created_utc": datetime.now(UTC).isoformat(), "status": "pass_metadata_and_actual_consumer_mapping",
        "validator_sha256": file_sha256(SOURCE / "src/wnn/distillation_cache.py"),
        "helper_sha256": file_sha256(Path(__file__)), "bounded_reader_sha256": file_sha256(reader_path),
        "metadata_sha256": file_sha256(cache / "metadata.json"), "checks": checks,
        "prior_full_validation_provenance": str(prior_path),
        "prior_full_validation_provenance_sha256": file_sha256(prior_path),
        "payload_rehashed_here": False, "feature_or_label_payload_read_here": False,
        "cache_payload_read_here": False, "cuda_initialized": torch.cuda.is_initialized(),
        "limit": "Metadata/header/constructor checks only. Full numerical byte checks reuse the completed P2 receipt from the identical validator; the actual profile validates them again before fitting.",
    }
    (BASE / "ACTUAL_CACHE_CONSUMER_PREFLIGHT_R2.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "groups": [c["groups"] for c in checks]}))


if __name__ == "__main__":
    main()
