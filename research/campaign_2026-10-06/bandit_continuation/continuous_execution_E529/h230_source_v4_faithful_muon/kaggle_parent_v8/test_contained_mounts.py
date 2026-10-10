from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
PACKET = Path(__file__).resolve().parent
PREDECESSOR = PACKET.parent / "kaggle_parent_v6"


def main() -> None:
    """Exercise actual frozen cache consumer with relocated external inputs."""
    spec = importlib.util.spec_from_file_location("bootstrap", PACKET / "kernel/kernel.py")
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    sys.path.insert(0, str(PREDECESSOR / "payload_root/src"))
    from wnn.all_data_distillation_cache import validate_all_data_distillation_cache

    manifest = json.loads((PREDECESSOR / "bundle/bundle-manifest.json").read_text())
    descriptors = {entry["path"]: entry for entry in manifest["files"]}
    cache_name = "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
    split_name = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    with tempfile.TemporaryDirectory(prefix="wnn-v7-real-consumer-") as temp:
        root = Path(temp)
        (root / "cache").mkdir()
        for name in ("train_targets.npz", "valid_targets.npz"):
            (root / "cache" / name).symlink_to(ROOT / "cache" / name)
        cache = root / cache_name
        cache.mkdir()
        for name in ("metadata.json", "main_source_train.npy", "main_source_valid.npy"):
            (cache / name).symlink_to(ROOT / cache_name / name)
        split = root / split_name
        split.parent.mkdir(parents=True)
        split.symlink_to(ROOT / split_name)
        try:
            validate_all_data_distillation_cache(cache, root, "source", require_families=False)
        except ValueError as error:
            assert str(error) == "teacher source split escapes the data root", str(error)
        else:
            raise AssertionError("Original relocated manifest was not rejected")
        for logical in (split_name, *(cache_name + "/" + name for name in ("metadata.json", "main_source_train.npy", "main_source_valid.npy"))):
            target = root / logical
            target.unlink()
            bootstrap.materialize_verified_input(ROOT / logical, target, descriptors[logical])
        result = validate_all_data_distillation_cache(cache, root, "source", require_families=False)
        assert len(result["verified_files"]) == 2
        old_bytes = split.read_bytes()
        try:
            bootstrap.materialize_verified_input(ROOT / split_name, split, descriptors[split_name])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing target overwritten")
        assert split.read_bytes() == old_bytes
        wrong = root / "tampered.json"
        descriptor = dict(descriptors[split_name])
        descriptor["sha256"] = "0" * 64
        try:
            bootstrap.materialize_verified_input(ROOT / split_name, wrong, descriptor)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong hash accepted")
        assert not wrong.exists()
        print(json.dumps({"status": "pass", "actual_frozen_consumer": True, "verified_files": result["verified_files"], "metadata_sha256": result["metadata_sha256"], "overwrite_rejected": True, "hash_mismatch_rejected": True}))


if __name__ == "__main__":
    main()
