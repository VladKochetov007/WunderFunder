from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
V4_ROOT = PACKET_ROOT.parent
REPO_ROOT = next(
    parent for parent in PACKET_ROOT.parents if (parent / "specs/README.md").is_file()
)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from kaggle_common import canonical_sha256, read_spec  # noqa: E402

SPEC_PATH = PACKET_ROOT / "spec/h230-seed9-parent-fit-v1.json"
TRANSPORT_PATH = PACKET_ROOT / "H230_KAGGLE_PARENT_FIT_TRANSPORT_V2_20261008.json"
ASSET_PATH = PACKET_ROOT / "cache_asset_upload/assets-manifest.json"
FOLD_ASSET_PATH = REPO_ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
RECIPE_PATH = V4_ROOT / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json"
SOURCE_MANIFEST_PATH = V4_ROOT / "H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_transport_is_pending_review_and_no_provider_side_effects() -> None:
    packet = load_json(TRANSPORT_PATH)
    assert packet["status"] == "prepared_for_root_review_not_uploaded_not_launched"
    assert packet["fit_authorized"] is False
    assert packet["transports"]["upload_or_push_performed"] is False
    assert packet["quota_status"].startswith("Not queried")
    assert packet["fit_spec_sha256"]
    assert packet["bundle_archive_sha256"]


def test_spec_closes_over_exact_v4_sources_and_mounts() -> None:
    spec = read_spec(SPEC_PATH)
    bundle = load_json(PACKET_ROOT / "bundle/bundle-manifest.json")
    kernel = load_json(PACKET_ROOT / "bundle/kernel/kernel-metadata.json")
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    source_paths = {item["path"] for item in source_manifest["files"]}
    code_paths = {item["path"] for item in spec["code"]}
    input_paths = {item["path"] for item in spec["inputs"]}

    assert len(source_paths) == 57
    assert source_paths <= code_paths
    assert {"scripts/kaggle_gpu_runner.py", "scripts/kaggle_common.py", "scripts/kaggle_kernel.py"} <= code_paths
    assert set(bundle["external_mounts"]) == input_paths
    assert len(input_paths) == 11
    assert canonical_sha256(spec["code"]) == spec["code_sha256"]
    assert kernel["is_private"] is True
    assert kernel["enable_gpu"] is True
    assert kernel["enable_tpu"] is False
    assert kernel["enable_internet"] is False
    assert kernel["machine_shape"] == "NvidiaTeslaT4"
    assert set(kernel["dataset_sources"]) == {
        "vladkochetov07/h230-seed9-parent-fit-v2",
        "vladkochetov07/h230-seed9-parent-source-cache-v2",
        "vladkochetov07/wnn-c20261004-fold0-assets",
        "vladkochetov07/wunder-funder",
    }


def test_fit_argv_only_changes_run_identity_and_cache_mount() -> None:
    spec = read_spec(SPEC_PATH)
    recipe = load_json(RECIPE_PATH)
    expected = list(recipe["fit_attempt"]["argv"])
    expected[expected.index("--name") + 1] = "bd26_h230_strong_parent_seed9_muon_kaggle_v2"
    expected[expected.index("--distill-cache-root") + 1] = (
        "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
    )
    assert spec["argv"][1:] == expected
    assert "--save-training-state" in spec["argv"]
    assert "--no-eval-during-fit" in spec["argv"]
    assert spec["limits"] == {
        "cpus": 2,
        "ram_bytes": 9_000_000_000,
        "timeout_seconds": 5_200,
        "gpu_memory_bytes": 4_300_000_000,
        "gpu_stop_bytes": 12_000_000_000,
    }


def test_parent_cache_upload_is_hardlinked_minimal_main_source_subset() -> None:
    manifest = load_json(ASSET_PATH)
    metadata = load_json(REPO_ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json")
    source_root = REPO_ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
    selected = {"main_source_train", "main_source_valid"}
    assert set(manifest["included_cache_keys"]) == selected
    assert manifest["source_training_scope"] == "clean_fold_fit_only"
    assert manifest["source_cache_metadata_sha256"] == "b05e652f5fdec4e6b1e4e07d1d3a705655d4588ab8ee9d802ccdf4339fd66500"
    assert len(manifest["files"]) == 3  # metadata plus the two selected main arrays

    for item in manifest["files"]:
        source = source_root / Path(item["path"]).name
        staged = PACKET_ROOT / "cache_asset_upload" / item["filename"]
        source_stat, staged_stat = source.stat(), staged.stat()
        assert item["filename"] == f"{item['sha256']}.wnn"
        assert source_stat.st_size == staged_stat.st_size == item["size"]
        assert (source_stat.st_dev, source_stat.st_ino) == (staged_stat.st_dev, staged_stat.st_ino)
        if item["path"].endswith("metadata.json"):
            assert metadata["files"]["main_source_train"]["path"] == "main_source_train.npy"
        else:
            assert item["path"].endswith(("main_source_train.npy", "main_source_valid.npy"))

    staged_names = {path.name for path in (PACKET_ROOT / "cache_asset_upload").iterdir()}
    assert "families_train.npy" not in staged_names
    assert "families_valid.npy" not in staged_names
    assert "main_normalized_train.npy" not in staged_names
    assert "main_homogeneous_train.npy" not in staged_names


def test_external_asset_inputs_match_manifest_and_raw_data_recipe() -> None:
    spec = read_spec(SPEC_PATH)
    fold_assets = load_json(FOLD_ASSET_PATH)
    fold_by_path = {item["path"]: item for item in fold_assets["files"]}
    input_by_path = {item["path"]: item for item in spec["inputs"]}
    recipe = load_json(RECIPE_PATH)

    for split in ("train", "valid"):
        path = f"datasets/{split}.parquet"
        assert input_by_path[path]["size"] == recipe["data"][split]["bytes"]
        assert input_by_path[path]["sha256"] == recipe["data"][split]["sha256"]

    for path in (
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        "artifacts/runs/c20260929_prop_f0/model.pt",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    ):
        entry = fold_by_path[path]
        assert input_by_path[path]["size"] == entry["size"]
        assert input_by_path[path]["sha256"] == entry["sha256"]

    external_names = set(load_json(PACKET_ROOT / "bundle/bundle-manifest.json")["external_mounts"])
    assert external_names == set(input_by_path)


def test_declared_cache_mounts_do_not_claim_parent_family_targets() -> None:
    packet = load_json(TRANSPORT_PATH)
    cache_manifest = load_json(ASSET_PATH)
    assert "no family head" in packet["cache_asset_policy"]
    assert cache_manifest["omitted_cache_arrays"]
    assert "families_train" in cache_manifest["omitted_cache_arrays"]
    assert "families_valid" in cache_manifest["omitted_cache_arrays"]
    assert packet["future_student_note"].startswith("A student is a fresh independent AdamW/OneCycle model")

