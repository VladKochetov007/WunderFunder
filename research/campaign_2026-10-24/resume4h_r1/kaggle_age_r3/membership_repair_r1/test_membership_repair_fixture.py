"""Exercise portable membership validation on actual two-fold cache metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from wnn.training.age_readout_portable_validate import fold_metadata
from wnn.training.online_relationship_cache import row_refs_sha256

ROOT = Path(__file__).resolve().parents[5]
PACKET = Path(__file__).resolve().parent
SOURCE_RECIPE = (
    ROOT / "research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json"
)
CACHE_RECEIPT = (
    ROOT
    / "research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json"
)
MILESTONE = (
    ROOT / "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
)
SOURCE_PINS = PACKET / "SOURCE_PINS_R1.json"

# `Any` is needed for values parsed from producer JSON metadata.
JsonObject = dict[str, Any]
SOURCE_MODULES = {
    "subset_loader": ROOT / "src/wnn/training/membership.py",
    "cache_metadata_producer": ROOT / "src/wnn/training/online_relationship_sources.py",
    "cache_sidecar_writer": ROOT / "src/wnn/training/online_relationship_data_writer.py",
    "cache_validator": ROOT / "src/wnn/training/online_relationship_cache_validation.py",
    "portable_identity_consumer": ROOT / "src/wnn/training/age_readout_portable_identity.py",
    "portable_metadata_consumer": ROOT / "src/wnn/training/age_readout_portable_validate.py",
}


def sha256_file(path: Path) -> str:
    """Hash a small metadata file without opening any cache payload."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    """Read one JSON metadata file."""
    return json.loads(path.read_text(encoding="utf-8"))


def peak_rss_bytes() -> int:
    """Read the current process high-water mark from Linux proc status."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux process status did not report VmHWM")


def cache_record(receipt: JsonObject, fold_id: int, kind: str) -> JsonObject:
    """Return the unique cache-verification record for a fold and kind."""
    matches = [
        item for item in receipt["files"] if int(item["fold"]) == fold_id and item["kind"] == kind
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {kind} record for fold {fold_id}")
    return matches[0]


def load_fold_fixture(
    fold_id: int, recipe: JsonObject, receipt: JsonObject, milestone: JsonObject
) -> tuple[JsonObject, JsonObject, dict[str, str]]:
    """Bind a frozen fold to its checked-in source-produced JSON sidecars."""
    folds = [item for item in recipe["folds"] if int(item["fold"]) == fold_id]
    if len(folds) != 1:
        raise AssertionError(f"expected one source recipe fold {fold_id}")
    source_fold = dict(folds[0])
    subset_path = PACKET / f"fixture_fold{fold_id}_fit_subset.json"
    if sha256_file(subset_path) != source_fold["subset"]["sha256"]:
        raise AssertionError(f"fold {fold_id} subset fixture differs from frozen recipe")
    source_fold["subset"] = {**source_fold["subset"], "path": str(subset_path)}
    sidecars: dict[str, JsonObject] = {}
    hashes: dict[str, str] = {}
    for kind in ("fit_sequences", "evaluation_sequences", "evaluation_predictions"):
        path = PACKET / f"fixture_fold{fold_id}_{kind}.json"
        expected = str(cache_record(receipt, fold_id, kind)["sidecar_sha256"])
        milestone_sha = milestone["parent_caches"][str(fold_id)]["sidecars"][f"{kind}.json"][
            "sha256"
        ]
        observed = sha256_file(path)
        if observed != expected or expected != milestone_sha:
            raise AssertionError(f"fold {fold_id} {kind} fixture differs from its source pin")
        sidecars[kind] = {"path": path.name, "sha256": expected}
        hashes[kind] = observed
    return source_fold, {"sidecars": sidecars}, hashes


def check_fold(
    fold_id: int,
    recipe: JsonObject,
    recipe_sha: str,
    receipt: JsonObject,
    milestone: JsonObject,
) -> JsonObject:
    """Run fold metadata validation and compare source-qualified membership."""
    source_fold, bundle_fold, sidecar_hashes = load_fold_fixture(
        fold_id, recipe, receipt, milestone
    )
    subset = read_json(Path(source_fold["subset"]["path"]))
    selected = subset["selected_groups"]
    members = sorted(
        (split, int(group)) for split in ("train", "valid") for group in selected[split]
    )
    expected_members_sha = subset["selected_members_sha256"]
    recomputed_sha = hashlib.sha256(
        json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    if recomputed_sha != expected_members_sha:
        raise AssertionError(f"fold {fold_id} subset group hash differs from its manifest")

    fit_refs, eval_refs, _, _ = fold_metadata(
        ROOT, recipe_sha, source_fold, fold_id, bundle_fold, receipt, milestone, PACKET
    )
    fit_meta = read_json(PACKET / f"fixture_fold{fold_id}_fit_sequences.json")
    eval_meta = read_json(PACKET / f"fixture_fold{fold_id}_evaluation_sequences.json")
    expected_refs = [
        {"split": split, "row_group": int(group)}
        for split in ("train", "valid")
        for group in selected[split]
    ]
    if fit_refs != expected_refs:
        raise AssertionError(f"fold {fold_id} fit source-qualified order changed")
    if any(
        meta.get("subset_members_sha256") != expected_members_sha for meta in (fit_meta, eval_meta)
    ):
        raise AssertionError(
            f"fold {fold_id} producer sidecar membership differs from source groups"
        )
    return {
        "fold": fold_id,
        "fit_source_qualified_groups": len(fit_refs),
        "evaluation_source_qualified_groups": len(eval_refs),
        "subset_manifest_sha256": source_fold["subset"]["sha256"],
        "selected_members_sha256_expected": expected_members_sha,
        "selected_members_sha256_recomputed_from_groups": recomputed_sha,
        "fit_cache_subset_members_sha256_observed": fit_meta["subset_members_sha256"],
        "evaluation_cache_subset_members_sha256_observed": eval_meta["subset_members_sha256"],
        "fit_row_refs_sha256": row_refs_sha256(fit_refs),
        "evaluation_row_refs_sha256": row_refs_sha256(eval_refs),
        "producer_sidecar_sha256": sidecar_hashes,
    }


def assert_mismatch_rejected(
    recipe: JsonObject, recipe_sha: str, receipt: JsonObject, milestone: JsonObject
) -> None:
    """Confirm that the exact subset equality check still rejects a changed digest."""
    source_fold, bundle_fold, _ = load_fold_fixture(0, recipe, receipt, milestone)
    with tempfile.TemporaryDirectory(prefix="age-membership-repair-") as temporary:
        bundle_root = Path(temporary)
        bad_bundle: JsonObject = {"sidecars": {}}
        for kind, entry in bundle_fold["sidecars"].items():
            output = bundle_root / entry["path"]
            shutil.copyfile(PACKET / entry["path"], output)
            bad_bundle["sidecars"][kind] = {"path": output.name, "sha256": entry["sha256"]}
        fit_path = bundle_root / bad_bundle["sidecars"]["fit_sequences"]["path"]
        fit_meta = read_json(fit_path)
        fit_meta["subset_members_sha256"] = "0" * 64
        fit_path.write_text(json.dumps(fit_meta), encoding="utf-8")
        try:
            fold_metadata(
                ROOT, recipe_sha, source_fold, 0, bad_bundle, receipt, milestone, bundle_root
            )
        except ValueError as error:
            if str(error) == "portable fit cache subset membership identity changed":
                return
            raise
    raise AssertionError("portable validator accepted a changed subset membership digest")


def main() -> None:
    """Validate both actual folds without loading arrays, labels, masks or models."""
    started = time.perf_counter()
    recipe, receipt, milestone = (
        read_json(SOURCE_RECIPE),
        read_json(CACHE_RECEIPT),
        read_json(MILESTONE),
    )
    recipe_sha = sha256_file(SOURCE_RECIPE)
    folds = [check_fold(fold, recipe, recipe_sha, receipt, milestone) for fold in (0, 1)]
    assert_mismatch_rejected(recipe, recipe_sha, receipt, milestone)
    if "torch" in sys.modules:
        raise AssertionError("metadata-only portable validation imported Torch")

    pins = read_json(SOURCE_PINS)
    if pins["source_files"]["source_cache_execution_recipe"]["sha256"] != recipe_sha:
        raise AssertionError("source pin manifest has a stale execution recipe hash")
    for name, path in SOURCE_MODULES.items():
        if pins["source_files"][name]["sha256"] != sha256_file(path):
            raise AssertionError(f"source pin manifest has a stale hash for {name}")
    if pins["source_files"]["fixture_test"]["sha256"] != sha256_file(Path(__file__)):
        raise AssertionError("source pin manifest has a stale fixture test hash")

    output = PACKET / "TEST_RECEIPT_R1.json"
    output.write_text(
        json.dumps(
            {
                "schema": "wnn.age-readout-membership-repair-metadata-test.v1",
                "status": "PASS",
                "scope": "actual source-produced JSON sidecars and frozen source-qualified fold manifests only",
                "checks": {
                    "both_fold_metadata_paths_pass": True,
                    "altered_subset_membership_digest_rejected": True,
                    "raw_arrays_opened": False,
                    "target_labels_or_masks_read": False,
                    "torch_imported": False,
                    "cache_rebuild_or_fit_run": False,
                },
                "execution": {
                    "command": "PYTHONPATH=src WNN_MEM_MAX=150M WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=60 scripts/limited python research/campaign_2026-10-24/resume4h_r1/kaggle_age_r3/membership_repair_r1/test_membership_repair_fixture.py",
                    "configured_memory_limit_bytes": 150_000_000,
                    "configured_cpu_quota": "100%",
                    "configured_threads": 1,
                    "configured_runtime_seconds": 60,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "peak_rss_bytes": peak_rss_bytes(),
                },
                "source_recipe_sha256": recipe_sha,
                "cache_verification_receipt_sha256": sha256_file(CACHE_RECEIPT),
                "parent_cache_milestone_sha256": sha256_file(MILESTONE),
                "source_pins_path": str(SOURCE_PINS.relative_to(ROOT)),
                "source_pins_sha256": sha256_file(SOURCE_PINS),
                "folds": folds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS: both folds and mismatch rejection; receipt={output}")


if __name__ == "__main__":
    main()
