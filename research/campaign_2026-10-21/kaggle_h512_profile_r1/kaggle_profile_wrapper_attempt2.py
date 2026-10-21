from __future__ import annotations

import hashlib
import json
import logging
import os
import runpy
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ORIGINAL_MANIFEST = Path(
    "research/campaign_2026-10-21/kaggle_h512_profile_r1/canonical_fold0_original.json"
)
PROFILE_MANIFEST = Path(
    "research/campaign_2026-10-21/kaggle_h512_profile_r1/fold0_profile_view.json"
)
PORTABLE_REGISTRY = Path("research/campaign_2026-10-04/portable_inputs.json")
EXPECTED_ORIGINAL_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_PROFILE_SHA256 = "60bc2897b88da6da784bf79fbc301f195f27fee182e0447547e4cc9512e17624"
EXPECTED_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
EXPECTED_SOURCES = {
    "datasets/train.parquet": {
        "bytes": 29240219011,
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    },
    "datasets/valid.parquet": {
        "bytes": 5615509128,
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    },
}
LOGGER = logging.getLogger("wnn.kaggle_profile_wrapper")


def configure_logging() -> None:
    """Write flushed JSON phase events to the sealed worker log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def emit_event(event: str, started_monotonic: float, **fields: object) -> None:
    """Record one timestamped profile phase event without logging data payloads."""
    payload = {
        "event": event,
        "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        **fields,
    }
    LOGGER.info("%s", json.dumps(payload, sort_keys=True, separators=(",", ":")))


def read_json_object(path: Path) -> dict[str, object]:
    """Read a small sealed JSON metadata file and require an object root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 identity of an already bounded metadata payload."""
    return hashlib.sha256(value).hexdigest()


def required_source_argument(arguments: list[str], name: str) -> str:
    """Return one required trainer option value without modifying its argv."""
    try:
        index = arguments.index(name)
    except ValueError as error:
        raise ValueError(f"Missing required profile option {name}") from error
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value for profile option {name}")
    return arguments[index + 1]


def validate_profile_manifests(root: Path, started_monotonic: float) -> None:
    """Check profile-view membership and the two source identities before imports."""
    original_bytes = ORIGINAL_MANIFEST.read_bytes()
    profile_bytes = PROFILE_MANIFEST.read_bytes()
    registry_bytes = PORTABLE_REGISTRY.read_bytes()
    observed_hashes = {
        "original_manifest_sha256": sha256_bytes(original_bytes),
        "profile_manifest_sha256": sha256_bytes(profile_bytes),
        "portable_registry_sha256": sha256_bytes(registry_bytes),
    }
    expected_hashes = {
        "original_manifest_sha256": EXPECTED_ORIGINAL_SHA256,
        "profile_manifest_sha256": EXPECTED_PROFILE_SHA256,
        "portable_registry_sha256": EXPECTED_REGISTRY_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(f"Profile manifest identity mismatch: {observed_hashes}")

    original = read_json_object(ORIGINAL_MANIFEST)
    profile = read_json_object(PROFILE_MANIFEST)
    registry = read_json_object(PORTABLE_REGISTRY)
    original_sources = original.get("sources")
    profile_sources = profile.get("sources")
    registry_sources = registry.get("sources")
    if not all(
        isinstance(value, dict) for value in (original_sources, profile_sources, registry_sources)
    ):
        raise ValueError("Manifest source sections must be JSON objects")
    if set(profile_sources) != set(EXPECTED_SOURCES):
        raise ValueError("Profile source view must contain exactly train and valid parquet")

    reconstructed = dict(original)
    reconstructed["sources"] = profile_sources
    if reconstructed != profile:
        raise ValueError("Profile view changed a canonical non-source field")
    if registry.get("split_manifest_sha256") != EXPECTED_ORIGINAL_SHA256:
        raise ValueError("Portable registry is not pinned to the canonical split")

    observed_mounts: dict[str, dict[str, object]] = {}
    for name, expected in EXPECTED_SOURCES.items():
        identity = profile_sources.get(name)
        registry_entry = registry_sources.get(name)
        if not isinstance(identity, dict) or not isinstance(registry_entry, dict):
            raise ValueError(f"Missing portable identity for {name}")
        if registry_entry.get("original_identity") != identity:
            raise ValueError(f"Portable identity differs from profile source: {name}")
        if registry_entry.get("sha256") != expected["sha256"]:
            raise ValueError(f"Portable content hash differs from pinned source: {name}")
        source_path = root / name
        stat = source_path.stat()
        if not source_path.is_symlink() or stat.st_size != expected["bytes"]:
            raise ValueError(f"Mounted source size or symlink mismatch: {name}")
        observed_mounts[name] = {
            "expected_bytes": expected["bytes"],
            "observed_bytes": stat.st_size,
            "expected_sha256": expected["sha256"],
            "sha256_check": "delegated_to_canonical_load_campaign_split",
        }

    emit_event(
        "profile_manifest_checks_passed",
        started_monotonic,
        fold=profile.get("fold"),
        n_folds=profile.get("n_folds"),
        non_source_fields_equal=True,
        source_names=sorted(profile_sources),
        mounts=observed_mounts,
        **observed_hashes,
    )


def main() -> None:
    """Verify the sealed profile view, time canonical source preflight, then run it."""
    configure_logging()
    started = time.monotonic()
    raw_arguments = sys.argv[1:]
    if not raw_arguments or raw_arguments[0] != "--":
        raise ValueError("Expected a -- delimiter before the pinned trainer argv")
    trainer_argv = raw_arguments[1:]
    if not trainer_argv or trainer_argv[0] != "scripts/train_rnn.py":
        raise ValueError("Only the sealed scripts/train_rnn.py entrypoint is permitted")
    if required_source_argument(trainer_argv, "--hidden") != "512":
        raise ValueError("H512 profile requires hidden size 512")
    if required_source_argument(trainer_argv, "--valid-fold") != "0":
        raise ValueError("H512 profile requires canonical fold 0")
    if required_source_argument(trainer_argv, "--profile-steps") != "3":
        raise ValueError("H512 profile requires three measured batches")
    if required_source_argument(trainer_argv, "--profile-warmup") != "1":
        raise ValueError("H512 profile requires one warmup batch")
    root = Path.cwd()
    split_argument = required_source_argument(trainer_argv, "--split-manifest")
    if split_argument != PROFILE_MANIFEST.as_posix():
        raise ValueError("Trainer argv is not bound to the derived profile view")
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST") != PORTABLE_REGISTRY.as_posix():
        raise ValueError("Portable data registry differs from the sealed profile")
    emit_event(
        "profile_wrapper_started",
        started,
        root=root.as_posix(),
        trainer_argv=trainer_argv,
        python=sys.version.split()[0],
    )
    validate_profile_manifests(root, started)

    emit_event(
        "canonical_split_loader_import_started",
        started,
        torch_loaded_before_import="torch" in sys.modules,
    )
    from wnn.splits import load_campaign_split

    if "torch" in sys.modules:
        raise RuntimeError("Torch loaded before canonical source preflight completed")
    emit_event("raw_source_preflight_started", started, loader="wnn.splits.load_campaign_split")
    split = load_campaign_split(PROFILE_MANIFEST, root)
    if split.fold != 0 or split.n_folds != 4:
        raise ValueError("Canonical split loader returned an unexpected fold")
    emit_event(
        "raw_source_preflight_completed",
        started,
        identity="passed",
        fold=split.fold,
        n_folds=split.n_folds,
        train_group_count=len(split.train_groups),
        valid_group_count=len(split.valid_groups),
        evaluation_group_count=len(split.evaluation_groups),
        torch_loaded_before_training="torch" in sys.modules,
    )

    sys.argv = trainer_argv
    emit_event("pinned_trainer_started", started, entrypoint=trainer_argv[0])
    runpy.run_path(str(root / trainer_argv[0]), run_name="__main__")
    emit_event("pinned_trainer_completed", started, profile_only=True)


if __name__ == "__main__":
    main()
