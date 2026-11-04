from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def training_source_for_stage(
    run_manifest: Mapping[str, object], stage: int
) -> tuple[Path, str, str, str]:
    """Return the frozen trainer root/revision/source hashes for one fit stage."""
    stage_sources = run_manifest.get("stage_training_sources")
    record = stage_sources.get(str(stage)) if isinstance(stage_sources, Mapping) else None
    if record is None:
        root = run_manifest.get("training_source_root")
        revision = run_manifest.get("training_source_revision")
        source_sha = run_manifest.get("training_source_sha256")
        files_sha = run_manifest.get("training_source_files_sha256")
    elif isinstance(record, Mapping):
        root = record.get("root")
        revision = record.get("revision")
        source_sha = record.get("sha256")
        files_sha = record.get("files_sha256")
    else:
        raise ValueError(f"stage {stage} training source record is malformed")
    if not all(isinstance(value, str) and value for value in (root, revision, source_sha, files_sha)):
        raise ValueError(f"stage {stage} training source identity is incomplete")
    return Path(root).resolve(strict=True), revision, source_sha, files_sha
