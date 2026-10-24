"""Verify that every KD cache came from its declared frozen G-cell recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import canonical_sha256, read_object, require
from catalog import normalized_runtime_config

# Cell declarations, cache manifests and endpoint receipts are heterogeneous JSON.


def validate_parent_config_binding(
    campaign_root: Path,
    cache_name: str,
    parent_name: str,
    declaration: dict[str, Any],
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Match the cache producer, G cell config, checkpoint identity, and status receipt."""
    # Resolve the validated G record lazily to avoid coupling packet import order.
    from materialize import get_cell

    cell_id = declaration.get("producer_cell_id")
    require(isinstance(cell_id, str) and bool(cell_id), "KD cache producer lacks a G cell ID")
    producer_cell = get_cell(campaign_root, "G", cell_id)
    expected_run = (campaign_root / producer_cell["run_dir"]).resolve()
    expected_cache = (campaign_root / producer_cell["teacher_cache"]).resolve()
    require(
        str(producer_cell.get("id")) == cell_id
        and producer_cell.get("kind") == "teacher"
        and declaration.get("fold") == producer_cell["fold"]
        and declaration.get("seed") == producer_cell["seed"]
        and expected_run == (campaign_root / parent_name).resolve()
        and expected_cache == (campaign_root / cache_name).resolve()
        and declaration.get("producer_run_dir") == producer_cell["run_dir"]
        and declaration.get("path") == producer_cell["teacher_cache"],
        "KD cache producer paths differ from its G cell",
    )
    config_path = campaign_root / producer_cell["config"]
    raw_config = read_object(config_path)
    raw_digest = canonical_sha256(raw_config)
    require(
        producer_cell.get("config_sha256") == raw_digest
        and declaration.get("producer_config_sha256") == raw_digest,
        "KD cache producer config differs from its frozen G cell",
    )
    expected_config = normalized_runtime_config(raw_config)
    cache_producer = manifest.get("producer")
    identity = status.get("identity")
    require(
        isinstance(cache_producer, dict)
        and cache_producer.get("config") == expected_config
        and isinstance(identity, dict)
        and identity.get("config_sha256") == canonical_sha256(expected_config),
        "KD cache and completed G run do not bind the frozen producer config",
    )
