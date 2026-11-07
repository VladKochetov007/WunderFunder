from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# This boundary validates heterogeneous operator-supplied JSON before report generation.
JsonObject = dict[str, Any]
SCHEMA = "wnn.sequence-progress-panel.v1"


def load_panel_manifest(path: Path) -> JsonObject:
    """Read frozen B300 recipes and the two explicitly authorized L4 GRU continuations.

    The returned byte digest binds reports to exactly the operator-supplied file.
    Loading once at startup prevents a later file edit from changing a running reporter.
    """
    from scripts.sequence_progress_report import CELLS, EPOCHS, SPLIT

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65_536:
        raise ValueError("panel manifest must be a bounded regular JSON file")
    payload = path.read_bytes()
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or not {"schema", "panels"} <= set(value)
        or set(value) - {"schema", "panels", "l4_continuations"}
        or value["schema"] != SCHEMA
        or not isinstance(value["panels"], dict)
        or set(value["panels"]) - {"b300"}
        or not (value["panels"] or value.get("l4_continuations"))
    ):
        raise ValueError("panel manifest contains an unsupported or empty panel mapping")
    digest = hashlib.sha256(payload).hexdigest()
    panels = {}
    if "b300" in value["panels"]:
        panel = value["panels"]["b300"]
        _validate_b300(panel, CELLS["b300"], EPOCHS["b300"], SPLIT)
        panels["b300"] = panel | {"manifest_sha256": digest}
    if "l4_continuations" in value:
        continuations = value["l4_continuations"]
        _validate_l4_continuations(continuations)
        panels["l4"] = {
            "run_prefix": "l4",
            "manifest_sha256": digest,
            "continuations": continuations,
        }
    return panels


def _validate_b300(panel: JsonObject, cells: tuple[str, ...], epochs: int, split: str) -> None:
    if (
        not isinstance(panel, dict)
        or set(panel) != {"run_prefix", "epochs", "source_sha256", "split_sha256", "cells"}
        or panel["run_prefix"] != "b300-opt"
        or type(panel["epochs"]) is not int
        or panel["epochs"] != epochs
        or panel["split_sha256"] != split
        or not _sha(panel["source_sha256"])
        or not isinstance(panel["cells"], dict)
        or set(panel["cells"]) != set(cells)
    ):
        raise ValueError("panel manifest recipe, source, split or exact four cells differ")
    for recipe in panel["cells"].values():
        if (
            not isinstance(recipe, dict)
            or set(recipe) != {"config_sha256", "expected_updates"}
            or not _sha(recipe["config_sha256"])
            or type(recipe["expected_updates"]) is not int
            or not 1 <= recipe["expected_updates"] <= 10_000_000
        ):
            raise ValueError("panel cells require exact config SHA and positive update budget")


def _validate_l4_continuations(value: object) -> None:
    cells = ("gru_s101001", "gru_s101002")
    if not isinstance(value, dict) or set(value) != set(cells):
        raise ValueError("L4 continuation mapping requires exactly the two GRU seeds")
    for cell, recipe in value.items():
        if (
            not isinstance(recipe, dict)
            or set(recipe)
            != {"run_id", "source_sha256", "config_sha256", "expected_updates", "epochs"}
            or recipe["run_id"]
            not in {
                f"l4-{cell}_monitor_opt-fit",
                f"l4-{cell}_monitor_opt_r2-fit",
                f"l4-{cell}_monitor_opt_r3-fit",
            }
            or not _sha(recipe["source_sha256"])
            or not _sha(recipe["config_sha256"])
            or type(recipe["expected_updates"]) is not int
            or recipe["expected_updates"] != 77520
            or type(recipe["epochs"]) is not int
            or recipe["epochs"] != 12
        ):
            raise ValueError("L4 continuation alias, identity or exact12E/77520 budget differs")


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_recipe(panel: JsonObject, cell: str, ids: dict[str, str], updates: int) -> None:
    """Require actual producer metadata to match every frozen optimized recipe identity."""
    recipe = panel["cells"][cell]
    if (
        ids["source"] != panel["source_sha256"]
        or ids["split"] != panel["split_sha256"]
        or ids["config"] != recipe["config_sha256"]
        or updates != recipe["expected_updates"]
    ):
        raise ValueError("native metadata differs from the frozen optimized panel recipe")


def validate_l4_continuation(
    recipe: JsonObject, cell: str, metadata: JsonObject, ids: dict[str, str], updates: int
) -> None:
    """Bind an exact GRU continuation while preserving the original scientific budget."""
    config = metadata["config"]
    if (
        ids["source"] != recipe["source_sha256"]
        or ids["config"] != recipe["config_sha256"]
        or updates != recipe["expected_updates"]
        or config["epochs"] != recipe["epochs"]
        or config.get("seed") != int(cell.removeprefix("gru_s"))
        or config.get("model", {}).get("architecture") != "gru"
        or metadata["provenance"]["world_size"] != 1
    ):
        raise ValueError("native GRU continuation differs from the frozen identity/budget")
