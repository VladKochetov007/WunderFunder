"""Bind publication labels to producer selections and safe archive run names."""

from __future__ import annotations

import json
import math
import re


def validate_inventory(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Require the exact panel sizes and unique safe identifiers before creating output."""
    present = [record for record in records if record.get("best_exists")]
    names = [record.get("run") for record in records]
    if any(
        not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) is None
        for name in names
    ):
        raise ValueError("Unsafe archive run identifier")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate run identifier")
    if len(present) != 24 or any(not record.get("hash_matches") for record in present):
        raise ValueError("Expected exactly24 verified existing best checkpoints")
    if any(
        sum(record["pr"] == pr for record in present) != count for pr, count in ((46, 8), (49, 16))
    ):
        raise ValueError("PR checkpoint panel counts changed")
    for record in present:
        name, pr = str(record["run"]), record["pr"]
        if not (pr == 46 and re.fullmatch(r"g[0-7]-[a-z0-9-]+", name)) and not (
            pr == 49 and name.startswith("tl3c_")
        ):
            raise ValueError("Run identifier differs from its experiment panel")
    return present


def verify_selection(record: dict[str, object], metadata: dict[str, bytes]) -> None:
    """Cross-check scores, epochs and hashes against one captured producer snapshot."""
    if int(record["pr"]) == 46:
        pointers = json.loads(metadata["checkpoint_pointers.json"])
        best = pointers["best"]
        if (
            best["sha256"] != record["expected_sha256"]
            or best["epoch"] != record["best_epoch"]
            or best["valid_wp"] != record["best_wp"]
            or pointers["best_raw_wp"] != record["best_wp"]
            or str(record["best_path"]).replace("\\", "/").rsplit("/", 2)[-2:]
            != str(best["path"]).replace("\\", "/").split("/")
        ):
            raise ValueError("Best pointer epoch, score, path or hash changed after inventory")
        return
    history = json.loads(metadata["history.json"])
    provenance = json.loads(metadata["provenance.json"])
    if not history or len(history) != record["epochs"]:
        raise ValueError("Checkpoint history length changed")
    if any(
        row.get("epoch") != index or not math.isfinite(row.get("weighted_pearson", float("nan")))
        for index, row in enumerate(history)
    ):
        raise ValueError("Checkpoint history epoch or score is invalid")
    best = max(history, key=lambda row: row["weighted_pearson"])
    fixed_index = provenance["selected_epoch"]
    hashes = provenance["checkpoint_sha256"]
    if (
        best["epoch"] + 1 != record["best_epoch"]
        or best["weighted_pearson"] != record["best_wp"]
        or fixed_index + 1 != record["fixed_epoch"]
        or history[fixed_index]["weighted_pearson"] != record["fixed_wp"]
        or hashes[f"model_ep{best['epoch']}.pt"] != record["expected_sha256"]
        or hashes[f"model_ep{fixed_index}.pt"] != record["fixed_expected_sha256"]
        or hashes["model.pt"] != record["selected_expected_sha256"]
    ):
        raise ValueError("Best/fixed labels differ from producer history or provenance")
