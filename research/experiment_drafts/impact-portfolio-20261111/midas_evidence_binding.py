"""Validate PR81 source, cell, split, checkpoint-index and training-metric identity."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from midas_evidence_io import _canonical, _object

SHA = re.compile(r"^[0-9a-f]{64}$")
VARIANTS = {"01": ("plain", "ple"), "02": ("plain", "groups"), "03": ("plain", "target_adapters")}


def pins(packet: Path) -> tuple[str, str, str]:
    """Read the PR81 runtime commit and split pins, plus the binding-file digest."""
    raw = (packet / "runtime/impact81/bindings.py").read_bytes()
    text = raw.decode("utf-8")
    commit = re.search(r'^RUNTIME_COMMIT\s*=\s*["\']([0-9a-f]{40})["\']$', text, re.M)
    split = re.search(r'^SPLIT_SHA256\s*=\s*["\']([0-9a-f]{64})["\']$', text, re.M)
    if not commit or not split:
        raise ValueError("PR81 source/split pin is malformed")
    return commit.group(1), split.group(1), hashlib.sha256(raw).hexdigest()


def cell_identity(config: dict[str, Any], fold: int) -> dict[str, Any]:
    """Project and validate a frozen PR81 card, arm, variant, seed and fold."""
    ext = config.get("impact81")
    cell = ext.get("cell") if isinstance(ext, dict) else None
    required = {"id", "card", "arm", "variant", "seed"}
    if not isinstance(ext, dict) or ext.get("schema") != "impact81.model.v1":
        raise ValueError("run config lacks PR81 model identity")
    if not isinstance(cell, dict) or not required <= set(cell):
        raise ValueError("run config lacks PR81 cell identity")
    card, arm, variant, seed = (cell[k] for k in ("card", "arm", "variant", "seed"))
    if not isinstance(card, str) or card not in VARIANTS:
        raise ValueError("PR81 card is not implemented by this exporter")
    if arm not in {"control", "treatment"}:
        raise ValueError("PR81 arm is invalid")
    if variant != VARIANTS[card][0 if arm == "control" else 1]:
        raise ValueError("PR81 variant does not match its card/arm")
    if type(seed) is not int or seed not in {204101, 204102} or config.get("seed") != seed:
        raise ValueError("PR81 seed differs from the frozen pair")
    if config.get("expected_fold") != fold or cell["id"] != f"ip{card}_{arm}_s{seed}":
        raise ValueError("PR81 cell does not bind its frozen fold or id")
    if cell.get("schema", "impact81.cell.v1") != "impact81.cell.v1":
        raise ValueError("PR81 cell schema is unsupported")
    return {k: cell[k] for k in ("id", "card", "arm", "variant", "seed")}


def run_identity(meta: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the identity fields MIDAS verifies against run metadata."""
    config, source, provenance = (meta.get(k) for k in ("config", "source", "provenance"))
    if not all(isinstance(value, dict) for value in (config, source, provenance)):
        raise ValueError("run metadata lacks identity objects")
    files, data = source.get("files"), provenance.get("data")
    if not isinstance(files, dict) or not isinstance(data, dict):
        raise ValueError("run metadata source/data identity is missing")
    source_sha, data_sha = source.get("sha256"), provenance.get("data_sha256")
    if not isinstance(source_sha, str) or not SHA.fullmatch(source_sha):
        raise ValueError("runtime source digest is malformed")
    if hashlib.sha256(_canonical(files)).hexdigest() != source_sha:
        raise ValueError("runtime source digest does not match the file map")
    if not isinstance(data_sha, str) or not SHA.fullmatch(data_sha):
        raise ValueError("data provenance digest is malformed")
    if hashlib.sha256(_canonical(data)).hexdigest() != data_sha:
        raise ValueError("data provenance digest does not match its payload")
    if any(not isinstance(v, str) or not SHA.fullmatch(v) for v in files.values()):
        raise ValueError("runtime source file digest is malformed")
    world = provenance.get("world_size")
    runtime, params = provenance.get("runtime"), provenance.get("model_parameters")
    if (
        type(world) is not int
        or world < 1
        or not isinstance(runtime, dict)
        or not isinstance(params, dict)
    ):
        raise ValueError("run metadata lacks runtime/world/model identity")
    return {
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "data_sha256": data_sha,
        "data": data,
        "source_sha256": source_sha,
        "source_files": files,
        "runtime": runtime,
        "world_size": world,
        "model_parameters": params,
    }


def bind(value: object, expected: dict[str, Any], label: str) -> None:
    """Reject a terminal or score producer with a different fit identity."""
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{label} does not bind fit config/source/data/runtime")


def validate_epoch_index(
    index: dict[str, Any], expected: dict[str, Any], epochs: int, complete: bool
) -> None:
    """Validate PR75's content-addressed retained-epoch index metadata."""
    required = {
        "schema",
        "identity",
        "completed_epochs",
        "global_step",
        "previous_index_sha256",
        "entries",
        "index_sha256",
    }
    body = {k: v for k, v in index.items() if k != "index_sha256"}
    count, entries = index.get("completed_epochs"), index.get("entries")
    if set(index) != required or index.get("schema") != "wnn.sequence-epoch-index.v1":
        raise ValueError("epoch-index schema differs from PR75")
    if hashlib.sha256(_canonical(body)).hexdigest() != index.get("index_sha256"):
        raise ValueError("epoch-index content hash mismatch")
    bind(index.get("identity"), expected, "epoch index")
    if type(count) is not int or not 1 <= count <= epochs:
        raise ValueError("epoch-index completed prefix is invalid")
    if not isinstance(entries, list) or len(entries) != count or (complete and count != epochs):
        raise ValueError("epoch index does not bind the configured completed prefix")
    previous = index.get("previous_index_sha256")
    if count == 1 and previous is not None:
        raise ValueError("first epoch index cannot name a previous index")
    if count > 1 and (not isinstance(previous, str) or not SHA.fullmatch(previous)):
        raise ValueError("epoch index omits its previous-index digest")
    last_step = 0
    for position, row in enumerate(entries, 1):
        keys = {"epoch", "completed_epochs", "global_step", "checkpoint_path", "checkpoint_sha256"}
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("epoch-index entry schema is invalid")
        digest, step = row["checkpoint_sha256"], row["global_step"]
        if type(row["epoch"]) is not int or row["epoch"] != position - 1:
            raise ValueError("epoch-index entries are not sequential")
        if type(row["completed_epochs"]) is not int or row["completed_epochs"] != position:
            raise ValueError("epoch-index completion counters are invalid")
        if not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise ValueError("epoch-index checkpoint digest is malformed")
        expected_path = f"epoch_checkpoints/epoch-{position:06d}-{digest}.pt"
        if row["checkpoint_path"] != expected_path:
            raise ValueError("epoch-index checkpoint path is not content-addressed")
        if type(step) is not int or step <= last_step:
            raise ValueError("epoch-index steps are not strictly increasing")
        last_step = step
    if type(index.get("global_step")) is not int or index["global_step"] != last_step:
        raise ValueError("epoch-index frontier differs from its entries")


def training_rows(raw: bytes, frontier: int | None) -> list[tuple[int, float, bytes]]:
    """Extract unique complete aggregate training rows through a safe frontier."""
    rows: dict[int, tuple[float, bytes]] = {}
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        row = _object(line, "metrics.jsonl row")
        epoch, loss = row.get("epoch"), row.get("loss")
        if type(epoch) is not int or epoch < 1:
            raise ValueError("training metric epoch is invalid")
        if frontier is not None and epoch > frontier:
            continue
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError("training loss must be finite numeric")
        if epoch in rows:
            raise ValueError("duplicate aggregate training metric epoch")
        rows[epoch] = (float(loss), line)
    return [(epoch, value, line) for epoch, (value, line) in sorted(rows.items())]
