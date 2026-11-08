from __future__ import annotations

import hashlib
import json
from pathlib import Path

from l4_wave_evidence import POSTFIT_SOURCE, REFERENCE, ROWS, SPLIT

from scripts.sequence_progress_report import _finite, _object, _sha, canonical_sha


def admission(path: Path | None) -> dict:
    """Require ROOT's exact separately admitted postfit implementation before grades count."""
    if path is None:
        return {}
    value = _object(path)
    if (
        value.get("approved") is not True
        or value.get("reviewer") != "ROOT"
        or value.get("reference_manifest_sha256") != REFERENCE
        or value.get("source_sha256") != POSTFIT_SOURCE
    ):
        raise ValueError("fixed-readout source/reference admission differs")
    return value


def grade(receipt: dict, endpoint: str, row: dict, index: dict, approved: dict) -> bool:
    """Verify a producer's fixed readout and its exact native fit/checkpoint ancestry."""
    if not approved or not row["fit_complete"]:
        return False
    producer, terminal = receipt.get("producer", {}), row["terminal"]
    metrics = receipt.get("metrics", {})
    if (
        receipt.get("schema") != "wnn.fixed-swa-e6-readout.v1"
        or receipt.get("complete") is not True
        or receipt.get("readout") != endpoint
        or receipt.get("split_sha256") != SPLIT
        or receipt.get("sequence_count") != 1873
        or receipt.get("selected_rows") != ROWS
        or receipt.get("reference", {}).get("manifest_sha256") != REFERENCE
        or receipt.get("mixture_weights") != {"reference": 0.9, "candidate": 0.1}
        or receipt.get("readout_implementation", {}).get("sha256") != approved["source_sha256"]
        or canonical_sha(receipt.get("config", {})) != row["ids"]["config"]
        or producer.get("identity") != terminal["identity"]
        or _sha(receipt.get("checkpoint_sha256")) is None
    ):
        return False
    for label in ("candidate", "mixture", "reference"):
        result = metrics.get(label, {})
        if (
            result.get("metric_id") != "global_wp_clip2_v1"
            or result.get("selected_rows") != ROWS
            or not all(_finite(result.get(k)) for k in ("wp", "t0", "t1", "wmse"))
        ):
            return False
    if abs(metrics["reference"]["wp"] - 0.6923774248169172) > 1e-10:
        return False
    entries = receipt.get("sequences", [])
    if (
        len(entries) != 1873
        or len({(e.get("source"), e.get("row_group")) for e in entries}) != 1873
        or any(e.get("source") != "valid" or _sha(e.get("sha256")) is None for e in entries)
        or sum(e.get("selected_rows", -1) for e in entries) != ROWS
    ):
        return False
    if endpoint == "epoch6":
        return receipt["checkpoint_sha256"] == terminal["model_sha256"]
    swa = producer.get("swa", {})
    parents = swa.get("parents", [])
    if (
        swa.get("epochs") != [3, 4, 5, 6]
        or swa.get("weights") != [0.25] * 4
        or swa.get("inference_only") is not True
        or swa.get("implementation_source", {}).get("sha256") != approved["source_sha256"]
        or index.get("completed_epochs") != 6
        or index.get("identity") != terminal["identity"]
        or index.get("index_sha256")
        != canonical_sha({k: v for k, v in index.items() if k != "index_sha256"})
        or len(parents) != 4
    ):
        return False
    return all(
        parent.get("completed_epochs") == epoch
        and parent.get("checkpoint_sha256") == index["entries"][epoch - 1]["checkpoint_sha256"]
        and parent.get("index_sha256") == index.get("index_sha256")
        and parent.get("identity_sha256") == canonical_sha(terminal["identity"])
        for epoch, parent in zip((3, 4, 5, 6), parents, strict=True)
    )


def observe_readouts(root: Path | None, fit_root: Path, row: dict, approved: dict) -> dict:
    """Read exact local copies of source receipts; no prediction or checkpoint upload claim."""
    result = {}
    if root is None:
        return result
    cell = row["cell"]
    index = _object(fit_root / f"l4-{cell}-fit/epoch_index.json")
    for endpoint in ("swa3456", "epoch6"):
        folder = root / cell / endpoint
        if folder.is_symlink() or folder.parent.is_symlink():
            raise ValueError("readout directory cannot be a symlink")
        receipt = _object(folder / "readout.json")
        if receipt and grade(receipt, endpoint, row, index, approved):
            raw = (folder / "readout.json").read_bytes()
            if json.loads(raw) != receipt:
                raise ValueError("readout receipt changed while being observed")
            result[endpoint] = {
                "metrics": receipt["metrics"],
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
    return result
