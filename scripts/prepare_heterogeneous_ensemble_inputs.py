"""Bind actual completed producer artifacts for the frozen twelve-row ensemble table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heterogeneous_ensemble_contract import (  # noqa: E402
    DEFAULT_ENSEMBLES,
    RESERVED_ROLES,
    SCHEMA,
    SPLIT_SHA256,
    read_object,
)

from wnn.chronological_source_index import OFFICIAL_SOURCE_PINS  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

# Native sidecars and checkpoint provenance are heterogeneous JSON metadata.
JsonObject = dict[str, Any]
LEGACY_ALIASES = ("w18", "tiedlead_s5", "tiedlead_s6", "fbc", "fin256_s3", "fin256_s4")
NEW_ROLES = {
    "lstm": ("het6_lstm_s1201", 3),
    "tcn": ("het6_tcn_s1202", 3),
    "control_s1301": ("het6_control_s1301", 3),
    "control_s1302": ("het6_control_s1302", 3),
    "hetero_s1301": ("het6_heterogeneous_s1301", 3),
    "hetero_s1302": ("het6_heterogeneous_s1302", 3),
}


def file_pin(path: Path) -> JsonObject:
    """Pin bytes without changing the producer's physical schema or file name."""
    return {"path": str(path.resolve(strict=True)), "sha256": file_sha256(path)}


def pending(reason: str) -> JsonObject:
    """Represent an unavailable endpoint explicitly, never as a zero-valued model."""
    return {"state": "pending", "reason": reason}


def collect_legacy_member(root: Path, run: str, epoch: int) -> JsonObject:
    """Bind a completed bank and its original runtime checkpoint/provenance sidecars."""
    paths = {
        "predictions": root / "cache" / f"pred_{run}_valid.npy",
        "metadata": root / "cache" / f"pred_{run}_valid.json",
        "checkpoint": root / "artifacts/runs" / run / "model.pt",
        "provenance": root / "artifacts/runs" / run / "provenance.json",
    }
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return pending("producer artifacts not complete: " + ", ".join(missing))
    try:
        metadata, provenance = read_object(paths["metadata"]), read_object(paths["provenance"])
    except json.JSONDecodeError:
        return pending("producer JSON write is not complete")
    if provenance.get("selected_epoch") != epoch:
        return pending(f"fixed endpoint epoch{epoch} is not selected yet")
    result: JsonObject = {"kind": "legacy_bank", "run": run, "epoch": epoch}
    result.update({name: file_pin(path) for name, path in paths.items()})
    expected = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "inference_precision": "fp32",
        "checkpoint_file": "model.pt",
        "split_manifest_sha256": SPLIT_SHA256,
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "prediction_sha256": result["predictions"]["sha256"],
        "run_provenance_sha256": result["provenance"]["sha256"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"completed producer pointer differs from actual frozen artifacts: {run}")
    return result


def collect_galam_probe(base: Path) -> JsonObject:
    """Pin the original scored-only probe and its exact human-epoch3 checkpoint."""
    run = "g2-galam-lstm128"
    paths = {
        "metadata": base / run / "receipt.json",
        "predictions": base / run / "scored-predictions.npy",
        "plan": base / "plan.json",
        "result": base / "result.json",
    }
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return pending("original GALAM scored probe is unavailable: " + ", ".join(missing))
    try:
        receipt = read_object(paths["metadata"])
    except json.JSONDecodeError:
        return pending("original GALAM receipt JSON is not complete")
    endpoint = receipt.get("endpoint", {})
    if endpoint.get("cell") != run or endpoint.get("epoch") != 3:
        raise ValueError("GALAM role must remain the original human-epoch3/native-epoch2 probe")
    for name in ("checkpoint", "epoch_index"):
        value = endpoint.get(name)
        if not isinstance(value, str):
            raise ValueError(f"original GALAM receipt omits {name}")
        paths[name] = Path(value)
        if not paths[name].is_file():
            return pending(f"original GALAM {name} is unavailable")
    result: JsonObject = {
        "kind": "galam_scored_probe",
        "run": run,
        "epoch": 2,
        "epoch_index_sha256": endpoint["epoch_index_sha256"],
    }
    result.update({name: file_pin(path) for name, path in paths.items()})
    if result["checkpoint"]["sha256"] != endpoint.get("checkpoint_sha256") or result["predictions"][
        "sha256"
    ] != receipt.get("prediction_sha256"):
        raise ValueError("original GALAM checkpoint/prediction bytes differ from its receipt")
    return result


def prepare(root: Path, legacy: Path, split: Path, galam: Path, output: Path) -> JsonObject:
    """Write one create-only artifact manifest; the downstream grader performs full admission."""
    if output.exists():
        raise FileExistsError("preserve prior table inputs; choose a fresh output path")
    split_pin = file_pin(split)
    if split_pin["sha256"] != SPLIT_SHA256:
        raise ValueError("table split differs from the frozen chronological primary manifest")
    validation = file_pin(root / "datasets/valid.parquet")
    if validation["sha256"] != OFFICIAL_SOURCE_PINS["valid"].sha256:
        raise ValueError("table validation bytes are not the admitted official source")
    members = {
        alias: collect_legacy_member(legacy, *RESERVED_ROLES[alias]) for alias in LEGACY_ALIASES
    }
    members.update({alias: collect_legacy_member(root, *role) for alias, role in NEW_ROLES.items()})
    members["galam"] = collect_galam_probe(galam)
    manifest = {
        "schema": SCHEMA,
        "baseline": "TL3",
        "data_root": str(root.resolve(strict=True)),
        "split": split_pin,
        "validation": validation,
        "members": members,
        "ensembles": DEFAULT_ENSEMBLES,
        "scope": "actual producer-artifact binding only; source/ancestry/mask admission belongs to the downstream grader",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "output": str(output.resolve()),
        "manifest_sha256": file_sha256(output),
        "ready": [name for name, record in members.items() if record.get("state") != "pending"],
        "pending": {
            name: record["reason"]
            for name, record in members.items()
            if record.get("state") == "pending"
        },
        "ensemble_rows": len(DEFAULT_ENSEMBLES),
        "new_fits": 0,
    }


def main() -> None:
    """Collect existing native assets only; never infer, train or manufacture sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--galam-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.root, args.legacy_root, args.split, args.galam_probe, args.output))
    )


if __name__ == "__main__":
    main()
