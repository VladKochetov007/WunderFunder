"""Package one graded endpoint for the pinned MIDAS training-evidence importer."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from midas.training_evidence_schema import validate_external_receipt

PACKET = Path(__file__).resolve().parents[1]
CAMPAIGN = PACKET.name
RUNTIME = "8626000a4b965763a70011a0291a891aa6535d7e"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
ARMS = ("plain", "angular", "radial")
CELLS = {f"{arm}_s{seed}" for arm in ARMS for seed in (209401, 209402)}


def read_bytes(path: Path) -> bytes:
    """Read a regular metadata file; keep the producer's exact byte encoding."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
        raise ValueError(f"Missing or oversized metadata file: {path}")
    return path.read_bytes()


def save(root: Path, name: str, raw: bytes, expected: str | None = None) -> dict[str, str]:
    """Write one file and bind it to the comparison digest when provided."""
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and expected != digest:
        raise ValueError(f"Published panel digest differs for {name}")
    with (root / name).open("xb") as stream:
        stream.write(raw)
    return {"path": name, "sha256": digest}


def write_bundle(root: Path, cell: str, split: Path, dest: Path) -> None:
    """Build the portable receipt from a complete six-cell readout, then validate it."""
    panel = json.loads(read_bytes(root / "comparison.json"))
    if (
        panel["schema"] != "wnn.query-geometry.comparison.v1"
        or panel["status"] != "complete_exposed_fixed_e4"
        or panel["runtime_commit"] != RUNTIME
        or set(panel["receipts"]) != CELLS
    ):
        raise ValueError("Expected the complete frozen perturbation panel")
    identity = panel["receipts"][cell]
    if identity["split_sha256"] != SPLIT:
        raise ValueError("Comparison references a different split")
    run = root / cell
    meta_raw = read_bytes(run / "fit/run_metadata.json")
    meta = json.loads(meta_raw)
    grade_raw = read_bytes(run / "score.json")
    grade = json.loads(grade_raw)
    config = meta["config"]
    arm, seed = cell.rsplit("_s", 1)
    modes = {"plain": "none", "angular": "angular", "radial": "radial"}
    if (
        config["model"]["query_perturbation"] != modes[arm]
        or config["seed"] != int(seed)
        or identity["source_sha256"] != meta["source"]["sha256"]
        or grade["predictions_manifest_sha256"] != identity["predictions_manifest_sha256"]
        or grade["producer"]["checkpoint_sha256"] != identity["checkpoint_sha256"]
    ):
        raise ValueError("The requested cell, endpoint or producer differs from the readout")
    config_raw = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config_ref = save(dest, "config.json", config_raw, identity["config_sha256"])
    score_ref = save(dest, "score.json", grade_raw, identity["score_sha256"])
    terminal = save(
        dest,
        "terminal_status.json",
        read_bytes(run / "fit/terminal_status.json"),
        identity["terminal_status_sha256"],
    )
    checkpoint = identity["checkpoint_sha256"]
    receipt = {
        "schema": "midas.external-training-receipt.v1",
        "provider": "b300-team",
        "external_run_id": f"{CAMPAIGN}:{cell}:{checkpoint}",
        "campaign": {"id": CAMPAIGN, "stage": "B"},
        "experiment": {"cell_id": cell, "variant": arm, "fold": 0, "seed": int(seed)},
        "source": {
            "repository": "VladKochetov007/WunderFunder",
            "commit": panel["packet_commit"],
            "runtime_source_sha256": identity["source_sha256"],
        },
        "execution": {"status": "completed", "started_at": None, "finished_at": None},
        "terminal_status": terminal,
        "run_metadata": save(dest, "run_metadata.json", meta_raw),
        "config": config_ref,
        "split": save(dest, "split.json", read_bytes(split), SPLIT) | {"fold": 0},
        "metrics": [
            {
                "epoch": 4,
                "scope": "outer_validation",
                "name": "global_wp_clip2_v1.wp",
                "value": grade["score"]["wp"],
                "split_sha256": SPLIT,
                "checkpoint_sha256": checkpoint,
                "producer_path": score_ref["path"],
                "producer_sha256": score_ref["sha256"],
            }
        ],
        "checkpoints": [
            {
                "epoch": 4,
                "path": None,
                "reference": "model.pt",
                "sha256": checkpoint,
                "bytes": None,
                "parent_sha256": None,
            }
        ],
    }
    validate_external_receipt(receipt, evidence_root=dest)
    save(dest, "receipt.json", (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())


def main() -> None:
    """Write a new bundle under the run directory, with no checkpoint or dataset copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    target = run_root / args.cell / "midas-evidence"
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staging = Path(temp) / "evidence"
        staging.mkdir()
        write_bundle(run_root, args.cell, args.split_manifest, staging)
        staging.rename(target)


if __name__ == "__main__":
    main()
