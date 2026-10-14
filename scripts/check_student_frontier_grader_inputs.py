"""Check small frozen scorer sources and target dimensions before F0 grading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a bounded source or split file by streaming it from disk."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(
    manifest_path: Path,
    expected_manifest_sha256: str,
    consumer_contract_path: Path,
    expected_consumer_contract_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Verify grader code and split pins without rescanning the large target NPZ."""
    manifest_size = manifest_path.stat().st_size
    if manifest_size <= 0 or manifest_size > 16_000:
        raise ValueError("grader-input manifest exceeds its 16 KB bound")
    with manifest_path.open("rb") as source:
        raw = source.read(16_001)
    if len(raw) != manifest_size:
        raise ValueError("grader-input manifest exceeds its 16 KB bound")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("grader-input manifest differs from the sequence pin")
    manifest = json.loads(raw)
    contract_size = consumer_contract_path.stat().st_size
    if contract_size <= 0 or contract_size > 256_000:
        raise ValueError("consumer contract exceeds its 256 KB read bound")
    with consumer_contract_path.open("rb") as source:
        contract_raw = source.read(256_001)
    if len(contract_raw) != contract_size:
        raise ValueError("consumer contract changed while being read")
    consumer_contract_sha256 = hashlib.sha256(contract_raw).hexdigest()
    if consumer_contract_sha256 != expected_consumer_contract_sha256:
        raise ValueError("consumer contract differs from the sequence pin")
    grader = manifest["grader"]
    checked: dict[str, str] = {}
    for section, field in (
        ("prediction", "upstream_path"),
        ("grader", "score_fold0_panel"),
        ("grader", "evaluate_bandit_continuation"),
        ("grader", "metric"),
    ):
        if section == "prediction":
            entry = manifest[section]
            path = Path(entry[field])
            expected = entry["upstream_sha256"]
        else:
            entry = grader[field]
            path = Path(entry["path"])
            expected = entry["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen grader source changed: {path}")
        checked[str(path)] = actual

    prediction_copy = Path(manifest["ops_root"]) / manifest["prediction"]["pinned_copy_path"]
    copy_sha = sha256_file(prediction_copy)
    if copy_sha != manifest["prediction"]["pinned_copy_sha256"]:
        raise ValueError("frozen prediction-script copy changed")
    checked[str(prediction_copy)] = copy_sha

    target = grader["targets"]
    target_path = Path(target["path"])
    if target_path.stat().st_size != target["bytes"]:
        raise ValueError("frozen validation-target file size changed")
    split = manifest["split"]
    split_path = Path(manifest["project_root"]) / split["path"]
    if sha256_file(split_path) != split["sha256"]:
        raise ValueError("frozen fold-0 split manifest changed")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite grader-input receipt: {output_path}")
    receipt = {
        "format": "wnn.student-frontier-grader-input-check",
        "version": 1,
        "manifest_sha256": manifest_sha256,
        "consumer_contract_sha256": consumer_contract_sha256,
        "small_source_hashes": checked,
        "target_path": str(target_path),
        "target_bytes": target["bytes"],
        "target_sha256_was_streamed_once_at_manifest_freeze": target["sha256"],
        "target_rehashed_here": False,
        "split_sha256": split["sha256"],
        "status": "passed; target payload hash will be independently reported by each frozen scorer run",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Check grader inputs and emit a small immutable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--consumer-contract", type=Path, required=True)
    parser.add_argument("--consumer-contract-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = check_inputs(
        args.manifest.resolve(strict=True),
        args.manifest_sha256,
        args.consumer_contract.resolve(strict=True),
        args.consumer_contract_sha256,
        args.out.resolve(),
    )
    print(json.dumps({"manifest_sha256": receipt["manifest_sha256"], "status": receipt["status"]}))


if __name__ == "__main__":
    main()
