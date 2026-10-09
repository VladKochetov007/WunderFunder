"""Collect a finished per-fold parsed-cache CPU kernel's small receipts and write its pins.

Downloads only small hash-bound files (never the multi-GB cache arrays), checks the
source-stage receipt against the local split/registry/membership, and writes
``fold_pins/fold_<k>.json`` which the rolling driver uses to admit fold-k rounds.
Exits 0 without writing when the kernel is not complete yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
KAGGLE = str(ROOT / ".venv/bin/kaggle")
MULTIFOLD = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1"
PATTERN = (
    r"(^|/)(boot_marker\.json|final_status\.json|phase\.jsonl|source_stage_receipt\.json|failure\.json|"
    r"manifest\.json|source_verification\.json|reader_validation\.json|reader_profile\.json|"
    r"membership\.json|cache_build\.log)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(root: Path, name: str, parent: str | None = None) -> Path:
    hits = [p for p in root.rglob(name) if parent is None or p.parent.name == parent]
    if len(hits) != 1:
        raise SystemExit(f"expected one {name} under {root}, found {len(hits)}")
    return hits[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    fold = args.fold
    source = json.loads((ROOT / MULTIFOLD / "FOLD_SOURCES.json").read_text())[str(fold)]
    kernel = source["kernel"]
    out_path = HERE / "fold_pins" / f"fold_{fold}.json"
    unsupported = HERE / "fold_pins" / f"fold_{fold}.unsupported"
    if out_path.exists() or unsupported.exists() or list((HERE / "fold_pins").glob(f"fold_{fold}.disabled_*")):
        return 0
    status = subprocess.run([KAGGLE, "kernels", "status", kernel], capture_output=True, text=True, timeout=120)
    if "KernelWorkerStatus.COMPLETE" not in status.stdout:
        print(status.stdout.strip())
        return 0
    dest = ROOT / MULTIFOLD / f"cpu_f{fold}_s{source['seed']}_terminal"
    dest.mkdir(parents=True, exist_ok=True)
    download = subprocess.run(
        [KAGGLE, "kernels", "output", kernel, "-p", str(dest), "--file-pattern", PATTERN, "-o"],
        capture_output=True, text=True, timeout=900,
    )
    if download.returncode != 0:
        raise SystemExit(download.stdout + download.stderr)
    manifest = one(dest, "manifest.json", "parsed_cache")
    cache = manifest.parent
    receipt = one(dest, "source_stage_receipt.json")
    phase = [p for p in dest.rglob("phase.jsonl") if p.parent == receipt.parent.parent]
    if len(phase) != 1:
        raise SystemExit("phase.jsonl beside the kernel output root is missing")
    split_rel = f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    registry_rel = f"{MULTIFOLD}/portable_inputs_fold{fold}.json"
    membership_rel = source["membership"]
    pins = {
        "SOURCE_KERNEL": kernel,
        "EXPECTED_SOURCE_STAGE_RECEIPT_SHA256": sha256(receipt),
        "EXPECTED_PHASE_LOG_SHA256": sha256(phase[0]),
        "EXPECTED_CACHE_MANIFEST_SHA256": sha256(manifest),
        "EXPECTED_CACHE_KEY": json.loads(manifest.read_text())["cache_key"],
        "EXPECTED_CPU_VALIDATION_SHA256": sha256(cache / "reader_validation.json"),
        "EXPECTED_CPU_PROFILE_SHA256": sha256(cache / "reader_profile.json"),
        "EXPECTED_SOURCE_VERIFICATION_SHA256": sha256(cache / "source_verification.json"),
        "EXPECTED_SPLIT_SHA256": sha256(ROOT / split_rel),
        "EXPECTED_REGISTRY_SHA256": sha256(ROOT / registry_rel),
        "EXPECTED_MEMBERSHIP_SHA256": sha256(ROOT / membership_rel),
        "SPLIT_RELATIVE": split_rel,
        "REGISTRY_RELATIVE": registry_rel,
        "MEMBERSHIP_RELATIVE": membership_rel,
    }
    stage = json.loads(receipt.read_text())
    checks = {
        "state": "complete",
        "cache_manifest_sha256": pins["EXPECTED_CACHE_MANIFEST_SHA256"],
        "cache_key": pins["EXPECTED_CACHE_KEY"],
        "reader_validation_sha256": pins["EXPECTED_CPU_VALIDATION_SHA256"],
        "reader_profile_sha256": pins["EXPECTED_CPU_PROFILE_SHA256"],
        "raw_source_verification_sha256": pins["EXPECTED_SOURCE_VERIFICATION_SHA256"],
        "split_manifest_sha256": pins["EXPECTED_SPLIT_SHA256"],
        "portable_registry_sha256": pins["EXPECTED_REGISTRY_SHA256"],
        "membership_sha256": pins["EXPECTED_MEMBERSHIP_SHA256"],
        "gpu_enabled": False,
    }
    bad = {k: (stage.get(k), v) for k, v in checks.items() if stage.get(k) != v}
    if bad:
        raise SystemExit(f"source-stage receipt disagrees with local pins: {bad}")
    membership = json.loads((ROOT / membership_rel).read_text())
    fit_train = membership["member_counts"]["fit_train"]
    batches = -(-fit_train // 32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if batches * 10 * 2 != 220:
        unsupported.write_text(f"fit sample gives {batches * 20} updates, not the worker's fixed 220\n")
        print(unsupported.read_text().strip())
        return 0
    out_path.write_text(json.dumps({
        "fold": fold,
        "pins": pins,
        "fit_train_count": fit_train,
        "evaluation_count": membership["member_counts"]["evaluation_valid"],
        "terminal_dir": str(dest.relative_to(ROOT)),
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
