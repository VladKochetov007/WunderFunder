"""Offline-check the admission-embedded D02 R4 payload without starting a fit."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

KERNEL_ID = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v4"
CONTROL_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _run(
    command: list[str], cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=65,
        check=False,
    )


def check_candidate(candidate: Path, source_asset: Path | None = None) -> dict[str, Any]:
    candidate = candidate.resolve()
    packet_path = candidate / "gpu_loss_pair.packet.json"
    admission_path = candidate / "root_admission.json"
    finalization_path = candidate / "finalization_receipt.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    admission_raw = admission_path.read_bytes()
    admission = json.loads(admission_raw)
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    kernel_path = candidate / "kernel" / packet["kernel"]["code_file"]
    bootstrap = kernel_path.read_bytes()

    if (
        finalization.get("state") != "finalized_for_root_review_only"
        or finalization.get("final_packet_sha256") != _sha256_file(packet_path)
        or finalization.get("final_bootstrap_sha256") != _sha256_file(kernel_path)
    ):
        raise ValueError("finalization receipt does not bind the packet and bootstrap")
    if _sha256_file(kernel_path) != packet["kernel"]["code_file_sha256"]:
        raise ValueError("admitted bootstrap hash differs from the final packet")
    if packet.get("authorized_for_execution") is not False:
        raise ValueError("offline candidate must still await exact kernel push admission")
    if admission.get("schema") != "wnn.d02.root-admission.v1":
        raise ValueError("embedded execution admission schema mismatch")
    if (
        admission.get("kernel_id") != KERNEL_ID
        or admission.get("push_allowed") is not True
        or admission.get("conditional_fit_allowed") is not True
    ):
        raise ValueError("the admitted D02 execution flags are not embedded")
    if (
        admission.get("source_code_manifest_sha256") != packet["source"]["code_manifest_sha256"]
        or admission.get("control_gpu_kernel") != CONTROL_KERNEL
        or admission.get("gpu_control_contract_sha256") != CONTROL_CONTRACT_SHA256
    ):
        raise ValueError("embedded admission is not bound to the frozen source/control")
    if packet.get("root_admission_sha256") != hashlib.sha256(admission_raw).hexdigest():
        raise ValueError("packet root-admission hash does not match the embedded receipt")
    admission_base64 = base64.b64encode(admission_raw)
    if admission_base64 not in bootstrap:
        raise ValueError("final bootstrap does not embed the exact admitted bytes")

    asset_root = (source_asset or candidate / "source_asset").resolve()
    if _sha256_file(asset_root / "assets-manifest.json") != packet["source"][
        "asset_manifest_sha256"
    ]:
        raise ValueError("source asset manifest differs from the sealed packet")
    manifest = json.loads((asset_root / "assets-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != packet["source"]["source_dataset"]:
        raise ValueError("source asset manifest dataset ID mismatch")
    rows_in_manifest = manifest.get("files", [])
    if manifest.get("private") is not True or manifest.get("version") != 1:
        raise ValueError("source asset manifest privacy/version mismatch")
    if len(rows_in_manifest) != 5:
        raise ValueError("source asset manifest must have exactly five payload rows")
    for row in rows_in_manifest:
        payload = asset_root / row["filename"]
        if payload.stat().st_size != int(row["size"]) or _sha256_file(payload) != row["sha256"]:
            raise ValueError(f"source asset payload does not match manifest: {row['path']}")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="wnn-d02-r4-admitted-offline-") as temp:
        working = Path(temp)
        environment = os.environ.copy()
        environment.update(
            {
                "KAGGLE_INPUT": str(asset_root),
                "KAGGLE_WORKING": str(working),
                "WNN_D02_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        result = _run([sys.executable, str(kernel_path)], candidate, environment)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout[-4000:])
        offline_receipt = json.loads((working / "offline_source_receipt.json").read_text())
        if (
            offline_receipt.get("status") != "pass"
            or offline_receipt.get("fit_admitted") is not False
            or offline_receipt.get("code_manifest_sha256")
            != packet["source"]["code_manifest_sha256"]
        ):
            raise ValueError("admitted bootstrap source-only execution receipt mismatch")

        extracted = working / "wnn"
        environment["PYTHONPATH"] = str(extracted / "src")
        for filename, mode in (
            ("loss_pair_gpu_worker.py", "import"),
            ("train_rnn.py", "help"),
            ("predict_run.py", "help"),
        ):
            script = extracted / "scripts" / filename
            command = (
                [
                    sys.executable,
                    "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1], run_name='sealed_import_check')",
                    str(script),
                ]
                if mode == "import"
                else [sys.executable, str(script), "--help"]
            )
            result = _run(command, extracted, environment)
            if result.returncode != 0:
                raise RuntimeError(f"{filename}: {result.stderr or result.stdout[-4000:]}")
            rows.append(
                {
                    "entrypoint": filename,
                    "check": mode,
                    "returncode": result.returncode,
                    "stderr_bytes": len(result.stderr),
                    "gpu_work": False,
                }
            )

    return {
        "schema": "wnn.d02.r4.admitted-offline-check.v1",
        "state": "pass",
        "packet_sha256": _sha256_file(packet_path),
        "bootstrap_sha256": _sha256_file(kernel_path),
        "embedded_root_admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
        "source_code_manifest_sha256": packet["source"]["code_manifest_sha256"],
        "source_asset_manifest_sha256": packet["source"]["asset_manifest_sha256"],
        "entrypoints": rows,
        "execution_admission_embedded": True,
        "exact_kernel_push_admission_pending": True,
        "provider_mutations": False,
        "actual_device_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-asset", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = check_candidate(args.candidate, args.source_asset)
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
