from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R2 = ROOT.parent / "d02_muon_loss_successor_r5_profile_export_r2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(bootstrap: Path, working: Path, *, offline: bool) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "KAGGLE_INPUT": str(R2 / "source_asset"),
            "KAGGLE_WORKING": str(working),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if offline:
        environment["WNN_D02_OFFLINE"] = "1"
    else:
        environment.pop("WNN_D02_OFFLINE", None)
    return subprocess.run(
        [sys.executable, str(bootstrap)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=65,
        check=False,
    )


def main() -> int:
    """Validate the exact pending source path and admitted startup guard."""
    packet_path = ROOT / "gpu_loss_pair.packet.json"
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(packet_bytes)
    bootstrap = ROOT / "kernel" / packet["kernel"]["code_file"]
    if _sha256(bootstrap) != packet["kernel"]["code_file_sha256"]:
        raise ValueError("R3 bootstrap differs from the sealed packet SHA")

    admission_bytes = (ROOT / "root_admission.json").read_bytes()
    admission = json.loads(admission_bytes)
    source_receipt_bytes = (ROOT / "ROOT_D02_R5_SOURCE_ADMISSION_R1.json").read_bytes()
    source_receipt = json.loads(source_receipt_bytes)
    if admission != source_receipt["root_admission_payload"]:
        raise ValueError("R3 embedded admission differs from the root receipt payload")
    if admission.get("kernel_id") != packet["packet_id"] or not admission.get("push_allowed"):
        raise ValueError("R3 admitted bootstrap does not bind the packet kernel ID")

    with tempfile.TemporaryDirectory(prefix="wnn-d02-r5-r3-offline-") as temporary:
        work = Path(temporary)
        offline_dir = work / "source_only"
        offline_dir.mkdir()
        offline = _run(bootstrap, offline_dir, offline=True)
        if offline.returncode != 0:
            raise RuntimeError(offline.stderr or offline.stdout)
        offline_receipt = json.loads((offline_dir / "offline_source_receipt.json").read_text())
        if (
            offline_receipt.get("status") != "pass"
            or offline_receipt.get("fit_admitted") is not False
        ):
            raise ValueError("R3 source-only offline bootstrap did not pass")

        admitted_dir = work / "admitted_gate"
        admitted_dir.mkdir()
        admitted = _run(bootstrap, admitted_dir, offline=False)
        if admitted.returncode != 1:
            raise ValueError("R3 admitted branch did not stop at the absent fold-0 fixture")
        failure_path = admitted_dir / "failure.json"
        failure = json.loads(failure_path.read_text())
        expected_boundary = "expected one fold-0 auxiliary asset manifest, found 0"
        if failure.get("error") != expected_boundary:
            raise ValueError(
                "R3 admitted branch failed before the expected fold-0 fixture boundary: "
                f"{failure.get('error')}"
            )
        if (admitted_dir / "bootstrap_receipt.json").exists() or (admitted_dir / "out").exists():
            raise ValueError("R3 must stop before worker launch at the fold-0 fixture boundary")

    receipt = {
        "schema": "wnn.d02.r5-kernel-successor-offline-check.v1",
        "state": "pass",
        "packet_sha256": _sha256(packet_path),
        "bootstrap_sha256": _sha256(bootstrap),
        "root_admission_sha256": _sha256(ROOT / "root_admission.json"),
        "root_source_admission_sha256": _sha256(ROOT / "ROOT_D02_R5_SOURCE_ADMISSION_R1.json"),
        "source_asset_manifest_sha256": _sha256(R2 / "source_asset" / "assets-manifest.json"),
        "pending_source_bootstrap": "pass",
        "admitted_branch": "passed admission validation; stopped at absent fold-0 fixture",
        "admitted_branch_boundary_error": expected_boundary,
        "admission_fixture_is_root_receipt_payload": True,
        "fit_or_provider_mutation": False,
        "actual_device_validation": False,
    }
    receipt_path = ROOT / "SEALED_OFFLINE_CHECK_R5_R2.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
