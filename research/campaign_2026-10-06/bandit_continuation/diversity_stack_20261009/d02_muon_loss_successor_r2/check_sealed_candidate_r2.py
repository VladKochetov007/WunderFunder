"""Validate the exact D02 R2 source payload and bootstrap without GPU work."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    packet = json.loads((root / "gpu_loss_pair.packet.json").read_text(encoding="utf-8"))
    kernel = root / "kernel" / packet["kernel"]["code_file"]
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == packet["kernel"]["code_file_sha256"]
    admission = json.loads((root / "root_admission.json").read_text(encoding="utf-8"))
    assert admission["push_allowed"] is False
    assert admission["conditional_fit_allowed"] is False
    assert admission["gpu_control_contract_sha256"] is None
    assert admission["control_gpu_kernel"].endswith("-v4")
    rows = []
    with tempfile.TemporaryDirectory(prefix="wnn-d02-r2-offline-") as temporary:
        work = Path(temporary)
        environment = os.environ.copy()
        environment.update(
            KAGGLE_INPUT=str(root / "source_asset"),
            KAGGLE_WORKING=str(work),
            WNN_D02_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
        )
        bootstrap = subprocess.run(
            [sys.executable, str(kernel)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert not bootstrap.stderr
        offline_receipt = json.loads((work / "offline_source_receipt.json").read_text())
        assert offline_receipt["status"] == "pass"
        assert offline_receipt["fit_admitted"] is False
        extracted = work / "wnn"
        environment["PYTHONPATH"] = str(extracted / "src")
        entrypoints = {
            "loss_pair_gpu_worker.py": "import",
            "train_rnn.py": "help",
            "predict_run.py": "help",
        }
        for name, check in entrypoints.items():
            entrypoint = extracted / "scripts" / name
            if check == "import":
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1], run_name='sealed_import_check')",
                    str(entrypoint),
                ]
            else:
                command = [sys.executable, str(entrypoint), "--help"]
            result = subprocess.run(
                command,
                cwd=extracted,
                env=environment,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if result.returncode:
                (root / "SEALED_OFFLINE_FAILURE_R2.json").write_text(
                    json.dumps(
                        {
                            "entrypoint": name,
                            "returncode": result.returncode,
                            "stderr": result.stderr,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise RuntimeError(result.stderr)
            assert not result.stderr, result.stderr
            if check == "help":
                assert "usage:" in result.stdout
            rows.append(
                {
                    "entrypoint": name,
                    "returncode": result.returncode,
                    "check": check,
                    "stderr_bytes": len(result.stderr),
                    "gpu_work": False,
                }
            )
    receipt = {
        "state": "pass",
        "source_commit": packet["source"]["source_commit"],
        "source_code_manifest_sha256": packet["source"]["code_manifest_sha256"],
        "bootstrap_sha256": packet["kernel"]["code_file_sha256"],
        "control_kernel": packet["control_gpu_kernel"],
        "control_contract_sha256": admission["gpu_control_contract_sha256"],
        "checks": rows,
        "actual_device_validation": False,
    }
    (root / "SEALED_OFFLINE_CHECK_R2.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
