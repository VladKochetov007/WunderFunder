"""Check the exact D02 bootstrap and extracted CLI closure without GPU work."""

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
    packet = json.loads((root / "gpu_loss_pair.packet.json").read_text())
    kernel = root / "kernel" / "wnn-d02-muon-loss-gpu-20261009-v1.py"
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == packet["kernel"]["code_file_sha256"]
    admission = json.loads((root / "root_admission.json").read_text())
    assert admission["push_allowed"] is False
    assert admission["conditional_fit_allowed"] is False
    assert admission["control_gpu_kernel"].endswith("-v3")
    rows = []
    with tempfile.TemporaryDirectory(prefix="wnn-d02-offline-") as temporary:
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
        assert (work / "offline_source_receipt.json").is_file()
        extracted = work / "wnn"
        environment["PYTHONPATH"] = str(extracted / "src")
        for name in ("loss_pair_gpu_worker.py", "train_rnn.py"):
            entrypoint = extracted / "scripts" / name
            command = [sys.executable, str(entrypoint), "--help"]
            if name == "loss_pair_gpu_worker.py":
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1], run_name='sealed_import_check')",
                    str(entrypoint),
                ]
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
                (root / "SEALED_OFFLINE_FAILURE_R1.json").write_text(
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
            if name == "train_rnn.py":
                assert "usage:" in result.stdout
            rows.append(
                {
                    "entrypoint": name,
                    "returncode": result.returncode,
                    "check": "import" if name == "loss_pair_gpu_worker.py" else "help",
                    "stderr_bytes": len(result.stderr),
                    "gpu_work": False,
                }
            )
    receipt = {
        "state": "pass",
        "source_commit": packet["source"]["source_commit"],
        "bootstrap_sha256": packet["kernel"]["code_file_sha256"],
        "checks": rows,
        "actual_device_validation": False,
    }
    (root / "SEALED_OFFLINE_CHECK_R1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
