"""Exercise the exact D02 R3 bootstrap and source archive without GPU work."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(
    command: list[str], cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one bounded source-only startup check."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=65,
        check=False,
    )


def main() -> int:
    """Verify the sealed payload, bootstrap, imports and CLI parsers."""
    candidate = Path(__file__).resolve().parent
    packet = json.loads((candidate / "gpu_loss_pair.packet.json").read_text(encoding="utf-8"))
    admission = json.loads((candidate / "root_admission.json").read_text(encoding="utf-8"))
    kernel_path = candidate / "kernel" / packet["kernel"]["code_file"]
    assert (
        hashlib.sha256(kernel_path.read_bytes()).hexdigest() == packet["kernel"]["code_file_sha256"]
    )
    assert admission["push_allowed"] is False
    assert admission["conditional_fit_allowed"] is False
    assert (
        admission["gpu_control_contract_sha256"]
        == packet["r5_control_parent"]["training_contract"]["sha256"]
    )
    assert admission["control_gpu_kernel"] == packet["control_gpu_kernel"]
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="wnn-d02-r3-offline-") as temporary:
        working = Path(temporary)
        environment = os.environ.copy()
        environment.update(
            {
                "KAGGLE_INPUT": str(candidate / "source_asset"),
                "KAGGLE_WORKING": str(working),
                "WNN_D02_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        bootstrap = _run([sys.executable, str(kernel_path)], candidate, environment)
        if bootstrap.returncode != 0:
            raise RuntimeError(bootstrap.stderr or bootstrap.stdout)
        offline_receipt = json.loads((working / "offline_source_receipt.json").read_text())
        assert offline_receipt["status"] == "pass"
        assert offline_receipt["fit_admitted"] is False
        extracted = working / "wnn"
        environment["PYTHONPATH"] = str(extracted / "src")
        entrypoints = {
            "loss_pair_gpu_worker.py": "import",
            "train_rnn.py": "help",
            "predict_run.py": "help",
        }
        for name, check in entrypoints.items():
            script = extracted / "scripts" / name
            if check == "import":
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1], run_name='sealed_import_check')",
                    str(script),
                ]
            else:
                command = [sys.executable, str(script), "--help"]
            result = _run(command, extracted, environment)
            if result.returncode != 0:
                failure = {
                    "entrypoint": name,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "stdout_tail": result.stdout[-4000:],
                }
                (candidate / "SEALED_OFFLINE_FAILURE_R3.json").write_text(
                    json.dumps(failure, indent=2) + "\n", encoding="utf-8"
                )
                raise RuntimeError(result.stderr or result.stdout[-4000:])
            assert not result.stderr, result.stderr
            if check == "help":
                assert "usage:" in result.stdout.lower()
            rows.append(
                {
                    "entrypoint": name,
                    "check": check,
                    "returncode": result.returncode,
                    "gpu_work": False,
                    "stderr_bytes": len(result.stderr),
                }
            )

    receipt = {
        "schema": "wnn.d02.r3-sealed-offline-check.v1",
        "state": "pass",
        "source_commit": packet["source"]["source_commit"],
        "source_code_manifest_sha256": packet["source"]["code_manifest_sha256"],
        "bootstrap_sha256": packet["kernel"]["code_file_sha256"],
        "control_kernel": packet["control_gpu_kernel"],
        "control_contract_sha256": packet["r5_control_parent"]["training_contract"]["sha256"],
        "checks": rows,
        "actual_device_validation": False,
        "provider_mutations": False,
    }
    receipt_path = candidate / "SEALED_OFFLINE_CHECK_R3.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
