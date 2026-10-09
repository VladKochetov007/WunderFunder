from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = CANDIDATE_ROOT / "code"
WORKER_PATH = CODE_ROOT / "scripts" / "loss_pair_gpu_worker.py"
PREDICT_PATH = CODE_ROOT / "scripts" / "predict_run.py"
EXPECTED_CHECKPOINT_SHA256 = {
    "wmse": "71fcba3d9eaff3710f8fd42f3778ef74cd425f11977668a5575c21563a82c309",
    "huber_d1": "b61b5026386273b11db871f06f09eda7654af5814c066e850274e32d9cdfca77",
}


def _worker_module():
    sys.path.insert(0, str(CODE_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("d02_r2_gpu_worker", WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the candidate GPU worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r4_checkpoint_export_argv_is_parser_valid_without_device_flag(tmp_path: Path) -> None:
    """Check actual R4 checkpoints and parser-only argv; never run inference."""
    worker = _worker_module()
    for arm, expected_sha in EXPECTED_CHECKPOINT_SHA256.items():
        variable = f"WNN_D02_R4_{arm.upper()}_MODEL_DIR"
        model_dir_raw = os.environ.get(variable)
        if not model_dir_raw:
            pytest.skip(f"set {variable} to run the collected R4 checkpoint parser check")
        model_dir = Path(model_dir_raw).resolve(strict=True)
        checkpoint = model_dir / "model.pt"
        provenance_path = model_dir / "provenance.json"
        assert checkpoint.is_file()
        assert provenance_path.is_file()
        actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert actual_sha == expected_sha
        run_name = json.loads(provenance_path.read_text(encoding="utf-8"))["args"]["name"]

        export = worker._export_predictions(
            arm,
            run_name,
            "0",
            tmp_path / "manifest.json",
            tmp_path / "cache_validation.json",
            "f" * 64,
            tmp_path / arm,
        )
        command = export["command"]
        assert Path(command[1]) == PREDICT_PATH
        assert "--device" not in command
        result = subprocess.run(
            [*command, "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=CODE_ROOT,
            env=export["env"],
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "unrecognized arguments" not in result.stderr
        assert "name [split]" in result.stdout
        assert run_name in command
