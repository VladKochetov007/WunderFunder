from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    source_root = Path(__file__).resolve().parent / "source"
    trainer = source_root / "scripts" / "train_rnn.py"
    with tempfile.TemporaryDirectory(prefix="h232-args-smoke-") as temporary:
        root = Path(temporary)
        split_path = root / "fold0.json"
        trace_path = root / "profile_update_trace.json"
        command = [
            sys.executable,
            str(trainer),
            "--name",
            "h232-path-serialization-smoke",
            "--device",
            "cuda",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--hidden",
            "384",
            "--layers",
            "2",
            "--seed",
            "3",
            "--train-source",
            "train+valid",
            "--train-seqs",
            "10607",
            "--split-manifest",
            str(split_path),
            "--valid-fold",
            "0",
            "--epochs",
            "4",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "3",
            "--no-eval-during-fit",
            "--batch",
            "8",
            "--chunk",
            "2000",
            "--precision",
            "fp16",
            "--profile-warmup",
            "3",
            "--profile-steps",
            "20",
            "--profile-trace-output",
            str(trace_path),
            "--amp-init-scale",
            "4096.0",
            "--serialization-smoke",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root / "src")
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env=environment,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"trainer argv JSON smoke failed ({result.returncode}):\n"
                f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise AssertionError(f"expected one JSON line before device setup, got {len(lines)}")
        decoded = json.loads(lines[0])
        assert decoded["profile_trace_output"] == str(trace_path)
        assert decoded["split_manifest"] == str(split_path)
        assert decoded["serialization_smoke"] is True
        assert decoded["amp_init_scale"] == 4096.0
        assert not trace_path.exists(), "smoke mode must not enter a training profile"
        print(
            json.dumps(
                {
                    "status": "passed",
                    "parsed_path_options": ["profile_trace_output", "split_manifest"],
                    "actual_profile_trace_path": decoded["profile_trace_output"],
                    "training_entered": False,
                    "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                    "return_code": result.returncode,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
