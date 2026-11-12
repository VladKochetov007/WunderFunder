"""Export four fixed-E6 chronological endpoints on their now-free L4 GPUs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data/lc-epanov_s/tabdata-final-quality-20261112-r1")
SOURCE = Path("/data2/lc-epanov_s/tabdata-all-followups-20261112-r1/source-r2")
DATA = Path("/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project")
HELPERS = "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src"
PYTHON = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"
CELLS = (
    "label_since32_k16_seed203401",
    "label_since32_k16_seed203402",
    "zero_since_control_k16_seed203401",
    "zero_since_control_k16_seed203402",
)


def main() -> None:
    """Use the canonical exporter/scorer once per completed fit without training."""
    sys.path.insert(0, HELPERS)
    from wnn.multistage_search import atomic_write_json
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import current_process_owner

    clock = json.loads((ROOT / "control/CLOCK_R1.json").read_bytes())
    assert time.time() < clock["wave_deadline_epoch"]
    control = ROOT / "control/label-endpoint-exports-r1"
    control.mkdir(exist_ok=False)
    (control / "owned").mkdir()
    owner = current_process_owner()
    atomic_write_json(control / "owner.json", owner, replace=False)

    def run_cell(gpu: int, cell: str) -> dict[str, object]:
        fit = ROOT / "runs" / cell / "fit"
        assert json.loads((fit / "terminal_status.json").read_bytes())["status"] == "completed"
        assert json.loads((fit / "epoch_index.json").read_bytes())["completed_epochs"] == 6
        checkpoint = fit / "model.pt"
        assert checkpoint.is_file()
        output = fit.parent / "export-e6-r1"
        assert not output.exists()
        env = dict(
            os.environ,
            PYTHONPATH=str(SOURCE / "src"),
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="8",
            NVIDIA_TF32_OVERRIDE="0",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=str(ROOT / "scratch"),
        )
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "CUBLAS_EMULATE_SINGLE_PRECISION",
            "CUBLAS_WORKSPACE_CONFIG",
        ):
            env.pop(key, None)
        commands = [
            [
                PYTHON,
                str(SOURCE / "scripts/predict_sequence.py"),
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(DATA),
                "--split-manifest",
                str(DATA / "chronological_primary.json"),
                "--membership",
                "evaluation",
                "--output",
                str(output),
            ],
            [
                PYTHON,
                str(SOURCE / "scripts/score_sequence.py"),
                "--data-root",
                str(DATA),
                "--split-manifest",
                str(DATA / "chronological_primary.json"),
                "--predictions",
                str(output),
                "--output",
                str(fit.parent / "score-e6-r1.json"),
            ],
        ]
        results = []
        for stage, command in zip(("export", "score"), commands, strict=True):
            remaining = int(clock["wave_deadline_epoch"] - time.time())
            assert remaining > 0
            result = run_owned_command(
                command,
                cwd=SOURCE,
                env=env,
                log_path=control / f"{cell}-{stage}.log",
                receipt_path=control / "owned" / f"{cell}-{stage}.json",
                owner={"controller": owner, "cell_id": cell},
                stage=1,
                trial_number=gpu,
                mode="fixed_endpoint_export" if stage == "export" else "official_score",
                gpu_key=str(gpu),
                output_dir=output,
                timeout_seconds=min(3600, remaining),
            )
            code, timed_out, _ = result
            results.append({"stage": stage, "exit_code": code, "timed_out": timed_out})
            if code or timed_out:
                break
        result = {"cell": cell, "gpu": gpu, "stages": results, "at_epoch": time.time()}
        score_path = fit.parent / "score-e6-r1.json"
        if len(results) == 2 and results[-1]["exit_code"] == 0 and not results[-1]["timed_out"]:
            score = json.loads(score_path.read_bytes())
            assert score["sequences"] == 1873 and score["score"]["selected_rows"] == 3528273
            assert score["score"]["metric_id"] == "global_wp_clip2_v1"
            result["official_score"] = score["score"]
            result["score_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        atomic_write_json(control / f"{cell}-result.json", result, replace=False)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_cell, gpu, cell) for gpu, cell in enumerate(CELLS)]
        results = [future.result() for future in futures]
    atomic_write_json(control / "complete.json", {"results": results}, replace=False)


if __name__ == "__main__":
    main()
