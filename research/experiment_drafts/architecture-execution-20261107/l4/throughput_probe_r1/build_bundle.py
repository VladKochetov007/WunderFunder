"""Create immutable native-CLI L4 pulse profiles from the actual mirrored recipes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
PYTHON = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"
FAMILIES = ("gru", "gru_local", "mamba2", "ema_tabm_style")


def digest(value: object) -> str:
    """Hash normalized JSON with the canonical source runtime encoding."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def build(artifacts: Path, source: Path, output: Path) -> dict[str, object]:
    """Generate only configurations and command metadata; execute no model or profile."""
    defaults = {}
    for node in ast.parse((source / "src/wnn/training/sequence_config.py").read_text()).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in ("DEFAULTS", "MODEL_DEFAULTS")
        ):
            defaults[node.target.id] = ast.literal_eval(node.value)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for gpu, family in zip((0, 2, 4, 6), FAMILIES, strict=True):
        metadata_path = (
            artifacts / "live-dashboard/runs" / f"l4-{family}_s101001-fit/run_metadata.json"
        )
        metadata = json.loads(metadata_path.read_bytes())
        base = (
            source
            / "research/experiment_drafts/architecture-execution-20261107/l4/configs"
            / f"{family}_s101001.json"
        )
        baseline = json.loads(base.read_bytes())
        baseline["batch_size"] = 16
        baseline["monitoring"].update(
            diagnostics_every_steps=20, training_telemetry_every_steps=10, eval_batch_size=32
        )
        normalized = defaults["DEFAULTS"] | baseline
        normalized["model"] = (
            defaults["MODEL_DEFAULTS"] | baseline["model"] | {"precision_mode": "fp32"}
        )
        if normalized != metadata["config"]:
            raise ValueError(
                f"derived runtime-r3 baseline differs from actual {family} training metadata"
            )
        if (
            metadata["source"]["sha256"]
            != "3ea26eb2cc9beefd7d07e9d2216c937cc6f78f6ee99719272261ce47e15f2f2a"
        ):
            raise ValueError("actual training source is not frozen source-r1")
        for batch in (16, 32, 64):
            config = baseline | {"batch_size": batch}
            name = f"{family}_s101001_b{batch}.json"
            payload = (json.dumps(config, indent=2) + "\n").encode()
            with (output / name).open("xb") as stream:
                stream.write(payload)
            remote_config = str(RUNTIME / "configs/throughput-probe-r1" / name)
            remote_output = str(RUNTIME / "runs/throughput-probe-r1" / f"{family}_b{batch}")
            argv = [
                PYTHON,
                str(RUNTIME / "source-r1/scripts/train_sequence.py"),
                "--config",
                remote_config,
                "--data-root",
                str(RUNTIME / "project"),
                "--split-manifest",
                str(RUNTIME / "project/chronological_primary.json"),
                "--output",
                remote_output,
                "--profile-updates",
                "20",
            ]
            new_normalized = normalized | {"batch_size": batch}
            entries.append(
                {
                    "family": family,
                    "seed": 101001,
                    "gpu_index": gpu,
                    "batch_size": batch,
                    "config_file": name,
                    "config_sha256": hashlib.sha256(payload).hexdigest(),
                    "normalized_config_sha256": digest(new_normalized),
                    "actual_baseline_metadata_sha256": hashlib.sha256(
                        metadata_path.read_bytes()
                    ).hexdigest(),
                    "source_sha256": metadata["source"]["sha256"],
                    "argv": argv,
                    "env_set": {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONPATH": str(RUNTIME / "source-r1/src"),
                        "OMP_NUM_THREADS": "8",
                        "MKL_NUM_THREADS": "8",
                        "OPENBLAS_NUM_THREADS": "8",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "NVIDIA_TF32_OVERRIDE": "0",
                        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                    },
                    "env_unset": [
                        "CUBLAS_WORKSPACE_CONFIG",
                        "CUBLAS_EMULATE_SINGLE_PRECISION",
                        "RANK",
                        "WORLD_SIZE",
                        "LOCAL_RANK",
                    ],
                    "output_dir": remote_output,
                    "cwd": str(RUNTIME / "source-r1"),
                    "timeout_seconds": 200,
                    "nominal_rows_per_update": batch * 2000,
                    "future_twelve_epoch_updates": math.ceil(10325 / batch) * 10 * 12,
                }
            )
    plan = {
        "schema": "wnn.l4-throughput-pulse.v1",
        "status": "prepared ROOT-owned engineering only",
        "scientific_fits": 0,
        "profile_jobs_max": 12,
        "profile_updates_each": 20,
        "profile_timeout_seconds": 200,
        "family_cumulative_seconds_max": 420,
        "SIGCONT_watchdog_seconds": 480,
        "prior_active_fits_changed": False,
        "source": str(RUNTIME / "source-r1"),
        "precision": "unchanged strictFP32/IEEE environment; no TF32/AMP change",
        "warmup_excluded": False,
        "timing_note": "canonical20-update receipt includes cold per-process shape/kernel startup; compare native ordinary step10 against diagnostic step20 and full owned wall time; do not label this twenty post-warmup updates",
        "B128": "not prepared; requires measured retained allocation plus measured profile reserve below90% device capacity and remaining pulse time",
        "profile_admission": "ROOT must bind current physical UUID, prove paused trainer PID/birth, SM idle, memory reserve, 420s cumulative budget, and independent480s SIGCONT watchdog; stop this owned profile before resuming original trainer",
        "entries": entries,
    }
    with (output / "PULSE_PLAN_R1.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    return plan


def main() -> None:
    """Build a portable POSIX-path bundle without SSH, CUDA, Torch import or fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build(args.artifacts, args.source_root, args.output)
    print(
        json.dumps(
            {
                "profiles": len(plan["entries"]),
                "plan_sha256": hashlib.sha256(
                    (args.output / "PULSE_PLAN_R1.json").read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
