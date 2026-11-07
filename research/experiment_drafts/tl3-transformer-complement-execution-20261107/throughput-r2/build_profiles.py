from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
REMOTE = RUNTIME / "throughput-engineering-r2"
CASES = (
    ("labels", 2, 256, "0,1", False),
    ("complement", 2, 128, "2,3", False),
    ("labels", 1, 256, "4", False),
    ("complement", 1, 256, "5", False),
    ("labels", 2, 128, "6,7", True),
)


def main() -> None:
    """Materialize only ROOT's five new engineering profiles, without changing R1 or S3."""
    packet = Path(__file__).resolve().parents[1]
    output = Path(__file__).resolve().parent / "configs"
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    for arm, world, batch, devices, peak in CASES:
        base = packet / "configs" / f"{arm}_s101001.json"
        config = json.loads(base.read_text())
        config.update(batch_size=batch, required_world_size=world)
        name = f"{arm}_s101001_w{world}_b{batch}_a4" + ("_peak" if peak else "")
        if peak:
            config["warmup_fraction"] = 0.0
        raw = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
        path = output / f"{name}.json"
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError("preserve existing different R2 profile config")
        path.write_bytes(raw)
        command = [
            "/data/epanov_s/tabdata/.venv/bin/python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={world}",
            str(RUNTIME / "source-r3/scripts/train_sequence.py"),
            "--config",
            str(REMOTE / "configs" / path.name),
            "--data-root",
            str(RUNTIME / "project"),
            "--split-manifest",
            "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json",
            "--monitor-reference-root",
            str(RUNTIME / "reference"),
            "--output",
            str(REMOTE / "profiles" / name),
            "--profile-updates",
            "50",
        ]
        if arm == "complement":
            command += ["--reference-cache", str(RUNTIME / "reference/fit")]
        global_sequences = world * batch
        updates = math.ceil(math.ceil(10325 / world) / batch) * 20
        profiles.append(
            {
                "id": name,
                "config": path.name,
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "base_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "cuda_visible_devices": devices,
                "world": world,
                "batch_per_rank": batch,
                "accumulate_chunks": 4,
                "global_chunk_mass": global_sequences * 4,
                "updates_per_epoch": updates,
                "updates_per_30_epochs": updates * 30,
                "schedule_warmup_updates": math.ceil(updates * 30 * config["warmup_fraction"]),
                "profile_updates": 50,
                "discard_first_updates": 20,
                "measured_updates": 30,
                "full_window_rows": global_sequences * 1024,
                "short_window_rows": global_sequences * 544,
                "warm_first20_real_rows": global_sequences * 20000,
                "measured_last30_real_rows": global_sequences * 30240,
                "all50_real_rows": global_sequences * 50240,
                "all50_selected_rows": global_sequences * 49943,
                "max_wall_seconds": 900,
                "max_reserved_fraction": 0.9,
                "scientific_fits": 0,
                "command": command,
            }
        )
    environment = {
        "PYTHONPATH": str(RUNTIME / "source-r3/src"),
        "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "WNN_MONITOR_HEARTBEAT_SECONDS": "5",
        "TMPDIR": "/data/epanov_s/tdn7run/throughput-r2/scratch",
        "TORCHINDUCTOR_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/inductor",
        "TRITON_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/triton",
    }
    manifest = {
        "schema": "wnn.pr67-shape-engineering-r2.v1",
        "status": "prepared_not_executed",
        "source_commit": "6ab2f679e70928f913b97d5a29f0d1e09ceacaa3",
        "new_scientific_fits": 0,
        "profiles": profiles,
        "environment": environment,
        "allocation": "five ROOT-admitted 50-update profiles,900 seconds each; after R1 completes",
    }
    for profile in profiles:
        assert all("\\" not in value for value in profile["command"])
        assert (
            hashlib.sha256((output / profile["config"]).read_bytes()).hexdigest()
            == profile["config_sha256"]
        )
    assert all("\\" not in value for value in environment.values())
    (output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
