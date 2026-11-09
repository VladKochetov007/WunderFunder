from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
PYTHON = PurePosixPath("/data/epanov_s/tabdata/.venv/bin/python")
SPLIT = PurePosixPath(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json"
)


def build(args: argparse.Namespace) -> None:
    """Write identified shape recipes and direct canonical commands; launch no process."""
    base_root = Path(__file__).resolve().parents[1] / "configs"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms.split(","):
        if arm not in ("labels", "complement"):
            raise ValueError("unknown objective arm")
        for world in (int(value) for value in args.worlds.split(",")):
            if world not in (1, 2):
                raise ValueError("world size must be one or two")
            for batch in (int(value) for value in args.batches.split(",")):
                if batch not in (4, 16, 32, 64, 128):
                    raise ValueError("batch must belong to the declared engineering matrix")
                base = base_root / f"{arm}_s{args.seed}.json"
                config = json.loads(base.read_text())
                config.update(batch_size=batch, required_world_size=world)
                name = f"{arm}_s{args.seed}_w{world}_b{batch}_a4"
                if args.peak:
                    config["warmup_fraction"] = 0.0
                    name += "_peak"
                assert config["accumulate_chunks"] == 4 and config["chunk_rows"] == 256
                assert config["model"]["context_rows"] == 1024
                path = args.output / f"{name}.json"
                raw = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
                if path.exists() and path.read_bytes() != raw:
                    raise FileExistsError("preserve an existing different profile recipe")
                path.write_bytes(raw)
                updates_per_epoch = math.ceil(math.ceil(10325 / world) / batch) * 20
                source = RUNTIME / "source-r3"
                command = [
                    str(args.python),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={world}",
                    str(source / "scripts/train_sequence.py"),
                    "--config",
                    str(args.remote_root / "configs" / path.name),
                    "--data-root",
                    str(RUNTIME / "project"),
                    "--split-manifest",
                    str(SPLIT),
                    "--monitor-reference-root",
                    str(RUNTIME / "reference"),
                    "--output",
                    str(args.remote_root / "profiles" / name),
                    "--profile-updates",
                    str(args.updates),
                ]
                if arm == "complement":
                    command += ["--reference-cache", str(RUNTIME / "reference/fit")]
                rows.append(
                    {
                        "id": name,
                        "config": path.name,
                        "config_sha256": hashlib.sha256(raw).hexdigest(),
                        "base_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                        "world": world,
                        "batch_per_rank": batch,
                        "accumulate_chunks": 4,
                        "global_sequences_per_window": world * batch,
                        "global_chunk_mass": world * batch * 4,
                        "full_window_rows": world * batch * 1024,
                        "updates_per_epoch": updates_per_epoch,
                        "updates_per_30_epochs": updates_per_epoch * 30,
                        "schedule_warmup_updates": math.ceil(
                            updates_per_epoch * 30 * config["warmup_fraction"]
                        ),
                        "profile_updates": args.updates,
                        "discard_first_updates": 20,
                        "measured_updates": args.updates - 20,
                        "scientific_fits": 0,
                        "command": command,
                    }
                )
    manifest = {
        "schema": "wnn.pr67-shape-engineering.v1",
        "status": "prepared_not_executed",
        "source_commit": "6ab2f679e70928f913b97d5a29f0d1e09ceacaa3",
        "new_scientific_fits": 0,
        "precision": "unchanged bf16_bf16x9",
        "environment": {
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
            "TMPDIR": "/data/epanov_s/tdn7run/throughput-r1/scratch",
            "TORCHINDUCTOR_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/inductor",
            "TRITON_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/triton",
        },
        "profiles": rows,
    }
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Materialize a bounded engineering matrix without modifying frozen recipes or source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "configs")
    parser.add_argument(
        "--remote-root", type=PurePosixPath, default=RUNTIME / "throughput-engineering-r1"
    )
    parser.add_argument("--python", type=PurePosixPath, default=PYTHON)
    parser.add_argument("--batches", default="4,16,32,64,128")
    parser.add_argument("--worlds", default="1,2")
    parser.add_argument("--arms", default="labels,complement")
    parser.add_argument("--seed", type=int, choices=(101001, 101002), default=101001)
    parser.add_argument("--updates", type=int, choices=range(21, 51), default=50)
    parser.add_argument("--peak", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
