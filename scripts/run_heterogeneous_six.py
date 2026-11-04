from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

if __package__:
    from scripts.heterogeneous_six_support import (
        Runtime,
        gpu_snapshot,
        pin,
        source_binding,
        ticks,
        verify_fit,
        write_json,
    )
else:
    from heterogeneous_six_support import (
        Runtime,
        gpu_snapshot,
        pin,
        source_binding,
        ticks,
        verify_fit,
        write_json,
    )

if __package__:
    from scripts.heterogeneous_six_evidence import (
        advance_cache,
        export_args,
        external_status,
        profile_gate,
    )
else:
    from heterogeneous_six_evidence import advance_cache, export_args, external_status, profile_gate

from scripts.heterogeneous_tl3_recipes import build_nodes
from wnn.heterogeneous_pool_sources import read_object
from wnn.splits import file_sha256, load_campaign_split


def run(args: argparse.Namespace) -> None:
    """Complete only the already-authorized two-teacher/four-student dependency graph."""
    root, source = args.root.resolve(strict=True), args.source.resolve(strict=True)
    status_path = root / "status" / f"controller-six-{args.attempt}.json"
    if status_path.exists():
        raise FileExistsError("preserve the first controller attempt; do not duplicate admission")
    split = load_campaign_split(args.split, root)
    binding, old_binding = source_binding(source), source_binding(root / "source-r1")
    python = root.parents[1] / ".venv/bin/python"
    started = time.monotonic()
    runtime = Runtime(root, source, python, binding, started + args.max_wall_seconds)
    runtime.helper_sha256 = file_sha256(root / "tools-r1/heterogeneous_owned_command.py")
    nodes = build_nodes(args.split, root)
    state = {
        "schema": "wnn.heterogeneous-six-controller.v1",
        "attempt": args.attempt,
        "status": "running",
        "controller_pid": os.getpid(),
        "controller_start_ticks": ticks(os.getpid()),
        "source_identity": binding,
        "controller_script": pin(Path(__file__)),
        "verified_fits": {},
        "exports": {},
        "caches": {},
        "profiles": {},
        "adopted_jobs": {},
        "errors": {},
        "jobs": runtime.jobs,
    }
    if args.attempt == "r2":
        predecessor = root / "status/controller-six-r1.json"
        prior = read_object(predecessor)
        if ticks(prior["controller_pid"]) == prior["controller_start_ticks"]:
            raise RuntimeError("the original controller is still alive; refuse duplicate admission")
        state["predecessor"] = {
            "status": pin(predecessor),
            "stop_receipt": pin(root / "controller-r1-stop.json"),
        }
    state["controller_helpers"] = [
        pin(Path(__file__).with_name(name))
        for name in ("heterogeneous_six_support.py", "heterogeneous_six_evidence.py")
    ]
    write_json(
        root / "status" / f"controller-six-{args.attempt}.claim.json",
        {"pid": os.getpid(), "start_ticks": ticks(os.getpid())},
    )
    write_json(status_path, state, replace=True)

    def fit_phase(
        key: str, node: dict, tokens: list[str], gpu: int, phase: str, seconds: int
    ) -> str:
        request = root / "requests" / f"{key}.json"
        if key not in runtime.jobs and request.exists():
            if not (root / "status" / f"{key}.json").exists():
                return "launching"
            adopted = external_status(root, key, binding)
            if adopted["request"]["command"][-len(tokens) :] != tokens:
                raise ValueError(f"adopted {phase} recipe differs: {key}")
            state["adopted_jobs"][key] = adopted["status"]
            return adopted["status"]["status"]
        return runtime.job(
            key, "train_rnn.py", tokens, gpu=gpu, phase=phase, role=node["id"], seconds=seconds
        )

    def export(node: dict, name: str, gpu: int) -> None:
        key = f"export-{node['id']}-{name}-r1"
        if key in state["exports"]:
            return
        result = runtime.job(
            key,
            "predict_run.py",
            export_args(node["run"], name, args.split),
            gpu=gpu,
            phase=f"{name}_export",
            role=node["id"],
            seconds=1800,
        )
        if result == "complete":
            path = root / "cache" / f"pred_{node['run']}_{name}.npy"
            side = read_object(path.with_suffix(".json"))
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            count = 10607 if name == "train" else 1873
            if (
                values.shape != (count, 20000, 2)
                or side.get("prediction_sha256") != file_sha256(path)
                or side.get("checkpoint_sha256")
                != state["verified_fits"][node["id"]]["artifacts"]["model.pt"]["sha256"]
                or side.get("epoch") != 3
            ):
                raise ValueError(f"export identity or full coverage differs: {key}")
            state["exports"][key] = {
                "prediction": pin(path),
                "sidecar": pin(path.with_suffix(".json")),
            }

    def student(node: dict, gpu: int, profile: str) -> None:
        if profile not in state["caches"]:
            return
        key = f"profile-{node['id']}-r1"
        tokens = [token for token in node["args"] if token != "--save-training-state"]
        tokens[tokens.index("--name") + 1] += "_profile_r1"
        tokens += ["--profile-steps", "3", "--profile-warmup", "1"]
        if fit_phase(key, node, tokens, gpu, "profile", 900) != "complete":
            return
        if key not in state["profiles"]:
            state["profiles"][key] = profile_gate(root / "logs" / f"{key}.log")
        if fit_phase(f"fit-{node['id']}-r1", node, node["args"], gpu, "fit", 21600) != "complete":
            return
        if node["id"] not in state["verified_fits"]:
            state["verified_fits"][node["id"]] = verify_fit(root, source, node, split)
        export(node, "valid", gpu)

    while time.monotonic() < runtime.deadline + 2000:
        actions = [
            (
                "cache:gru-only",
                lambda: advance_cache(runtime, state, split, args, nodes, "gru-only"),
            )
        ]
        for index, node in enumerate(nodes[:2]):

            def teacher(node: dict = node, gpu: int = index) -> None:
                observed = external_status(root, f"fit-{node['id']}-r1", old_binding)
                if observed["request"]["command"][-len(node["args"]) :] != node["args"]:
                    raise ValueError("adopted teacher recipe differs")
                if observed["status"]["status"] != "complete":
                    return
                if node["id"] not in state["verified_fits"]:
                    state["verified_fits"][node["id"]] = verify_fit(
                        root, root / "source-r1", node, split
                    )
                export(node, "train", gpu)
                if f"export-{node['id']}-train-r1" in state["exports"]:
                    export(node, "valid", gpu)

            actions.append((f"teacher:{node['id']}", teacher))
        actions.append(
            (
                "cache:heterogeneous",
                lambda: advance_cache(runtime, state, split, args, nodes, "heterogeneous"),
            )
        )
        for node in nodes[2:]:
            control = node["id"].startswith("control_")
            gpu = (5 if control else 0) + (node["id"].endswith("1302"))
            actions.append(
                (
                    node["id"],
                    lambda node=node, gpu=gpu, control=control: student(
                        node, gpu, "gru-only" if control else "heterogeneous"
                    ),
                )
            )
        for phase, action in actions:
            if phase not in state["errors"]:
                try:
                    action()
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    state["errors"][phase] = f"{type(error).__name__}: {error}"
        state.update(
            elapsed_seconds=time.monotonic() - started,
            scientific_fits_admitted=2
            + sum(job["phase"] == "fit" for job in runtime.jobs.values())
            + sum(key.startswith("fit-") for key in state["adopted_jobs"]),
            gpu_observation=gpu_snapshot(),
        )
        write_json(status_path, state, replace=True)
        if len(state["verified_fits"]) == 6 and all(
            f"export-{node['id']}-valid-r1" in state["exports"] for node in nodes
        ):
            state["status"] = "complete"
            break
        if time.monotonic() >= runtime.deadline and not any(
            job.get("observed", {}).get("status") in {"running", "launching"}
            for job in runtime.jobs.values()
        ):
            state["status"] = "partial_deadline"
            break
        time.sleep(15)
    if state["status"] == "running":
        state["status"] = "partial_deadline"
    write_json(status_path, state, replace=True)


def main() -> None:
    """Run one detached bounded continuation; never duplicate the original teachers."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "source", "split", "legacy-root", "galam-export"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--control-cache-key", default="cache-gru-only-r2")
    parser.add_argument("--attempt", choices=("r1", "r2"), default="r1")
    parser.add_argument("--max-wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 21600:
        parser.error("controller allocation must be positive and at most21600seconds")
    run(args)


if __name__ == "__main__":
    main()
