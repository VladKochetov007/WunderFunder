from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def main() -> int:
    """Run six one-shot50-update profiles through ROOT's accepted owned-session relay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PACKET / "profile-plan.json")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--accepted-receiver", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(PACKET / "receiving"))
    from admission import digest, read
    from lifecycle import accepted_helpers

    sys.path.insert(0, str(args.accepted_receiver))
    from controller import Controller, load_helpers

    accepted_helpers(args.accepted_receiver, Controller)
    helpers = load_helpers(args.support_src)
    control = args.runtime / "control/root-profiles-r2"
    if args.detach:
        command = [
            str(args.python),
            str(Path(__file__).resolve()),
            *sum(
                (
                    [flag, str(value)]
                    for flag, value in (
                        ("--plan", args.plan),
                        ("--runtime", args.runtime),
                        ("--python", args.python),
                        ("--accepted-receiver", args.accepted_receiver),
                        ("--support-src", args.support_src),
                    )
                ),
                [],
            ),
        ]
        print(
            json.dumps(
                helpers.detach(
                    command,
                    cwd=args.runtime,
                    env=dict(os.environ),
                    receipt_path=control / "controller-owner.json",
                    metadata={"profile_cap": 6, "new_scientific_fits": 0},
                ),
                sort_keys=True,
            )
        )
        return 0

    with helpers.lock(control / "profiles.lock", blocking=False):
        plan = read(args.plan)
        if (
            plan.get("schema") != "wnn.transformer-optimizer-horizon-panel.v1"
            or plan.get("source_commit") != "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
            or plan.get("split_sha256") != SPLIT_SHA
            or plan.get("profile_cap") != 6
            or plan.get("new_scientific_fit_cap") != 6
        ):
            raise ValueError("profile plan differs from the authorized six-cell panel")
        source = Path(plan["source_root"])
        if source.resolve().parent != args.runtime.resolve():
            raise ValueError("runtime must own the exact source-r2 capsule")
        manifest_path = Path(plan["source_manifest"])
        if digest(manifest_path) != plan["source_manifest_sha256"]:
            raise ValueError("source-r2 manifest identity differs")
        manifest = read(manifest_path)
        if digest(Path(plan["split_manifest"])) != SPLIT_SHA:
            raise ValueError("official chronological split identity differs")
        scratch = Path(plan["environment"]["TMPDIR"])
        if not scratch.is_symlink() or scratch.resolve() != (args.runtime / "scratch").resolve():
            raise ValueError("new tdh8 alias must resolve to this runtime's scratch")
        os.environ.update(plan["environment"])
        identity_files = {
            key: value
            for key, value in manifest["files"].items()
            if (key.startswith("src/") and key.endswith(".py"))
            or key == "scripts/train_sequence.py"
        }
        source_sha = hashlib.sha256(
            json.dumps(
                identity_files, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        cells = {item["cell"]: item for item in plan["cells"]}
        if len(cells) != 6 or {name for wave in plan["profile_wave_order"] for name in wave} != set(
            cells
        ):
            raise ValueError("profile waves must cover six distinct cells")
        if any(item["profile_command"][0] != str(args.python) for item in cells.values()):
            raise ValueError("profile Python differs from the approved environment")
        control.mkdir(parents=True, exist_ok=True)
        for name in ("owned", "descendants", "logs", "terminal"):
            (control / name).mkdir(exist_ok=True)
        helpers.write(
            control / "attempts.json",
            {
                "source_commit": plan["source_commit"],
                "new_scientific_fits": 0,
                "profile_attempts": [
                    {"cell": item["cell"], "pair": item["pair"]} for item in plan["cells"]
                ],
            },
            replace=False,
        )
        owner = helpers.owner()
        pair_locks = {item["pair"]: threading.Lock() for item in cells.values()}

        def run_profile(item: dict) -> dict:
            name, pair = item["cell"], item["pair"]
            config = args.runtime / "configs" / item["config"]
            if digest(config) != item["config_sha256"]:
                raise ValueError(f"frozen config changed: {name}")
            output = args.runtime / "profiles-r2" / name
            receipt_path = output / "profile_receipt.json"
            attempt = control / "owned" / f"{name}.json"
            if output.exists() or attempt.exists():
                raise FileExistsError(f"no profile retry allowed: {name}")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=pair,
                PYTHONDONTWRITEBYTECODE="1",
                WNN_CONTROLLER_DESCENDANTS_RECEIPT=str(control / "descendants" / f"{name}.json"),
            )
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
                env.pop(key, None)
            command = [
                str(args.python),
                str(args.accepted_receiver / "owned_phase.py"),
                "--support-src",
                str(args.support_src),
                "--",
                *item["profile_command"],
            ]
            with pair_locks[pair]:
                code, timed_out, _ = helpers.run(
                    command,
                    cwd=source,
                    env=env,
                    log_path=control / "logs" / f"{name}.log",
                    receipt_path=attempt,
                    owner={"controller": owner, "cell_id": name},
                    stage=0,
                    trial_number=int(pair.split(",")[0]),
                    mode="engineering_profile",
                    gpu_key=pair,
                    output_dir=output,
                    timeout_seconds=900,
                )
            helpers.write(
                control / "terminal" / f"{name}.json",
                {
                    "exit_code": code,
                    "timed_out": timed_out,
                    "new_scientific_fits": 0,
                },
                replace=False,
            )
            if code != 0 or timed_out:
                raise RuntimeError(f"owned profile failed; no retry: {name}")
            receipt = read(receipt_path)
            identity, reload = receipt.get("identity", {}), receipt.get("engineering_reload", {})
            ranks = reload.get("by_rank", [])
            if (
                receipt.get("status") != "profile_complete"
                or receipt.get("updates") != 50
                or receipt.get("errors") != []
                or receipt.get("scientific_checkpoint_written") is not False
                or identity.get("source_sha256") != source_sha
                or identity.get("world_size") != 2
                or identity.get("data", {}).get("split_sha256") != SPLIT_SHA
                or reload.get("status") != "passed"
                or [r.get("rank") for r in ranks] != [0, 1]
                or any(
                    r.get("status") != "passed"
                    or r.get("model_optimizer_scheduler_rng_equal") is not True
                    for r in ranks
                )
            ):
                raise ValueError(f"profile source/config/reload check failed: {name}")
            return {
                "cell": name,
                "status": receipt["status"],
                "updates": 50,
                "config_sha256": identity.get("config_sha256"),
                "elapsed_seconds": receipt.get("elapsed_seconds"),
                "peak_reserved_bytes": receipt.get("peak_reserved_bytes"),
            }

        for index, wave in enumerate(plan["profile_wave_order"], start=1):
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                results = list(pool.map(run_profile, (cells[name] for name in wave)))
            helpers.write(control / f"wave-{index}.json", {"profiles": results}, replace=False)
        helpers.write(control / "COMPLETE.json", {"status": "six_profiles_complete"}, replace=False)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
