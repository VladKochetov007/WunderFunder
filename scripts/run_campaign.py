from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn import campaign, campaign_local, campaign_spec  # noqa: E402


def main() -> int:
    """Operate only the new explicitly initialized, content-bound campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state", type=Path, default=Path("research/campaign_2026-10-04/runner"))
    commands = parser.add_subparsers(dest="action", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("input", type=Path)
    seal.add_argument("output", type=Path)
    init = commands.add_parser("init")
    init.add_argument("--contract", type=Path, required=True)
    init.add_argument("--block", required=True)
    block = commands.add_parser("block")
    block.add_argument("id")
    block.add_argument("--review", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("spec", type=Path)
    launch.add_argument("--attempt", required=True)
    launch.add_argument("--python", default=sys.executable)
    worker = commands.add_parser("worker")
    worker.add_argument("attempt")
    worker.add_argument("--python", required=True)
    commands.add_parser("status")
    commands.add_parser("reconcile")
    pause = commands.add_parser("pause")
    pause.add_argument("--reason", required=True)
    acknowledge = commands.add_parser("acknowledge")
    acknowledge.add_argument("incident")
    acknowledge.add_argument("--review", required=True)
    job_ack = commands.add_parser("acknowledge-job")
    job_ack.add_argument("job")
    job_ack.add_argument("incident")
    job_ack.add_argument("--review", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = args.state.resolve() if args.state.is_absolute() else (root / args.state).resolve()
    if args.action == "seal":
        value = campaign_spec.seal_spec(root, campaign.read_json(args.input))
        if args.output.exists():
            raise FileExistsError("spec already exists; choose a new identity")
        campaign.atomic_json(args.output, value)
    elif args.action == "init":
        campaign_local.check_parent(campaign_local.properties(campaign_spec.SLICE))
        campaign.initialize(state, root, args.contract, args.block)
    elif args.action == "block":
        campaign_local.check_orphans(state)
        campaign.open_block(state, root, args.id, args.review)
    elif args.action == "launch":
        spec = campaign_spec.validate_spec(campaign.read_json(args.spec))
        record = campaign_local.launch(
            root, state, spec, args.attempt, str(Path(args.python).absolute())
        )
        print(json.dumps({"id": record["id"], "unit": record["unit"], "status": record["status"]}))
    elif args.action == "worker":
        return campaign_local.run_worker(root, state, args.attempt, args.python)
    elif args.action == "reconcile":
        campaign_local.reconcile(state)
    elif args.action == "pause":
        campaign.pause(state, args.reason)
    elif args.action == "acknowledge":
        campaign_local.check_parent(campaign_local.properties(campaign_spec.SLICE))
        campaign_local.check_orphans(state)
        campaign.acknowledge(state, root, args.incident, args.review)
    elif args.action == "acknowledge-job":
        campaign_local.check_parent(campaign_local.properties(campaign_spec.SLICE))
        campaign_local.check_orphans(state)
        campaign.acknowledge_job(state, root, args.job, args.incident, args.review)
    else:
        ready = campaign.read_json(state / "READY.json")
        print(
            json.dumps(
                {
                    "ready": ready,
                    "paused": (state / "PAUSE.json").exists(),
                    "quarantines": campaign.quarantines(state),
                    "budget": campaign.budget(state, str(ready["block_id"])),
                    "attempts": campaign.attempts(state),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
