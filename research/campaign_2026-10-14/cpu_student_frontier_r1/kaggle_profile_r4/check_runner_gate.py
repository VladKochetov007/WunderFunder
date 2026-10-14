"""Exercise the sealed runner's real file gate, stopping before any worker launch."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


class ReachedLaunch(RuntimeError):
    """Raised instead of creating a child process after real identity checks."""


def deny_launch(*args: object, **kwargs: object) -> None:
    """Make GPU/CPU worker execution impossible in this offline gate check."""
    stdout = kwargs.get("stdout")
    if hasattr(stdout, "close"):
        stdout.close()
    raise ReachedLaunch("all input gates passed; worker launch intentionally intercepted")


def main() -> None:
    """Use actual pinned producer files with real ordinary-file/symlink semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root / "source"
    packet = Path(__file__).resolve().parent
    before = packet.parent / "kaggle_profile_r3/specs"
    current = packet / "specs"
    names = sorted(p.name for p in current.glob("*profile*.spec.json"))
    assert len(names) == 3
    materialized = {
        "cache/clean_t3_f0_h390_t3r5_a9_r2/" + name
        for name in ("metadata.json", "main_homogeneous_train.npy", "main_homogeneous_valid.npy")
    }
    all_inputs = {}
    for name in names:
        old = json.loads((before / name).read_text())
        new = json.loads((current / name).read_text())
        old_deferred = old["environment"].pop("WNN_DEFERRED_IDENTITY_PATHS").split(",")
        new_deferred = new["environment"].pop("WNN_DEFERRED_IDENTITY_PATHS").split(",")
        assert old == new, "unexpected scientific/code/spec drift"
        assert set(old_deferred) - set(new_deferred) == materialized
        assert [x for x in old_deferred if x not in materialized] == new_deferred
        all_inputs.update({x["path"]: x for x in new["inputs"]})
    assert (before / "preflight.spec.json").read_bytes() == (current / "preflight.spec.json").read_bytes()
    for name in all_inputs:
        dest = root / name
        if dest.exists():
            continue
        source = args.repo_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name in materialized:
            os.link(source, dest)
        else:
            dest.symlink_to(source.resolve())
    assert all((root / name).is_file() and not (root / name).is_symlink() for name in materialized)
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("sealed_runner", root / "scripts/kaggle_gpu_runner.py")
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Inventory is synthetic solely to reach the actual file gate. Popen always raises.
    host = {"affinity": [0, 1, 2, 3], "capacity_bytes": 32_212_254_720,
            "available_bytes": 29_000_000_000, "cgroup_headroom_known": True}
    devices = [{"index": i, "name": "Tesla T4", "uuid": f"OFFLINE-{i}",
                "total_bytes": 16_106_127_360, "used_bytes": 0} for i in range(2)]
    results = {}
    with patch.object(runner, "host_inventory", return_value=host), \
            patch.object(runner, "gpu_inventory", return_value=devices), \
            patch.object(runner.subprocess, "Popen", side_effect=deny_launch):
        try:
            runner.run(root, [before / name for name in names], args.stage_root / "old_gate_r2")
        except ValueError as error:
            assert "Deferred mounted input size mismatch" in str(error), str(error)
            results["r3_reproduced"] = str(error)
        else:
            raise AssertionError("R3 regular-file rejection was not reproduced")
        try:
            runner.run(root, [current / name for name in names], args.stage_root / "new_gate_r2")
        except ReachedLaunch as error:
            results["r4_gate_pass"] = str(error)
        else:
            raise AssertionError("R4 did not reach the intercepted launch")
    receipts = []
    for name in names:
        identity = args.stage_root / "new_gate_r2" / (json.loads((current / name).read_text())["name"] + ".input_identity.json")
        receipt = json.loads(identity.read_text())
        selected = [x for x in receipt["observations"] if x["path"] in materialized]
        assert len(selected) == 3
        for item in selected:
            assert item["observed"]["sha256"] == item["expected"]["sha256"]
        receipts.append({"path": str(identity), "materialized_file_observations": selected})
    results.update({"status": "pass", "scope": "actual producer bytes; exact sealed runner gate only; inventory mocked and every process launch intercepted", "workers_launched": 0, "cache_payload_bytes_per_candidate": 1_910_723_660, "scientific_specs_unchanged": True, "preflight_spec_byte_identical": True, "identity_receipts": receipts})
    with args.output.open("x") as output:
        json.dump(results, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"status": "pass", "workers_launched": 0, "candidates_checked": 3}))


if __name__ == "__main__":
    main()
