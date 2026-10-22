"""Recover NLL fits in new immutable attempts without renewing the node budget."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
BASE = ROOT / "runtime/next4h-20261022-r2"
TEMPLATE = ROOT / "runtime/next4h-20261022-wide"
COMMON = ROOT / "state/next4h-20261022-nllfixed"
START = 1792693855.4927282
BASE_SHA = "d76774e58d8d8c06c52bac701bde97ea2aa704f2536ac2f4815775e6a5e1eefd"


def sha(path: Path) -> str:
    """Hash exact source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(value: str, before: str, after: str) -> str:
    """Reject template drift instead of silently changing a different recipe."""
    if value.count(before) != 1:
        raise ValueError(f"Expected exactly one template fragment: {before!r}")
    return value.replace(before, after)


def project() -> None:
    """Mount original verified inputs beside new output paths."""
    COMMON.mkdir()
    original = ROOT / "state/next4h-20261022-r2/START.json"
    receipt = json.loads(original.read_text())
    assert receipt["start_unix"] == START
    (COMMON / "START.json").write_text(json.dumps(receipt, indent=2))
    destination = COMMON / "nll_project"
    destination.mkdir()
    for name, target in (
        ("src", BASE / "sources/nll/src"),
        ("scripts", BASE / "sources/nll/scripts"),
        ("datasets", ROOT / "data/datasets"),
        ("splits", ROOT / "splits"),
    ):
        (destination / name).symlink_to(target, target_is_directory=True)
    (destination / "cache").mkdir()
    for name in (
        "train_targets.npz",
        "valid_targets.npz",
        "chains.npz",
        "chain_folds.npz",
    ):
        (destination / "cache" / name).symlink_to(ROOT / "data/cache" / name)
    (destination / "artifacts/runs").mkdir(parents=True)


def stage(hidden: int) -> Path:
    """Freeze one sixteen-fit controller and its fit-only state-saving operator."""
    packet = ROOT / f"runtime/next4h-20261022-nllfixed-h{hidden}"
    packet.mkdir()
    for name in ("wide.py", "wide_publish.py", "prepare_wide.py"):
        value = (TEMPLATE / name).read_text()
        value = value.replace(
            "next4h-20261022-wide", f"next4h-20261022-nllfixed-h{hidden}"
        )
        value = value.replace(
            "tabdata-wide-20261022:PR37", f"tabdata-nllfixed-h{hidden}-20261022:PR37"
        )
        value = value.replace(
            'BASE_STATE = ROOT / "state/next4h-20261022-r2"',
            'BASE_STATE = ROOT / "state/next4h-20261022-nllfixed"',
        )
        value = value.replace("PROFILE_CUTOFF = 3600", "PROFILE_CUTOFF = 5400")
        value = value.replace("profiles +3600s", "profiles +5400s")
        value = value.replace("wide_f{", f"fixedh{hidden}_f{{")
        value = value.replace("H512", f"H{hidden}")
        value = value.replace("!= 512", f"!= {hidden}")
        value = value.replace('= "512"', f'= "{hidden}"')
        value = value.replace('"hidden": 512', f'"hidden": {hidden}')
        if name == "wide.py":
            value = replace_once(
                value,
                "sys.path.insert(0, str(BASE_PACKET))",
                "sys.path.insert(0, str(BASE_PACKET))\nsys.path.insert(0, str(PACKET))",
            )
            value = replace_once(
                value,
                "    if used_bytes + reserved >= 0.90 * total_bytes or reserved > free_bytes:",
                "    # Both width controllers admit at most 32 GiB per pair; reserve their sum.\n"
                "    aggregate_reserve = 64 * 1024**3\n"
                "    if reserved > 32 * 1024**3 or used_bytes + aggregate_reserve >= 0.90 * total_bytes or aggregate_reserve > free_bytes:",
            )
            if hidden == 256:
                value = value.replace(
                    "128 + slot * 16 + arm_index * 8", "32 + slot * 8 + arm_index * 4"
                )
                value = value.replace(
                    "range(cpu_start, cpu_start + 8)", "range(cpu_start, cpu_start + 4)"
                )
        elif name == "prepare_wide.py":
            value = replace_once(
                value,
                '("wide.py", "wide_publish.py", "prepare_wide.py")',
                '("wide.py", "wide_publish.py", "prepare_wide.py", "delivery.py")',
            )
            if hidden == 256:
                value = value.replace(
                    "128 + 8 * (pair * 2 + index)", "32 + 4 * (pair * 2 + index)"
                )
                value = value.replace("range(cpu, cpu + 8)", "range(cpu, cpu + 4)")
            value = value.replace(
                '"initial_density": "GPUs0-1:4 GRU workers each; GPUs2-7:1 Transformer +4 GRU workers each",',
                '"initial_density": "Recovery: 4 GRU workers/card across both widths; 6 numeric fits continue separately",',
            )
        else:
            begin = value.index(
                '    lines += [\n        "",\n        "#### Original cohort progress context"'
            )
            end = value.index("    lines += [", begin + 15)
            value = value[:begin] + value[end:]
            value = value.replace(
                '    base_plan = read(BASE_PACKET / "PLAN.json")\n', ""
            )
            value = value.replace(
                "This is a prospective ",
                "Recovery after SSH logout stopped the original attempt (Linger was disabled; now verified enabled). "
                "Original partial artifacts are preserved. These are fresh retries, not additional independent replications. "
                "Six numeric fits continue in a separate recovery service. This is a prospective ",
            )
        compile(value, str(packet / name), "exec")
        (packet / name).write_text(value)
    delivery = (BASE / "delivery.py").read_text()
    delivery = replace_once(
        delivery,
        'STATE = ROOT / "state/next4h-20261022-r2"',
        'STATE = ROOT / "state/next4h-20261022-nllfixed"',
    )
    delivery = replace_once(
        delivery,
        '    if stage == "profile":\n        argv += ["--profile-warmup", "2", "--profile-steps", "4"]',
        '    if stage == "fit":\n        argv += ["--save-training-state"]\n'
        '    if stage == "profile":\n        argv += ["--profile-warmup", "2", "--profile-steps", "4"]',
    )
    compile(delivery, str(packet / "delivery.py"), "exec")
    (packet / "delivery.py").write_text(delivery)
    state = ROOT / f"state/next4h-20261022-nllfixed-h{hidden}"
    state.mkdir()
    return packet


def main() -> None:
    """Verify ownership persistence and frozen sources before bounded recovery."""
    assert sha(BASE / "PLAN.json") == BASE_SHA
    assert time.time() < START + 5000
    assert (
        subprocess.check_output(
            ["loginctl", "show-user", str(os.getuid()), "-pLinger", "--value"],
            text=True,
        ).strip()
        == "yes"
    )
    original_plan = json.loads((TEMPLATE / "SIDEPLAN.json").read_text())
    for name, digest in original_plan["files"].items():
        assert sha(Path(name)) == digest, name
    project()
    packets = [stage(hidden) for hidden in (256, 512)]
    for packet in packets:
        code = (
            "import sys; sys.path.insert(0,sys.argv[1]); import delivery; "
            'j={"id":"preflight","fold":0,"seed":1201,"arm":"nll","family":"nll"}; '
            'p=delivery.nll_command(j,"profile"); f=delivery.nll_command(j,"fit"); '
            'e=delivery.nll_command(j,"evaluate"); '
            'assert "--save-training-state" not in p; '
            'assert f.count("--save-training-state")==1; '
            'assert "--save-training-state" not in e; print("FIT_ONLY_STATE_PREFLIGHT_OK")'
        )
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-c", code, str(packet)],
            check=True,
            timeout=20,
        )
        shutil.copy2(Path(__file__), packet / "recovery_setup.py")
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(packet / "prepare_wide.py")],
            check=True,
            timeout=45,
        )
    print(
        json.dumps(
            {
                "recovery_packets": list(map(str, packets)),
                "deadline_unix": START + 14400,
            }
        )
    )


if __name__ == "__main__":
    main()
