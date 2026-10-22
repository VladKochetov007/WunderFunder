"""Pin the existing F0 warm recipe and execute its bounded engineering profile."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path("/home/vlad/development/wunder/research_tmp/i1-warm-source-frozen-20261022")
PACKET = ROOT / "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1"


def sha256(path: Path) -> str:
    """Return a bounded-memory digest without loading arrays or checkpoints."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Fail closed on frozen identity/output collisions before running canonical training."""
    started = time.monotonic()
    recipe = json.loads((PACKET / "PROFILE_F0_WARM_R2.json").read_text())
    assert (
        subprocess.check_output(
            ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
        ).strip()
        == ""
    )
    source_head = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    assert (
        source_head
        == subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "66959b920"], text=True
        ).strip()
    ), source_head
    manifest = recipe["source_manifest"]
    assert sha256(ROOT / manifest["plan"]) == manifest["plan_sha256"]
    checked: dict[str, str] = {}
    for name, expected in manifest["canonical_source_sha256"].items():
        checked["source/" + name] = sha256(SOURCE / name)
        assert checked["source/" + name] == expected, name
    parent = recipe["actual_parent"]
    pins = {
        parent["checkpoint"]: parent["checkpoint_sha256"],
        "research/campaign_2026-09-29/artifacts/splits/fold0.json": parent["split_manifest_sha256"],
        parent["target_cache_root"] + "/metadata.json": parent["target_cache_metadata_sha256"],
        parent["target_cache_root"] + "/main_homogeneous_train.npy": parent[
            "target_main_train_sha256"
        ],
        parent["target_cache_root"] + "/main_homogeneous_valid.npy": parent[
            "target_main_valid_sha256"
        ],
        "artifacts/runs/" + parent["propensity_run"] + "/model.pt": parent[
            "propensity_checkpoint_sha256"
        ],
    }
    plan = json.loads((ROOT / manifest["plan"]).read_text())
    cell = plan["cells"][0]
    pins["artifacts/runs/" + parent["run"] + "/provenance.json"] = cell["parent"][
        "provenance_sha256"
    ]
    pins["artifacts/runs/" + parent["propensity_run"] + "/provenance.json"] = cell["propensity"][
        "provenance_sha256"
    ]
    for name, expected in pins.items():
        checked[name] = sha256(ROOT / name)
        assert checked[name] == expected, name
    argv = list(recipe["argv"])
    argv[0] = str(SOURCE / "scripts/train_rnn.py")
    assert "--save-training-state" not in argv and "--resume-state" not in argv
    assert argv[argv.index("--profile-steps") + 1] == "100"
    assert argv[argv.index("--profile-warmup") + 1] == "10"
    name = argv[argv.index("--name") + 1]
    assert not os.path.lexists(ROOT / "artifacts/runs" / name), name
    receipt = PACKET / "PROFILE_F0_LOCAL_EXECUTION_R2.json"
    assert not receipt.exists()
    environment = os.environ.copy()
    environment.update(
        {
            "WNN_PROJECT_ROOT": str(ROOT),
            "PYTHONPATH": str(SOURCE / "src") + os.pathsep + str(SOURCE),
        }
    )
    evidence = {
        "source_head": source_head,
        "source_root": str(SOURCE),
        "pins": checked,
        "argv": argv,
        "status": "pins_passed_profile_starting",
        "preflight_seconds": time.monotonic() - started,
    }
    receipt.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "pins_verified": len(checked),
                "preflight_seconds": evidence["preflight_seconds"],
            }
        ),
        flush=True,
    )
    result = subprocess.run([sys.executable, "-u", *argv], cwd=ROOT, env=environment, check=False)
    evidence.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        wall_seconds=time.monotonic() - started,
    )
    receipt.write_text(json.dumps(evidence, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
