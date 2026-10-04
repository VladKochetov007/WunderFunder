"""Run one bounded E427 full-shape CUDA smoke and write an audit sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[1]
TRAINER = HERE / "scripts/train_e427_smoke.py"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("aligned", "stale", "noop"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, child_args = parser.parse_known_args()
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    if not child_args or child_args[0] != "--name":
        raise ValueError("E427 smoke wrapper requires explicit child trainer arguments after --")
    child_text = list(child_args)
    pairs = {child_text[i]: child_text[i + 1] for i in range(len(child_text) - 1)
             if child_text[i].startswith("--") and not child_text[i + 1].startswith("--")}
    flags = set(child_text)
    if (
        pairs.get("--arch") != "e427-volume-gate"
        or pairs.get("--e427-arm") != args.arm
        or pairs.get("--profile-steps") != "2"
        or pairs.get("--profile-warmup") != "0"
        or pairs.get("--batch") != "32"
        or pairs.get("--chunk") != "1000"
        or pairs.get("--epochs") != "1"
        or "--no-eval-during-fit" not in flags
        or "--save-training-state" in flags
    ):
        raise ValueError("E427 smoke must be the matched full-shape two-batch profile without evaluation/checkpointing")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("smoke result must be written inside the repository")
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable, str(TRAINER), *child_args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    marker = None
    for line in completed.stdout.splitlines():
        if line.startswith("E427_SMOKE_RESULT "):
            if marker is not None:
                raise ValueError("trainer emitted duplicate E427 smoke result markers")
            marker = json.loads(line[len("E427_SMOKE_RESULT "):])

    stats_path = ROOT / "research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json"
    parent_ckpt = ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt"
    parent_provenance = ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/provenance.json"
    report = {
        "schema": "wnn.e427-production-smoke-record-v1",
        "arm": args.arm,
        "status": "pass" if completed.returncode == 0 and marker and marker.get("status") == "pass" else "fail",
        "exit_code": completed.returncode,
        "command": command,
        "training_result": marker,
        "parent_checkpoint_sha256": digest(parent_ckpt),
        "parent_provenance_sha256": digest(parent_provenance),
        "context_stats_sha256": digest(stats_path),
        "smoke_trainer_sha256": digest(TRAINER),
        "smoke_model_sha256": digest(HERE / "src/wnn/e427_volume_gate.py"),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "model_checkpoint_written": False,
        "scientific_fit": False,
        "no_holdout_evaluation": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"smoke_record": str(output.relative_to(ROOT)), "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
