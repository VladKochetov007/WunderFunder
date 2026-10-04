"""Verify completed E426 fit artifacts and fixed epoch-3 EMA state.

This is a provenance gate only. It reads the three completed fit outputs and
does not predict or score any validation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
ARMS = (
    ("e426-f0-constant-s3-4ep-v2", "e426_f0_constant_s3_4ep_v2", "constant"),
    ("e426-f0-aligned-s3-4ep-v2", "e426_f0_aligned_s3_4ep_v2", "aligned"),
    ("e426-f0-shuffled-s3-4ep-v2", "e426_f0_shuffled_s3_4ep_v2", "shuffled"),
)
EXPECTED_EPOCHS = 4
EXPECTED_BATCHES_PER_EPOCH = 373
EXPECTED_UPDATES_PER_EPOCH = 7_460
EXPECTED_OUTPUTS = 23


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require_bound_artifacts(name: str, record: dict, spec: dict) -> dict[str, str]:
    run_dir = ROOT / "artifacts/runs" / name
    actual = {item["path"]: item["sha256"] for item in record.get("artifacts", [])}
    expected = {
        f"artifacts/runs/{name}/model_ep3.pt",
        f"artifacts/runs/{name}/provenance.json",
        f"artifacts/runs/{name}/history.json",
        f"artifacts/runs/{name}/training_state.pt",
    }
    spec_paths = {item["path"] if isinstance(item, dict) else item
                  for item in spec.get("artifacts", [])}
    if spec_paths != expected or set(actual) != expected:
        raise ValueError(f"fit artifact set differs from fixed E426 contract: {name}")
    digests = {}
    for relative in sorted(expected):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != actual[relative]:
            raise ValueError(f"artifact missing or hash differs from attempt record: {relative}")
        digests[relative] = actual[relative]
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return digests


def require_counters(name: str, clock: str, history: list, provenance: dict) -> None:
    if len(history) != EXPECTED_EPOCHS:
        raise ValueError(f"expected four fixed epochs for {name}")
    totals = {"optimizer_attempts": 0, "amp_skipped_updates": 0,
              "optimizer_completed_updates": 0, "ema_updates": 0,
              "completed_batches": 0}
    for epoch, row in enumerate(history):
        if row.get("epoch") != epoch or row.get("evaluation") != "withheld":
            raise ValueError(f"epoch/evaluation policy changed for {name}, epoch {epoch}")
        updates = row.get("update_diagnostics", {}).get("epoch", {})
        expected = {
            "optimizer_attempts": EXPECTED_UPDATES_PER_EPOCH,
            "ema_updates": EXPECTED_UPDATES_PER_EPOCH,
            "completed_batches": EXPECTED_BATCHES_PER_EPOCH,
        }
        for field, value in expected.items():
            if updates.get(field) != value:
                raise ValueError(f"{name} epoch {epoch} {field}={updates.get(field)}")
        skipped = updates.get("amp_skipped_updates")
        completed = updates.get("optimizer_completed_updates")
        if (type(skipped) is not int or skipped < 0 or type(completed) is not int
                or completed + skipped != EXPECTED_UPDATES_PER_EPOCH):
            raise ValueError(f"optimizer skip/completion counters are inconsistent: {name}")
        for field in totals:
            totals[field] += updates[field]
        cumulative = row.get("update_diagnostics", {}).get("cumulative", {})
        if cumulative != totals:
            raise ValueError(f"cumulative counters mismatch history for {name}, epoch {epoch}")

    if provenance.get("cumulative_updates") != totals:
        raise ValueError(f"provenance/history final counters differ for {name}")
    clock_state = provenance.get("teacher_ema_clock", {})
    clock_epochs = clock_state.get("epochs", [])
    if clock_state.get("mode") != clock or len(clock_epochs) != EXPECTED_EPOCHS:
        raise ValueError(f"teacher clock identity differs from {clock}: {name}")
    for epoch, row in enumerate(clock_epochs):
        digest = row.get("applied_count_sha256", "")
        if (row.get("epoch") != epoch or row.get("updates_per_epoch") != EXPECTED_UPDATES_PER_EPOCH
                or row.get("total_train_events_epoch", 0) <= 0 or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"teacher clock event schedule is incomplete: {name}, epoch {epoch}")


def state_equal(left: dict, right: dict, torch) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError(f"preflight output must be new and inside repository: {output}")

    # Torch is intentionally isolated to the bounded CPU preflight worker.
    import torch

    rows = []
    for attempt_name, name, clock in ARMS:
        spec_path = CAMPAIGN / f"specs/{name}.sealed.json"
        record_path = RUNNER / f"attempts/{attempt_name}.json"
        spec, record = read_json(spec_path), read_json(record_path)
        if (record.get("id") != attempt_name or record.get("status") != "finished"
                or record.get("exit_code") != 0 or record.get("teardown_verified") is not True
                or record.get("spec") != spec):
            raise ValueError(f"authoritative fit attempt is not terminal-successful: {attempt_name}")
        artifacts = require_bound_artifacts(name, record, spec)
        history = read_json(ROOT / f"artifacts/runs/{name}/history.json")
        provenance = read_json(ROOT / f"artifacts/runs/{name}/provenance.json")
        require_counters(name, clock, history, provenance)
        ck = torch.load(ROOT / f"artifacts/runs/{name}/model_ep3.pt",
                        map_location="cpu", weights_only=False)
        training = torch.load(ROOT / f"artifacts/runs/{name}/training_state.pt",
                              map_location="cpu", weights_only=False)
        expected_state = training.get("evaluation_ema")
        checkpoint_state = ck.get("model")
        ck_args = ck.get("args", {})
        provenance_args = provenance.get("args", {})
        if (ck.get("epoch") != 3 or ck.get("n_in") != 211
                or ck_args.get("arch") != "gru" or ck_args.get("eval_ema") is not True
                or ck_args.get("no_eval_during_fit") is not True
                or ck_args.get("checkpoint_policy") != "fixed"
                or ck.get("args", {}).get("checkpoint_epoch") != 3
                or ck_args.get("epochs") != EXPECTED_EPOCHS
                or ck_args.get("teacher_ema_clock") != clock
                or provenance_args.get("name") != name
                or provenance_args.get("teacher_ema_clock") != clock
                or not isinstance(checkpoint_state, dict)
                or checkpoint_state.get("head.2.weight", torch.empty(0)).shape[0] != EXPECTED_OUTPUTS
                or not isinstance(expected_state, dict)
                or expected_state.get("kind") != "fast"
                or not isinstance(expected_state.get("state"), dict)
                or not state_equal(checkpoint_state, expected_state["state"], torch)):
            raise ValueError(f"fixed epoch-3 checkpoint is not the saved evaluation EMA: {name}")
        rows.append({"attempt": attempt_name, "run": name, "clock": clock,
                     "spec_sha256": record["spec_sha256"],
                     "artifacts": artifacts,
                     "epoch_count": len(history), "fixed_epoch": 3,
                     "n_in": ck["n_in"], "n_out": EXPECTED_OUTPUTS,
                     "evaluation_ema_kind": expected_state["kind"],
                     "evaluation_ema_tensor_equal": True,
                     "cumulative_updates": provenance["cumulative_updates"]})

    report = {
        "schema": "e426-fit-provenance-preflight-v1",
        "status": "passed",
        "scope": "artifact, counter, and evaluation-EMA checkpoint provenance only; no predictions or scores",
        "fits": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": output.relative_to(ROOT).as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
