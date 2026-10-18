from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
STATE = ROOT / "state/density4-v3"


def record(text: str, label: str) -> dict:
    """Require exactly one canonical profile evidence record."""
    rows = [json.loads(line[len(label) + 1:]) for line in text.splitlines() if line.startswith(label + " ")]
    if len(rows) != 1:
        raise ValueError(f"Expected one {label}, got {len(rows)}")
    return rows[0]


def progress_rate(points: list[dict], start: float, end: float) -> dict:
    """Compute observed useful work, rejecting sparse or stale progress."""
    rows = [row for row in points if start <= row["time_unix"] <= end]
    if len(rows) < 3:
        raise ValueError(f"Insufficient time-aligned progress points: {len(rows)}")
    if rows[0]["time_unix"] - start > 90 or end - rows[-1]["time_unix"] > 90:
        raise ValueError("Original training did not span the comparison window")
    span = rows[-1]["time_unix"] - rows[0]["time_unix"]
    batches = rows[-1]["batch_index"] - rows[0]["batch_index"]
    if span < 75 or batches <= 0:
        raise ValueError("Progress window is short or non-monotonic")
    return dict(sequences_per_second=batches * 8 / span, observed_seconds=span,
                completed_batches=batches, completed_optimizer_updates=batches * 10,
                first_unix=rows[0]["time_unix"], last_unix=rows[-1]["time_unix"])


def main() -> None:
    """Evaluate four-process throughput with existing R01 contention included."""
    baseline = json.loads((STATE / "BASELINE.json").read_text())
    samples = json.loads((STATE / "SAMPLES.json").read_text())
    observer = json.loads((STATE / "OBSERVER.json").read_text())["rows"]
    results = {}
    for gpu in ("2", "7"):
        timing = observer[gpu]
        start, end = timing["training_seen"], timing["completed_seen"]
        span = end - start
        if not 120 < span < 600:
            raise ValueError(f"Invalid observed profile interval: {span}")
        text = (STATE / "logs" / f"gpu{gpu}.log").read_text()
        updates = record(text, "PROFILE_UPDATE_EVIDENCE")
        resource = record(text, "PROFILE_RESOURCE_EVIDENCE")
        profile = record(text, "PROFILE")
        envelope = record(text, "PROCESS_ENVELOPE")
        if updates["optimizer_completed_updates"] != 660 or not updates["valid_for_throughput"]:
            raise ValueError("Profile did not complete 660 finite optimizer updates")
        if resource["precision"] != "fp32" or resource["matmul_tf32"] or resource["cudnn_tf32"] or resource["float32_matmul_precision"] != "highest":
            raise ValueError("Strict FP32 contract failed")
        during = {name: progress_rate(samples[name], start, end) for name in baseline[gpu]}
        fourth = dict(sequences_per_second=66 * 8 / span, completed_batches=66,
                      completed_optimizer_updates=660, observed_seconds=span)
        before = sum(row["sequences_per_second"] for row in baseline[gpu].values())
        after = sum(row["sequences_per_second"] for row in during.values()) + fourth["sequences_per_second"]
        r01_name = next(name for name in during if name.startswith("tabdata_b300_r01"))
        r01_change = during[r01_name]["sequences_per_second"] / baseline[gpu][r01_name]["sequences_per_second"] - 1
        before_progress = sum(row["sequences_per_second"] / (6 if name == r01_name else 3) for name, row in baseline[gpu].items())
        after_progress = sum(row["sequences_per_second"] / (6 if name == r01_name else 3) for name, row in during.items()) + fourth["sequences_per_second"] / 3
        results[gpu] = dict(baseline_three=baseline[gpu], existing_during_four=during,
                            added_profile=fourth, before_sequences_per_second=before,
                            after_sequences_per_second=after, aggregate_gain_fraction=after / before - 1,
                            epoch_budget_normalized_gain_fraction=after_progress / before_progress - 1,
                            r01_rate_change_fraction=r01_change, profile=profile, updates=updates,
                            resource=resource, envelope=envelope, window_start_unix=start, window_end_unix=end)
        if not all(math.isfinite(value) for value in (before, after, after_progress, before_progress)):
            raise ValueError("Nonfinite throughput evidence")
    receipt = dict(schema="tabdata-density4-walltime.v1", results=results,
                   interpretation="Engineering throughput only; existing actor rates from progress-log intervals. Added actor rate is all 528 sequences over observed loader-to-complete walltime, including warmup/overhead. Two GPUs and one short window do not establish sustained 8-GPU throughput.",
                   scientific_recipe_changed=False, original_controllers_untouched=True,
                   source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "runtime/density4-v3/bench.py", ROOT / "runtime/density4-v3/observe.py", Path(__file__))})
    output = STATE / "WALLTIME_RESULT.json"
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
