#!/usr/bin/env python3
"""Input-only redundancy screen for per-field ages on the pinned F0 fixture."""

from __future__ import annotations

import hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
PROJECT = Path(os.environ["WNN_PROJECT_ROOT"]).resolve()
sys.path.insert(0, str(WORKTREE / "src"))
from wnn.features import StreamingEventFeatures, event_features  # noqa: E402

MANIFEST_SHA256 = "941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05"
GROUPS = ((0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112))
AGE_CAP = 10_000
SINCE_OFFSETS = (74, 75, 76)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cgroup_snapshot() -> dict[str, object]:
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    d = Path("/sys/fs/cgroup") / group.lstrip("/")
    names = ("memory.current", "memory.peak", "memory.high", "memory.max", "memory.swap.max", "cpu.max")
    return {"group": group, "memory.events": (d / "memory.events").read_text().splitlines(), **{n: (d / n).read_text().strip() for n in names}}


def main() -> None:
    started = time.perf_counter()
    manifest_path = HERE / "CPU_TIMING_FIXTURE_R1.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError(f"fixture manifest hash changed: {manifest_sha}")
    manifest = json.loads(manifest_bytes)
    spec = manifest["arrays"]["features"]
    feature_path = PROJECT / manifest["payload_root"] / spec["file"]
    expected_bytes = int(spec["bytes"])
    shape = tuple(map(int, spec["shape"]))
    if shape != (8, 20_000, 112) or expected_bytes != int(np.prod(shape)) * 4:
        raise ValueError("unexpected raw feature fixture schema")
    if feature_path.stat().st_size != expected_bytes:
        raise ValueError("raw feature fixture byte length differs from manifest")
    feature_sha = sha256_file(feature_path)
    if feature_sha != spec["sha256"]:
        raise ValueError(f"raw feature fixture hash changed: {feature_sha}")

    raw = np.memmap(feature_path, dtype=np.dtype(spec["dtype"]), mode="r", shape=shape)
    field_count = shape[2]
    slots, rows = shape[:2]
    field_ix = np.arange(field_count)
    denom = slots * (rows - 1)
    total_rows = slots * rows
    change_counts = np.zeros(field_count, dtype=np.int64)
    cap_counts = np.zeros(field_count, dtype=np.int64)
    position_age_counts = np.zeros(field_count, dtype=np.int64)
    maximum_age = np.zeros(field_count, dtype=np.int32)
    age_seen = np.zeros((field_count, AGE_CAP + 1), dtype=np.bool_)
    group_diff_counts = np.zeros(len(GROUPS), dtype=np.int64)
    group_cell_counts = np.zeros(len(GROUPS), dtype=np.int64)
    since_match_slots = np.zeros((3, field_count), dtype=np.int64)
    since_match_all = np.ones((3, field_count), dtype=np.bool_)
    since_reset_counts = np.zeros(3, dtype=np.int64)
    per_sequence = []

    for slot, provenance in enumerate(manifest["sequences"]):
        seq = raw[slot]
        stream = StreamingEventFeatures()
        age = np.zeros(field_count, dtype=np.int32)
        group_age = np.zeros(len(GROUPS), dtype=np.int32)
        field_resets = np.zeros((rows, field_count), dtype=np.bool_)
        since_resets = np.zeros((rows, 3), dtype=np.bool_)
        stream_rows = np.empty((rows, 99), dtype=np.float32)
        first_output = None
        first_age_zero = False
        for t in range(rows):
            row = seq[t]
            previous = row if stream.prev is None else stream.prev
            changed = np.not_equal(row, previous)
            if t == 0:
                age.fill(0)
                group_age.fill(0)
                field_resets[t] = True  # sequence-local clock initialization
                first_age_zero = bool(not age.any() and not group_age.any())
            else:
                change_counts += changed
                age[changed] = 0
                age[~changed] = np.minimum(age[~changed] + 1, AGE_CAP)
                field_resets[t] = changed
            for gi, (lo, hi) in enumerate(GROUPS):
                group_changed = bool(changed[lo:hi].any())
                if t > 0 and group_changed:
                    group_age[gi] = 0
                elif t > 0:
                    group_age[gi] = min(int(group_age[gi]) + 1, AGE_CAP)
                group_cell_counts[gi] += hi - lo
                group_diff_counts[gi] += int(np.count_nonzero(age[lo:hi] != group_age[gi]))
            age_seen[field_ix, age] = True
            cap_counts += age == AGE_CAP
            position_age_counts += age == min(t, AGE_CAP)
            maximum_age = np.maximum(maximum_age, age)

            out = stream.update(row)
            stream_rows[t] = out
            since_resets[t] = out[list(SINCE_OFFSETS)] == 0.0
            if t == 0:
                first_output = out.copy()

        batched = event_features(seq)
        if not np.array_equal(stream_rows, batched):
            raise AssertionError(f"existing event feature stream/batch parity failed at slot {slot}")
        stream.reset()
        reset_output = stream.update(seq[0])
        reset_parity = bool(np.array_equal(reset_output, first_output) and first_age_zero)
        if not reset_parity:
            raise AssertionError(f"existing callback reset parity failed at slot {slot}")

        same_clock = np.stack([
            np.all(field_resets == since_resets[:, k, None], axis=0) for k in range(3)
        ])
        since_match_slots += same_clock
        since_match_all &= same_clock
        since_reset_counts += since_resets.sum(0)
        field_patterns = {np.packbits(field_resets[:, j]).tobytes() for j in range(field_count)}
        group_patterns = {
            np.packbits(field_resets[:, lo:hi].any(1)).tobytes() for lo, hi in GROUPS
        }
        per_sequence.append({
            **{key: provenance[key] for key in ("slot", "source", "component_id", "sequence_id", "train_row_group")},
            "distinct_field_reset_patterns": len(field_patterns),
            "distinct_group_reset_patterns": len(group_patterns),
            "field_matches_each_existing_since_clock": same_clock.sum(1).tolist(),
            "stream_batch_event_features_equal": True,
            "sequence_reset_event_features_equal": reset_parity,
        })

    per_field = []
    for j in range(field_count):
        per_field.append({"index": j, "changes": int(change_counts[j]), "change_rate": float(change_counts[j] / denom),
                          "distinct_age_values": int(age_seen[j].sum()), "max_age": int(maximum_age[j]),
                          "rows_at_age_cap": int(cap_counts[j]), "rows_equal_sequence_position_age": int(position_age_counts[j]),
                          "reset_pattern_matches_by_existing_since_clock_slots": since_match_slots[:, j].tolist()})

    output = {
        "schema": "wnn.per-field-clock-redundancy-screen.r1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "input-only fitting fixture; not prediction, quality, CV, or held-out evidence",
        "fixture_manifest_sha256": manifest_sha,
        "feature_payload": {"path": str(feature_path), "bytes": expected_bytes, "sha256": feature_sha, "shape": list(shape)},
        "code": {
            "script_sha256": sha256_file(Path(__file__)),
            "features_py_sha256": sha256_file(WORKTREE / "src/wnn/features.py"),
            "nn_py_sha256": sha256_file(WORKTREE / "src/wnn/nn.py"),
            "numpy": np.__version__,
            "torch_imported": "torch" in sys.modules,
        },
        "execution": {"elapsed_seconds": time.perf_counter() - started, "cgroup": cgroup_snapshot()},
        "method": {
            "age_reset": "raw field inequality, row 0 reset, saturating count at 10000",
            "group_ranges": [list(value) for value in GROUPS],
            "group_reset": "any raw field in range changes",
            "existing_reset_source": "StreamingEventFeatures since outputs at offsets 74:77; actual _flags indices",
            "raw_change_denominator": denom,
            "total_field_rows": total_rows,
        },
        "summary": {
            "constant_fields": np.flatnonzero(change_counts == 0).tolist(),
            "always_changing_fields": np.flatnonzero(change_counts == denom).tolist(),
            "change_rate_min_median_max": [float(np.min(change_counts / denom)), float(np.median(change_counts / denom)), float(np.max(change_counts / denom))],
            "fields_matching_each_existing_since_clock_all_slots": since_match_all.sum(1).tolist(),
            "fields_matching_any_existing_since_clock_all_slots": int(since_match_all.any(0).sum()),
            "existing_since_clock_reset_rows": since_reset_counts.tolist(),
            "group_age_vs_individual_age_differing_cell_fraction": (group_diff_counts / np.maximum(group_cell_counts, 1)).tolist(),
            "field_age_rows_at_cap": int(cap_counts.sum()),
            "field_age_rows_equal_sequence_position_age": int(position_age_counts.sum()),
            "field_age_position_degeneracy_fraction": float(position_age_counts.sum() / (total_rows * field_count)),
            "batch_stream_reset_parity_all_slots": all(x["sequence_reset_event_features_equal"] for x in per_sequence),
        },
        "per_sequence": per_sequence,
        "per_field": per_field,
        "interpretation_limit": "Distinct deterministic clock patterns establish added representation resolution only; they do not establish predictive value or justify a model-quality claim.",
    }
    output_path = HERE / "PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(output_path), "fields": field_count, "components": len(per_sequence), "summary": output["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
