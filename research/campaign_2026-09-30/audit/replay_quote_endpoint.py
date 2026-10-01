"""Independently replay the causal a6 endpoint versus TRAIN-voted movement."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pyarrow.parquet as pq
from replay_pooled_code import AUDIT, DATA, ROOT, read_group


def main() -> None:
    """Use raw fit labels for votes and held-out current inputs for endpoint flags."""
    started = time.monotonic()
    panel = json.loads((DATA / "selection.json").read_text())
    actor = json.loads((DATA / "full112_relations/quote_endpoint_results.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    votes = defaultdict(Counter)
    fit_minimum = np.float32(np.inf)
    for group in panel["fit_train_groups"]:
        x, t, _ = read_group(source, group)
        a6 = (
            source.read_row_group(group, columns=["a6"], use_threads=False)["a6"]
            .to_numpy()
            .astype(np.float32)
        )
        assert np.isfinite(a6).all()
        fit_minimum = min(fit_minimum, a6.min())
        changes = np.nonzero(np.any(x[1:] != x[:-1], axis=1))[0] + 1
        for row in changes:
            if (t[row] <= 0).all() and (t[row - 1] <= 0).all():
                before = np.round(x[row - 1], 4).tobytes()
                after = np.round(x[row], 4).tobytes()
                delta = t[row] - t[row - 1]
                votes[before, after][(-int(delta[0]), int(delta[1]))] += 1
    moves = {}
    for pair, counts in votes.items():
        value, support = max(counts.items(), key=lambda item: item[1])
        if support >= 3 and support >= 0.8 * counts.total():
            moves[pair] = value
    assert len(votes) == actor["pooled_fit_voted_pairs"]
    assert len(moves) == actor["accepted_pairs"]
    declared = np.float32(NormalDist().inv_cdf(1e-7 - np.finfo(float).eps))
    assert fit_minimum == declared
    # These retrospective classes were checked against raw targets by the prior
    # independent pooled replay; they never enter the current-input marker.
    classes = {
        r["group"]: r["diagnostic_class"]
        for r in json.loads((DATA / "pooled_results.json").read_text())["by_group"]
    }
    totals = defaultdict(Counter)
    group_results = []
    for group in panel["test_train_groups"]:
        columns = [*[f"i0_p{i}" for i in range(22)], "a6"]
        table = source.read_row_group(group, columns=columns, use_threads=False)
        x = np.column_stack([table[f"i0_p{i}"].to_numpy() for i in range(22)]).astype(np.float32)
        a6 = table["a6"].to_numpy().astype(np.float32)
        marker = a6 == fit_minimum
        changed = np.concatenate(([False], np.any(x[1:] != x[:-1], axis=1)))
        changes = np.nonzero(changed)[0]
        count = Counter(
            rows=len(x),
            code_events=len(changes),
            endpoint_rows=int(marker.sum()),
            endpoint_quiet_rows_including_origin=int((marker & ~changed).sum()),
            endpoint_origin_rows=int(marker[0]),
            endpoint_quiet_rows_excluding_origin=int((marker[1:] & ~changed[1:]).sum()),
        )
        for row in changes:
            move = moves.get((np.round(x[row - 1], 4).tobytes(), np.round(x[row], 4).tobytes()))
            if move is None:
                count["unknown_move_events"] += 1
                count["endpoint_on_unknown_move_events"] += bool(marker[row])
            else:
                name = "nonzero" if any(move) else "zero"
                count[f"known_{name}_events"] += 1
                count[f"endpoint_on_known_{name}_events"] += bool(marker[row])
                count[f"not_endpoint_on_known_{name}_events"] += not bool(marker[row])
        for key in ("all", str(classes[group])):
            totals[key].update(count)
        group_results.append(
            {"group": group, "class_diagnostic_only": classes[group], "counts": dict(count)}
        )
    observed = {key: dict(value) for key, value in totals.items()}
    assert observed == actor["counts"] and group_results == actor["by_group"]
    all_counts = observed["all"]
    assert (
        all_counts["known_nonzero_events"]
        == all_counts["endpoint_on_known_nonzero_events"]
        == 67839
    )
    assert (
        all_counts["known_zero_events"] == 45649
        and all_counts["endpoint_on_known_zero_events"] == 0
    )
    assert all_counts["code_events"] == 130382 and all_counts["unknown_move_events"] == 16894
    assert all_counts["endpoint_quiet_rows_including_origin"] == 13
    assert (
        all_counts["endpoint_origin_rows"] == 10
        and all_counts["endpoint_quiet_rows_excluding_origin"] == 3
    )
    result = {
        "status": "pass",
        "raw_train_parquet_replay": True,
        "actor_modules_imported": False,
        "accepted_move_count": len(moves),
        "fit_vote_pairs": len(votes),
        "counts": observed,
        "by_group": group_results,
        "endpoint_float32": float(fit_minimum),
        "endpoint_fit_min_matches_independent_normal_inverse": True,
        "heldout_inference_columns": columns,
        "heldout_targets_masks_not_read": True,
        "class_used_for_grading_only": "retrospective pooled-class sidecar independently reproduced earlier",
        "interpretation": "Current a6 equality is causal; movement labels are TRAIN-supervised proxy coordinates, not independently identified physical quote changes. Unknown events remain unknown.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256((AUDIT / "replay_pooled_code.py").read_bytes()).hexdigest(),
        "actor_result_sha256": hashlib.sha256(
            (DATA / "full112_relations/quote_endpoint_results.json").read_bytes()
        ).hexdigest(),
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output = AUDIT / "quote_endpoint_independent_replay.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "counts": all_counts, "resources": result["resources"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
