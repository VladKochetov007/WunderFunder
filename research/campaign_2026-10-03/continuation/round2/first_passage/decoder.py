from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PRICE = [f"i0_p{i}" for i in range(22)]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
ENDPOINT = np.float32(-5.199337482452393)


def digest(path: Path) -> str:
    """Hash bounded source/provenance files, never complete parquet files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys_for(x: np.ndarray) -> list[bytes]:
    """Use inspected historical rounded float32 code grammar, without hash collisions."""
    return [row.tobytes() for row in np.round(x, 4).astype(np.float32)]


def read_group(
    source: pq.ParquetFile, group: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stream one TRAIN group including diagnostic labels and actual ID."""
    t = source.read_row_group(
        group, columns=PRICE + ["a6", "t0", "t1", "seq_ix"], use_threads=False
    )
    x = np.column_stack([t[c].to_numpy() for c in PRICE]).astype(np.float32)
    y = np.column_stack([t[c].to_numpy() for c in ["t0", "t1"]]).astype(float)
    a6 = t["a6"].to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, a6, y, int(t["seq_ix"][0].as_py())


def decode(x: np.ndarray, spreads: dict, moves: dict) -> dict:
    """Replay frozen tables using only the input prefix; never anchor from later rows."""
    n = len(x)
    rows = np.r_[0, np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1]
    event_keys = keys_for(x[rows])
    ask, bid = np.zeros(n), np.zeros(n)
    segment = np.full(n, -1, dtype=int)
    classes = np.full(n, -1, dtype=int)
    spread_rows = np.full(n, -1, dtype=int)
    reasons = Counter()
    current_a, current_b, anchored = 0, 0, False
    current_segment, anchor_class = 0, -1
    for i, row in enumerate(rows):
        row = int(row)
        end = int(rows[i + 1]) if i + 1 < len(rows) else n
        lookup = spreads.get(event_keys[i])
        c, sp = lookup if lookup is not None else (-1, -1)
        classes[row:end] = c
        spread_rows[row:end] = sp
        if i:
            move = moves.get((event_keys[i - 1], event_keys[i]))
            if move is None:
                reasons["unknown_transition"] += 1
                current_segment += 1
                anchored = False
            elif anchored:
                current_a += move[0]
                current_b += move[1]
                if sp >= 1 and (current_a - current_b != sp or c != anchor_class):
                    reasons["inconsistent_spread_or_class"] += 1
                    current_segment += 1
                    anchored = False
            else:
                reasons["known_move_without_anchor"] += 1
        if not anchored and sp >= 1:
            current_a, current_b = sp, 0
            anchored, anchor_class = True, c
            current_segment += 1
            reasons["prefix_anchor"] += 1
        ask[row:end], bid[row:end] = current_a, current_b
        if anchored:
            segment[row:end] = current_segment
        else:
            reasons["unanchored_rows"] += end - row
    return {
        "ask": ask,
        "bid": bid,
        "segment": segment,
        "class": classes,
        "spread": spread_rows,
        "decoder_counts": dict(reasons),
        "code_events": len(rows),
    }


def load_tables() -> tuple[dict, dict]:
    """Read this pass's frozen fit-only supervised coordinate."""
    model = json.loads((HERE / "decoder_tables.json").read_text())
    spreads = {bytes.fromhex(row[0]): tuple(row[1:]) for row in model["spreads"]}
    moves = {
        (bytes.fromhex(row[0]), bytes.fromhex(row[1])): tuple(row[2:]) for row in model["moves"]
    }
    return spreads, moves


def fit() -> None:
    """Freeze the existing component panel and fit without reading test targets."""
    started = time.monotonic()
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "fitting_decoder_on_frozen_train_components"
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    prereg = json.loads((HERE / "PREREG.json").read_text())
    original = ROOT / prereg["source_panel"]
    panel = json.loads(original.read_text())
    fit_groups, test_groups = panel["fit_train_groups"], panel["test_train_groups"][:12]
    fit_components, test_components = panel["fit_components"], panel["test_components"][:12]
    assert not set(fit_components) & set(test_components)
    selection = {
        "fit_groups": fit_groups,
        "test_groups": test_groups,
        "fit_components": fit_components,
        "test_components": test_components,
        "source_panel_sha256": digest(original),
        "prereg_sha256": digest(HERE / "PREREG.json"),
        "selection_without_new_targets": True,
        "previously_exposed": True,
    }
    (HERE / "selection.json").write_text(json.dumps(selection, indent=2))
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    spread_votes, move_votes = defaultdict(Counter), defaultdict(Counter)
    alias_values: dict[bytes, set[bytes]] = defaultdict(set)
    samples = []
    for group, component in zip(fit_groups, fit_components, strict=True):
        x, _, y, seq_id = read_group(source, group)
        errors = np.max(
            np.abs(np.rint(y[None] / QUANTA[:, None, None]) * QUANTA[:, None, None] - y),
            axis=(1, 2),
        )
        cls = int(errors.argmin())
        if errors[cls] > 0.002:
            raise ValueError(f"Fit-only lattice failure group{group}")
        ticks = np.rint(y / QUANTA[cls]).astype(int)
        interior = (ticks <= 0).all(axis=1)
        sp = -ticks.sum(axis=1)
        rows = np.r_[0, np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1]
        code_keys = keys_for(x[rows])
        for i, (code, row) in enumerate(zip(code_keys, rows, strict=True)):
            end = int(rows[i + 1]) if i + 1 < len(rows) else len(x)
            alias_values[code].add(x[row].tobytes())
            sample_rows = np.arange(row, end, 7)
            sample_rows = sample_rows[interior[sample_rows] & (sp[sample_rows] >= 1)]
            spread_votes[code].update((cls, int(v)) for v in sp[sample_rows])
            if i and interior[row - 1] and interior[row]:
                dy = ticks[row] - ticks[row - 1]
                move_votes[code_keys[i - 1], code][(-int(dy[0]), int(dy[1]))] += 1
        samples.append(
            {
                "group": group,
                "component": component,
                "seq_ix": seq_id,
                "fit_class": cls,
                "lattice_error": float(errors[cls]),
                "selected_array_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
            }
        )
        print(f"fit group={group} component={component} id={seq_id} class={cls}", flush=True)
    spreads, moves = {}, {}
    for code, counter in spread_votes.items():
        if counter:
            value, support = counter.most_common(1)[0]
            if support >= 3 and support / counter.total() >= 0.95:
                spreads[code] = value
    for pair, counter in move_votes.items():
        value, support = counter.most_common(1)[0]
        if support >= 3 and support / counter.total() >= 0.80:
            moves[pair] = value
    model = {
        "spreads": [[c.hex(), int(v[0]), int(v[1])] for c, v in spreads.items()],
        "moves": [[a.hex(), b.hex(), int(v[0]), int(v[1])] for (a, b), v in moves.items()],
        "training_class_counts": dict(Counter(r["fit_class"] for r in samples)),
        "rounded_alias_codes": sum(len(v) > 1 for v in alias_values.values()),
        "source_hashes": {
            "decoder.py": digest(Path(__file__)),
            "path_laws.py": digest(HERE / "path_laws.py"),
        },
        "selection_sha256": digest(HERE / "selection.json"),
    }
    (HERE / "decoder_tables.json").write_text(json.dumps(model))
    (HERE / "fit_results.json").write_text(
        json.dumps(
            {
                "samples": samples,
                "accepted_spreads": len(spreads),
                "accepted_moves": len(moves),
                "rounded_alias_codes": model["rounded_alias_codes"],
                "table_sha256": digest(HERE / "decoder_tables.json"),
                "cost": {
                    "wall_seconds": time.monotonic() - started,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                },
            },
            indent=2,
        )
    )
    state["phase"] = "decoder_frozen_before_test_target_reads"
    state["decoder_sha256"] = digest(HERE / "decoder_tables.json")
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    fit()
