from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from l4_wave_evidence import CELLS, INVENTORY, PANEL, REFERENCE, SOURCE, SPLIT, observe_fit
from l4_wave_readouts import observe_readouts

from scripts.sequence_progress_report import _finite, canonical_sha


def snapshot(root: Path, readout_root: Path | None, approved: dict, now: float) -> dict:
    """Observe the fixed six-cell wave; inconsistent evidence never counts as completed."""
    rows = []
    for cell in CELLS:
        try:
            row = observe_fit(root, cell, now)
            row["grades"] = observe_readouts(readout_root, root, row, approved)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            row = {
                "cell": cell,
                "phase": "artifact evidence unavailable or inconsistent",
                "fit_complete": False,
                "grades": {},
            }
        rows.append(row)
    terminal = [
        {
            "cell": row["cell"],
            "fit": row["fit_complete"],
            "status": row.get("terminal_observation"),
            "sha": row.get("checkpoint_sha256"),
            "grades": {k: v["receipt_sha256"] for k, v in row["grades"].items()},
            "exit": "exit" in row["phase"],
        }
        for row in rows
    ]
    return {
        "host": "l4-generalization-20261108",
        "observed_at_utc": datetime.fromtimestamp(now, UTC).isoformat(),
        "panel": {
            "run_prefix": "l4-generalization-20261108",
            "manifest_sha256": PANEL,
            "config_inventory_sha256": INVENTORY,
        },
        "cells": rows,
        "terminal_signature": canonical_sha(terminal),
    }


def _score(values: dict, keys: tuple[str, str, str]) -> str:
    if not all(_finite(values.get(key)) for key in keys):
        return "pending"
    return f"{values[keys[0]]:.6f} ({values[keys[1]]:.6f}, {values[keys[2]]:.6f})"


def render(value: dict) -> str:
    """Publish factual cached progress and every prescribed endpoint without selection."""
    rows = value["cells"]
    completed = sum(row["fit_complete"] for row in rows)
    grades = 2 * sum(len(row["grades"]) for row in rows)
    lines = [
        f"L4 generalization wave: {completed}/6 fits complete; {grades}/24 fixed endpoint grades recorded.",
        f"Generated {value['observed_at_utc']}; observations below are cached source data.",
        "Six fresh fits, fixed6E/38,760 updates. Primary equal SWA E3/4/5/6; secondary E6. Each endpoint has standalone and fixed0.9TL3+0.1candidate grades.",
        "",
        "| Cell | Last observed phase | Successful updates | Epochs | Full-valid looks | Latest full-valid WP (t0, t1) | Source age |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        age = row.get("source_age_seconds")
        age_text = (
            "unknown"
            if age is None
            else (f"{age:.0f}s" if age >= 0 else f"clock ahead {-age:.0f}s")
        )
        epoch = f"{row['epoch']:.3f}" if _finite(row.get("epoch")) else "unknown"
        score = _score(row.get("latest_valid", {}), ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
        lines.append(
            f"| {row['cell']} | {row['phase']} | {row.get('updates', 'unknown')} | {epoch}/6 | {row.get('valid_count', 0)}/6 | {score} | {age_text} |"
        )
    if grades:
        lines.extend(
            [
                "",
                "| Cell | Fixed endpoint | Standalone WP (t0, t1) | Fixed mixture WP (t0, t1) |",
                "|---|---|---|---|",
            ]
        )
        for row in rows:
            for endpoint, item in row["grades"].items():
                scores = item["metrics"]
                lines.append(
                    f"| {row['cell']} | {endpoint} | {_score(scores['candidate'], ('wp', 't0', 't1'))} | {_score(scores['mixture'], ('wp', 't0', 't1'))} |"
                )
    lines.extend(
        [
            "",
            f"Training source `{SOURCE}`; split `{SPLIT}`.",
            f"Frozen panel `{PANEL}`; qualified reused reference manifest `{REFERENCE}`.",
            "Normalized config identities and observed diagnostics:",
        ]
    )
    for row in rows:
        if row.get("ids"):
            counters = (
                ", ".join(f"{k}={v:g}" for k, v in sorted(row["nonfinite"].items()))
                or "not separately recorded"
            )
            lines.append(
                f"- {row['cell']}: config `{row['ids']['config']}`; clipping {row['clipping']}; nonfinite {counters}; last source UTC {row.get('source_observed_utc') or 'unknown'}."
            )
        if row.get("checkpoint_sha256"):
            lines.append(f"  Producer checkpoint SHA256 `{row['checkpoint_sha256']}`.")
    lines.extend(
        [
            "",
            "Producer completion/grade receipts record hashes; artifact upload and independent local checkpoint/prediction byte verification remain pending.",
            "The24 fixed grades and36 per-epoch full-valid observations are development-exposed. Reference qualification inherits its recorded ancestry limitations. Scientific interpretation remains with ROOT; no public-score forecast.",
        ]
    )
    return "\n".join(lines)
