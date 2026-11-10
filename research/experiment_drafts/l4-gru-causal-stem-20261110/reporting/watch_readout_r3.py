"""Wait for the native causal-stem R3 readout summary and publish it once to PR79."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\admin\Documents\Codex\artifacts\l4-causal-stem-20261110")
RPC = Path(r"C:\Users\admin\Documents\Codex\artifacts\l4-next70-20261110\l4-rpc")
WORK = Path(r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-causal-stem-20261110")
P77_PATH = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\publish_pr77.py"
)
CLIENT_PATH = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-architecture-execution-20261107\scripts\sequence_progress_github.py"
)
NATIVE_SUMMARY = "/data/lc-epanov_s/tabdata-stem-20261110-r1/control/readout-r3/complete.json"
SOURCE_SHA = "b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188"
SOURCE_COMMIT = "336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
ADMISSION_SHA = "48589244a653cc76a55abd3ffcc5113eccc81488722ac96599f4a8befd8cf5b7"
SUMMARY_SCHEMA = "wnn.l4-gru-causal-stem-readout-summary.r3"
CELL_SCHEMA = "wnn.l4-gru-causal-stem-readout-cell.r3"
CELLS = tuple(f"{arm}_s{seed}" for seed in (203301, 203302) for arm in ("plain", "k1", "k16"))
CONTRASTS = tuple(
    (seed, f"K16 minus {arm}", f"k16_s{seed}", f"{arm}_s{seed}")
    for seed in (203301, 203302)
    for arm in ("k1", "plain")
)
VALID_ROWS = 3_528_273
WAVE_DEADLINE_EPOCH = 1_794_349_606.643092
MAX_SECONDS = 21_600
POLL_SECONDS = 20
LOG = logging.getLogger("stem_readout_publication_r3")

COLLECT = f"""import hashlib,json
from pathlib import Path
p=Path({NATIVE_SUMMARY!r})
if p.exists():
 b=p.read_bytes()
 assert len(b)<=2000000
 print(json.dumps({{'ready':True,'native_summary_sha256':hashlib.sha256(b).hexdigest(),'summary':json.loads(b)}},allow_nan=False))
else: print(json.dumps({{'ready':False}}))
"""


def _module(path: Path, name: str) -> Any:
    """Load an existing local publisher helper without starting its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"local helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(P77_PATH, "l4_next70_pr77_readout_helper")


def atomic_json(path: Path, value: object) -> None:
    """Atomically save the small local readout or publication receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex[:8])
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def payload(path: Path) -> dict[str, Any]:
    """Extract one result envelope from the existing local RPC response."""
    value = json.loads(path.read_bytes())
    for line in str(value.get("output", "")).splitlines():
        try:
            result = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(result, dict) and type(result.get("ready")) is bool:
            return result
    raise ValueError("readout RPC did not return a bounded readiness envelope")


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def render(value: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate the frozen R3 summary and render one terminal PR update."""
    summary = value.get("summary")
    native_sha = value.get("native_summary_sha256")
    if (
        value.get("ready") is not True
        or not isinstance(summary, dict)
        or not P._is_sha256(native_sha)
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("source_sha256") != SOURCE_SHA
        or summary.get("source_commit") != SOURCE_COMMIT
        or summary.get("split_sha256") != SPLIT_SHA
        or summary.get("admission_sha256") != ADMISSION_SHA
        or not _finite(summary.get("wave_deadline_epoch"))
        or abs(float(summary["wave_deadline_epoch"]) - WAVE_DEADLINE_EPOCH) > 1.0
    ):
        raise ValueError(
            "native readout summary does not match the frozen R3 source/split contract"
        )
    cells = summary.get("cells")
    scores = summary.get("model_scores")
    if not isinstance(cells, dict) or set(cells) != set(CELLS) or not isinstance(scores, dict):
        raise ValueError("native R3 summary must contain the exact six admitted cells")
    for name, cell in cells.items():
        if (
            not isinstance(cell, dict)
            or cell.get("schema") != CELL_SCHEMA
            or cell.get("cell") != name
        ):
            raise ValueError(f"native R3 cell record is malformed: {name}")

    complete = summary.get("status") == "complete"
    if summary.get("status") not in {"complete", "partial_or_failed"}:
        raise ValueError("native R3 summary has an unknown terminal status")
    expected_scored = (
        set(CELLS)
        if complete
        else {name for name, cell in cells.items() if cell.get("status") == "readout_complete"}
    )
    if set(scores) != expected_scored:
        raise ValueError("native model score membership differs from cell completion status")
    for name, score in scores.items():
        if (
            not isinstance(score, dict)
            or score.get("within_tolerance") is not True
            or score.get("selected_rows") != VALID_ROWS
            or not all(_finite(score.get(metric)) for metric in ("t0", "t1", "wp"))
        ):
            raise ValueError(
                f"native score is incomplete or outside the fixed-valid contract: {name}"
            )

    paired = summary.get("paired_contrasts")
    if complete:
        if any(cells[name].get("status") != "readout_complete" for name in CELLS):
            raise ValueError("complete native summary contains an unsuccessful cell")
        if not isinstance(paired, list) or len(paired) != len(CONTRASTS):
            raise ValueError("complete native summary lacks the four predeclared paired contrasts")
        for row, (seed, contrast, left, right) in zip(paired, CONTRASTS, strict=True):
            if not isinstance(row, dict) or any(
                row.get(key) != expected
                for key, expected in (
                    ("seed", seed),
                    ("contrast", contrast),
                    ("left_cell", left),
                    ("right_cell", right),
                )
            ):
                raise ValueError(
                    "native contrast order or pair identity differs from the frozen plan"
                )
            for metric in ("t0", "t1", "wp"):
                observed, expected = row.get(metric), scores[left][metric] - scores[right][metric]
                if not _finite(observed) or not math.isclose(
                    float(observed), expected, rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "native paired contrast does not equal same-seed score subtraction"
                    )
    elif paired is not None:
        raise ValueError("partial readout must not contain cohort contrasts")

    lines = [
        "## L4 causal-stem R3 fixed-E6 readout",
        f"Native summary SHA `{native_sha}`; terminal status `{summary['status']}`.",
        f"Source `{SOURCE_SHA}` (`{SOURCE_COMMIT}`); split `{SPLIT_SHA}`.",
        "This re-export and canonical scorer reuse the repeatedly exposed official validation set; it is a pipeline agreement check, not fresh validation or public-score evidence.",
        "",
        "### Six fixed-E6 cells",
    ]
    for name in CELLS:
        cell = cells[name]
        error = cell.get("error")
        suffix = f"; {str(error)[:300]}" if error else ""
        lines.append(f"- `{name}`: `{cell.get('status')}`{suffix}.")
    if complete:
        lines.extend(["", "### Verified canonical full-valid scores"])
        for name in CELLS:
            score = scores[name]
            lines.append(
                f"- `{name}`: t0={score['t0']:.8f}, t1={score['t1']:.8f}, WP={score['wp']:.8f} over {score['selected_rows']:,} rows."
            )
        lines.extend(["", "### Predeclared same-seed paired contrasts (K16 minus comparator)"])
        for row in paired:
            lines.append(
                f"- Seed {row['seed']}, {row['contrast']}: t0={row['t0']:+.8f}, t1={row['t1']:+.8f}, WP={row['wp']:+.8f}."
            )
    else:
        lines.append(
            "\nPaired endpoint contrasts withheld because the six-cell readout did not complete."
        )
    lines.append(
        "\nThe native producer receipt and model bytes remain the source of record; this local publisher validates summary identity and score arithmetic."
    )
    local = {
        "schema": "wnn.l4-gru-causal-stem-readout-publication-r3.v1",
        "status": summary["status"],
        "native_summary_sha256": native_sha,
        "source_sha256": SOURCE_SHA,
        "source_commit": SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA,
        "cells": {name: cells[name].get("status") for name in CELLS},
        "model_scores": scores,
        "paired_contrasts": paired,
        "interpretation": summary.get("interpretation"),
    }
    return "\n".join(lines), local


class _PR79Client:
    """Route the reused PR77 idempotency helper to ROOT's PR79 allowlist."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def find_report(self, _number: int, marker: str, prepared_at: float) -> Any:
        return self.client.find_report(79, marker, prepared_at)

    def post_report(self, _number: int, body: str, *, before_post: Any) -> Any:
        return self.client.post_report(79, body, before_post=before_post)


def _load_client(deadline: float) -> _PR79Client:
    """Load the existing GitHub transport with only PR79 allowlisted."""
    client_module = _module(CLIENT_PATH, "stem_readout_pr79_github")
    return _PR79Client(client_module.ProgressGitHub({79}, deadline=deadline))


def run(args: argparse.Namespace) -> int:
    """Poll the existing local RPC until R3 completion or the original wave deadline."""
    if not 1 <= args.maxseconds <= MAX_SECONDS:
        raise ValueError(f"--maxseconds must be in [1, {MAX_SECONDS}]")
    wall_deadline = min(WAVE_DEADLINE_EPOCH, time.time() + args.maxseconds)
    mono_deadline = time.monotonic() + max(0.0, wall_deadline - time.time())
    if mono_deadline <= time.monotonic():
        raise TimeoutError("original L4 readout publication deadline has passed")
    client = None if args.dryrun else _load_client(mono_deadline)
    state_path = args.state_file or ROOT / "reporting" / "publisher-state-pr79-readout-r3.json"
    status_path = ROOT / "reporting" / "readout-publication-r3.json"
    result: dict[str, Any] | None = None
    pending: str | None = None
    while time.monotonic() < mono_deadline:
        try:
            if result is None:
                if pending is None:
                    pending = "stem-readout-r3-" + uuid.uuid4().hex
                    atomic_json(RPC / "requests" / f"{pending}.json", {"code": COLLECT})
                reply = RPC / "results" / f"{pending}.json"
                if reply.exists():
                    result = payload(reply)
                    pending = None
                    if result.get("ready") is not True:
                        result = None
            if result is not None:
                message, local = render(result)
                marker = f"<!-- wnn-l4-stem-pr79-readout-r3-{local['native_summary_sha256']} -->"
                atomic_json(ROOT / "readout-final-summary-r3.json", local)
                if args.dryrun:
                    print(message + "\n\n" + marker, flush=True)
                    return 0
                outcome = P.publish_once(
                    client, state_path, marker, message + "\n\n" + marker, time.time()
                )
                atomic_json(status_path, {"status": outcome, "marker": marker, "at": time.time()})
                print(outcome, marker, flush=True)
                return 0
            atomic_json(status_path, {"status": "waiting_for_native_readout_r3", "at": time.time()})
        except Exception as error:
            atomic_json(
                status_path,
                {
                    "status": "retry_within_original_deadline",
                    "error": str(error)[:500],
                    "at": time.time(),
                },
            )
            LOG.warning("%s: %s", type(error).__name__, str(error)[:300])
            if result is None:
                pending = None
        if args.once:
            return 0
        time.sleep(min(POLL_SECONDS, max(0.0, mono_deadline - time.monotonic())))
    atomic_json(status_path, {"status": "original_deadline_expired", "at": time.time()})
    return 2


def main() -> int:
    """Run ROOT's bounded local-RPC watcher; credentials are read only for publishing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxseconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
