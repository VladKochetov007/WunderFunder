"""Publish one bounded B300 canonical readout result to PR74."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

NB = Path(r"C:\Users\admin\Documents\Codex\artifacts\b300-causal-stem-20261110")
RPC = NB / "b300-rpc"
PUBLISHER_PATH = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-b300-causal-stem-20261110\research\experiment_drafts\b300-causal-stem-execution-20261110\reporting\publish_pr74.py"
)
NATIVE_SUMMARY = "/data/epanov_s/tabdata-causal-stem-b300-20261110-r1/control/b300-driver-r1/readout-r2/complete.json"
SOURCE_SHA = "959fc8a335f97a83de31c12dfa6a24f5db4776b78b70bf45f7890cf09b96b5d6"
SOURCE_COMMIT = "8a8afec964985a6ba8b13d1c9af1ab00e14c6504"
AUTHORITY_SHA = "74663454d5c8ace8f7c1cdeb6c651e3d9a93fb9b2e1c4cc756bd03fb45df3f77"
FIT_ADMISSION_SHA = "fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SUMMARY_SCHEMA = "wnn.b300-pr74-fourpairs-readout-summary.r2"
CELL_SCHEMA = "wnn.b300-pr74-fourpairs-readout-cell.r2"
CELLS = tuple(f"{arm}_s{seed}" for seed in (203201, 203202) for arm in ("plain", "k1", "k16"))
CONTRASTS = tuple(
    (seed, f"K16 minus {arm}", f"k16_s{seed}", f"{arm}_s{seed}")
    for seed in (203201, 203202)
    for arm in ("k1", "plain")
)
VALID_SEQUENCES, VALID_ROWS = 1_873, 3_528_273
WAVE_DEADLINE_EPOCH = 1_794_359_530.5156887
POLL_SECONDS, MAX_SECONDS = 20, 21_600
LOG = logging.getLogger("b300_readout_publication_r2")

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
    """Load an existing local helper without starting its command line."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"local helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _module(PUBLISHER_PATH, "b300_pr74_progress_publisher")


def atomic_json(path: Path, value: object) -> None:
    """Atomically save local watcher status without touching native readout data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex[:8])
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _read_pins() -> dict[str, Any]:
    """Check the local authority, admission and official split bytes before polling."""
    paths = {
        "authority": NB / "AUTHORITY_R2.json",
        "admission": NB / "ROOT-fit-admission-r1.json",
        "split": NB / "chronological_primary.json",
    }
    expected = {"authority": AUTHORITY_SHA, "admission": FIT_ADMISSION_SHA, "split": SPLIT_SHA}
    values: dict[str, Any] = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f"frozen local {name} bytes changed")
        values[name] = json.loads(raw)
    authority, admission = values["authority"], values["admission"]
    profiles = admission.get("profiles")
    if (
        authority.get("source", {}).get("source_sha256") != SOURCE_SHA
        or authority.get("source", {}).get("commit") != SOURCE_COMMIT
        or admission.get("approved") is not True
        or admission.get("source_sha256") != SOURCE_SHA
        or admission.get("authority_sha256") != AUTHORITY_SHA
        or admission.get("split_sha256") != SPLIT_SHA
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("local authority/admission identity differs from the B300 readout")
    return {**values, "profiles": profiles}


def payload(path: Path) -> dict[str, Any]:
    """Extract the JSON envelope from one response in the existing local RPC queue."""
    value = json.loads(path.read_bytes())
    for line in str(value.get("output", "")).splitlines():
        try:
            result = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(result, dict) and type(result.get("ready")) is bool:
            return result
    raise ValueError("B300 readout RPC returned no readiness envelope")


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _cell_score(name: str, cell: dict[str, Any], pins: dict[str, Any]) -> dict[str, Any]:
    """Validate one producer cell against its exact config and fixed-valid endpoint."""
    profile = pins["profiles"][name]
    score = cell.get("score")
    shas = (
        cell.get("fit_model_sha256"),
        cell.get("prediction_manifest_sha256"),
        cell.get("score_sha256"),
    )
    if (
        cell.get("schema") != CELL_SCHEMA
        or cell.get("cell") != name
        or cell.get("status") != "readout_complete"
        or cell.get("source_sha256") != SOURCE_SHA
        or cell.get("fit_admission_sha256") != FIT_ADMISSION_SHA
        or cell.get("split_sha256") != SPLIT_SHA
        or cell.get("config_sha256") != profile.get("config_sha256")
        or cell.get("prediction_world_size") != 2
        or cell.get("sequences") != VALID_SEQUENCES
        or cell.get("scored_rows") != VALID_ROWS
        or not math.isclose(
            float(cell.get("monitor_match_abs_tolerance", -1)), 1e-5, rel_tol=0, abs_tol=1e-15
        )
        or not isinstance(score, dict)
        or score.get("metric_id") != "global_wp_clip2_v1"
        or score.get("selected_rows") != VALID_ROWS
        or not all(P.P._is_sha256(value) for value in shas)
        or not all(_finite(score.get(metric)) for metric in ("t0", "t1", "wp"))
        or not math.isclose(
            float(score["wp"]),
            0.5 * (float(score["t0"]) + float(score["t1"])),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"cell failed fixed-endpoint/full-valid identity checks: {name}")
    return {metric: float(score[metric]) for metric in ("t0", "t1", "wp")}


def render(envelope: dict[str, Any], pins: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate the actual R2 producer schema and form only the predeclared paired contrasts."""
    summary, native_sha = envelope.get("summary"), envelope.get("native_summary_sha256")
    if (
        envelope.get("ready") is not True
        or not isinstance(summary, dict)
        or not P.P._is_sha256(native_sha)
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("source_sha256") != SOURCE_SHA
        or summary.get("source_commit") != SOURCE_COMMIT
        or summary.get("authority_sha256") != AUTHORITY_SHA
        or summary.get("fit_admission_sha256") != FIT_ADMISSION_SHA
        or summary.get("split_sha256") != SPLIT_SHA
        or summary.get("prediction_world_size") != 2
        or not _finite(summary.get("wave_deadline_epoch"))
        or abs(float(summary["wave_deadline_epoch"]) - WAVE_DEADLINE_EPOCH) > 1.0
    ):
        raise ValueError("native B300 R2 summary differs from frozen source/admission/deadline")
    cells = summary.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(CELLS):
        raise ValueError("native B300 R2 summary does not contain the exact six cells")
    scores: dict[str, dict[str, float]] = {}
    for name in CELLS:
        cell = cells[name]
        if (
            not isinstance(cell, dict)
            or cell.get("schema") != CELL_SCHEMA
            or cell.get("cell") != name
        ):
            raise ValueError(f"malformed native R2 cell record: {name}")
        if cell.get("status") == "readout_complete":
            scores[name] = _cell_score(name, cell, pins)

    panel = summary.get("panel")
    complete = summary.get("status") == "complete_exposed_fixed4E_screen"
    if summary.get("status") not in {"complete_exposed_fixed4E_screen", "partial_or_failed"}:
        raise ValueError("native R2 summary has an unknown terminal status")
    if complete:
        if set(scores) != set(CELLS) or not isinstance(panel, dict):
            raise ValueError("complete summary lacks six qualified cells or the canonical panel")
        if (
            panel.get("status") != "complete_exposed_fixed4E_screen"
            or panel.get("exposed_validation") is not True
            or panel.get("world_size_prediction") != 2
            or not P.P._is_sha256(panel.get("comparison_sha256"))
            or not isinstance(panel.get("comparison_path"), str)
            or not panel["comparison_path"].endswith("/readout-r2/panel-input/comparison.json")
        ):
            raise ValueError("canonical comparison-panel receipt is incomplete")
        contrasts = []
        for seed, label, left, right in CONTRASTS:
            contrasts.append(
                {
                    "seed": seed,
                    "contrast": label,
                    "left_cell": left,
                    "right_cell": right,
                    **{
                        metric: scores[left][metric] - scores[right][metric]
                        for metric in ("t0", "t1", "wp")
                    },
                }
            )
    else:
        if panel is not None:
            raise ValueError("partial readout summary unexpectedly claims a completed panel")
        contrasts = None

    lines = [
        "## B300 PR74 canonical readout R2",
        f"Native summary SHA `{native_sha}`; status `{summary['status']}`.",
        f"Source `{SOURCE_SHA}` (`{SOURCE_COMMIT}`); authority `{AUTHORITY_SHA}`; fit admission `{FIT_ADMISSION_SHA}`; split `{SPLIT_SHA}`.",
        "Canonical prediction export and scorer reuse the repeatedly exposed official-valid partition; this is pipeline verification, not a fresh validation or public-score claim.",
        "",
        "### Six fixed-E4 cells",
    ]
    for name in CELLS:
        cell = cells[name]
        detail = (
            f"; producer model SHA `{cell['fit_model_sha256'][:12]}…`" if name in scores else ""
        )
        error = cell.get("error")
        lines.append(
            f"- `{name}`: `{cell.get('status')}`{detail}{'; ' + str(error)[:300] if error else ''}."
        )
    if complete:
        lines.extend(
            [
                "",
                f"Canonical comparison panel `{panel['comparison_sha256']}` at `{panel['comparison_path']}`.",
                "",
                "### Full-valid scores",
            ]
        )
        for name in CELLS:
            score = scores[name]
            lines.append(
                f"- `{name}`: t0={score['t0']:.8f}, t1={score['t1']:.8f}, WP={score['wp']:.8f}."
            )
        lines.extend(["", "### Predeclared paired contrasts (K16 minus comparator, same seed)"])
        for row in contrasts:
            lines.append(
                f"- Seed {row['seed']}, {row['contrast']}: t0={row['t0']:+.8f}, t1={row['t1']:+.8f}, WP={row['wp']:+.8f}."
            )
    else:
        lines.append(
            "\nPaired contrasts withheld until all six cells and the canonical comparison panel pass."
        )
    lines.append(
        "\nThese exposed-valid results do not establish a public-score gain or statistical significance."
    )
    local = {
        "schema": "wnn.b300-pr74-readout-publication-r2.v1",
        "status": summary["status"],
        "native_summary_sha256": native_sha,
        "source_sha256": SOURCE_SHA,
        "authority_sha256": AUTHORITY_SHA,
        "fit_admission_sha256": FIT_ADMISSION_SHA,
        "split_sha256": SPLIT_SHA,
        "cells": {name: cells[name].get("status") for name in CELLS},
        "model_scores": scores,
        "paired_contrasts": contrasts,
        "panel": panel,
    }
    return "\n".join(lines), local


def run(args: argparse.Namespace) -> int:
    """Poll only NB's existing RPC and publish one terminal result before the fixed deadline."""
    pins = _read_pins()
    wall_deadline = min(WAVE_DEADLINE_EPOCH, time.time() + args.maxseconds)
    mono_deadline = time.monotonic() + max(0.0, wall_deadline - time.time())
    if mono_deadline <= time.monotonic():
        raise TimeoutError("original B300 wave deadline has passed")
    client = None
    if not args.dryrun:
        transport = _module(P.CLIENT_PATH, "b300_readout_pr74_github")
        client = P._PR74Client(transport.ProgressGitHub({74}, deadline=mono_deadline))
    state_path = args.state_file or NB / "reporting" / "publisher-state-pr74-readout-r2.json"
    status_path = NB / "reporting" / "readout-publication-r2.json"
    result: dict[str, Any] | None = None
    pending: str | None = None
    while time.monotonic() < mono_deadline:
        try:
            if result is None:
                if pending is None:
                    pending = "b300-readout-r2-" + uuid.uuid4().hex
                    atomic_json(RPC / "requests" / f"{pending}.json", {"code": COLLECT})
                reply = RPC / "results" / f"{pending}.json"
                if reply.exists():
                    result = payload(reply)
                    pending = None
                    if result.get("ready") is not True:
                        result = None
            if result is not None:
                try:
                    message, local = render(result, pins)
                except (TypeError, ValueError, KeyError) as error:
                    atomic_json(
                        status_path,
                        {
                            "status": "terminal_summary_rejected",
                            "error": str(error)[:500],
                            "at": time.time(),
                        },
                    )
                    LOG.error("terminal summary rejected: %s", str(error)[:300])
                    return 2
                marker = f"<!-- wnn-b300-pr74-readout-r2-{local['native_summary_sha256']} -->"
                atomic_json(NB / "readout-final-summary-r2.json", local)
                if args.dryrun:
                    print(message + "\n\n" + marker, flush=True)
                    return 0
                outcome = P.P.publish_once(
                    client, state_path, marker, message + "\n\n" + marker, time.time()
                )
                if outcome not in {"posted", "reconciled-existing", "already-posted"}:
                    raise RuntimeError(f"unconfirmed publication outcome: {outcome}")
                atomic_json(status_path, {"status": outcome, "marker": marker, "at": time.time()})
                print(outcome, marker, flush=True)
                return 0
            atomic_json(status_path, {"status": "waiting_for_native_readout_r2", "at": time.time()})
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
    """Run the ROOT-bounded local publisher; credentials are read only by PR74 transport."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxseconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.maxseconds <= MAX_SECONDS:
        parser.error(f"--maxseconds must be in [1, {MAX_SECONDS}]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
