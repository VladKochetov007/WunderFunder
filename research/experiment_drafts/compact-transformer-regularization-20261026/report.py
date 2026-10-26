"""Publish verified compact-panel progress and train/inner-validation curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def tail(path: Path) -> dict:
    """Read the last complete JSON record with bounded I/O."""
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 32768))
        rows = stream.read().splitlines()
    for row in reversed(rows):
        try:
            return json.loads(row)
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


def main() -> None:
    """Update one authorized PR comment on new validation points or hourly progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    pointer = args.root / "SELECTED_PLAN.json"
    publication = args.root / "GITHUB_PR.json"
    if not pointer.exists() or not publication.exists():
        return
    plan = json.loads(Path(json.loads(pointer.read_text())["path"]).read_text())
    run = Path(plan["run_root"])
    state_path = run / "STATE.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    points = []
    for path in run.glob("*/validation/epoch-*/RESULT.json"):
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            points.append(result)
    digest = hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest()
    stamp = args.root / "GITHUB_PROGRESS.json"
    prior = json.loads(stamp.read_text()) if stamp.exists() else {}
    phase = state.get("status")
    if (
        prior.get("digest") == digest
        and prior.get("phase") == phase
        and time.time() - prior.get("unix", 0) < 3600
    ):
        return
    body = "<!-- tabdata-compact-regularization-20261026 -->\n"
    body += (
        "Four fresh compact Transformer fits: control vs dropout0.1/WD0.1, F0/F2. "
        "Five-epoch warmup/cosine schedule, peak LR1e-4; BF16/BF16x9, FP32 masters/states. "
        "Co-residency with ARC-AGI is explicitly authorized. Shared GPU utilization "
        "includes foreign work; our throughput is measured from our own rank logs.\n\n"
    )
    body += (
        "| Cell | Completed fit epochs | Last update | Online train loss |\n|---|---:|---:|---:|\n"
    )
    for job in plan["jobs"]:
        fit = run / job["name"] / "fit"
        last = tail(fit / "rank0.jsonl")
        epoch = tail(fit / "metrics.jsonl")
        loss = last.get("label_loss")
        loss_text = f"{loss:.7f}" if isinstance(loss, (int, float)) else "pending"
        body += (
            f"| {job['name']} | {epoch.get('epoch', 0)}/5 | {last.get('step', 0)} | {loss_text} |\n"
        )
    body += "\n| Cell | Epoch | Same-weight train-probe WMSE | Full inner-val WMSE | Inner-val official WP |\n|---|---:|---:|---:|---:|\n"
    summary = []
    for result in sorted(points, key=lambda x: (x["job"], x["completed_epochs"])):
        train = result["train_probe"]["metrics"]["weighted_mse_all_after_warmup"]
        valid = result["heldout"]["metrics"]
        body += (
            f"| {result['job']} | {result['completed_epochs']} | {train:.7f} | "
            f"{valid['weighted_mse_all_after_warmup']:.7f} | {valid['official_wp']:.7f} |\n"
        )
        summary.append(
            {
                key: result[key]
                for key in [
                    "job",
                    "completed_epochs",
                    "checkpoint_sha256",
                    "epoch_index_sha256",
                    "config_sha256",
                    "split_sha256",
                    "source_sha256",
                ]
            }
        )
    body += (
        "\nTraining probe has 64 deterministic fitting sequences. WMSE comparisons "
        "use rows>=99; official WP uses the heldout scored mask. Inner selection "
        "is development tuning; outer folds remain exposed reporting folds. "
        "No student-transfer or public-score gain is claimed.\n\n"
    )
    body += (
        f"Controller status: `{phase}`. Completed full validation points: **{len(points)}/20**.\n"
    )
    body += (
        "\nCheckpoint/config/split provenance:\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n"
    )
    helper = Path("/data/epanov_s/tabdata/runtime/github-reporting-v1")
    sys.path.insert(0, str(helper / "code"))
    from github_api import GitHub, read_token

    settings = json.loads((helper / "CONFIG.json").read_text())
    api = GitHub(read_token(Path(settings["token_file"])))
    if api.request("GET", "/user")["login"].lower() != "semyonepanov":
        raise ValueError("unexpected reporting account")
    pr = int(json.loads(publication.read_text())["number"])
    if prior.get("comment_id"):
        response = api.patch(prior["comment_id"], body)
    else:
        response = api.post(pr, body)
    stamp.write_text(
        json.dumps(
            {
                "unix": time.time(),
                "digest": digest,
                "phase": phase,
                "comment_id": response["id"],
                "url": response["html_url"],
                "complete_points": len(points),
            },
            indent=2,
        )
        + "\n"
    )
    if len(points) == 20 and phase in {"complete", "completed"}:
        (args.root / "DELIVERY_COMPLETE.json").write_text(
            json.dumps({"unix": time.time(), "url": response["html_url"]}) + "\n"
        )


if __name__ == "__main__":
    main()
