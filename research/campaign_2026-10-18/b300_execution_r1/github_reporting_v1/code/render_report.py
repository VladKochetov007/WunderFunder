"""Render endpoint evidence without turning incomplete comparisons into claims."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime

CAMPAIGN = "tabdata-b300-20261018"
PUBLIC_RESULT_FIELDS = {
    "id", "arm", "fold", "seed", "wp", "metric_id", "pair_id", "source", "source_commit",
    "source_sha256", "checkpoint_sha256", "checkpoint_payload_sha256", "prediction_sha256",
    "score_sha256", "prediction_metadata_sha256", "receipt_sha256", "fit_recovery_sha256",
    "fit_receipt_sha256", "fit_timing_seconds", "fit_timing_note", "receipt_digests",
    "archive_manifest_path", "archive_manifest_sha256", "evidence_archive", "weights_archive",
    "export_receipt_sha256", "grade_receipt_sha256", "grade_sha256", "grade_command_sha256",
}


def canonical_sha(value: object) -> str:
    """Hash a deterministic JSON object used for deduplication and evidence binding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def marker(experiment: str, kind: str, digest: str = "") -> str:
    """Identify only comments belonging to this campaign and experiment."""
    return f"<!-- {CAMPAIGN}:{experiment}:{kind}:{digest} -->"


def validate_snapshot(snapshot: dict) -> None:
    """Reject malformed/duplicate scores before building a public-facing report."""
    rows = snapshot["completed"]
    if len({row["id"] for row in rows}) != len(rows) or len(rows) > snapshot["expected_cells"]:
        raise ValueError("Completed endpoint identities are duplicated or exceed the frozen panel")
    for row in rows:
        if not math.isfinite(row["wp"]) or not -1.00000001 <= row["wp"] <= 1.00000001:
            raise ValueError("Official endpoint score is nonfinite or outside Pearson bounds")
        digest = row["receipt_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Endpoint lacks a complete SHA256 binding")
    raw = json.dumps(snapshot)
    import re

    if re.search(r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)", raw):
        raise ValueError("Report rejected because it resembles credential material")


def summary_table(snapshot: dict) -> str:
    """Use equal fold weights and pair only identical fold/seed identities."""
    by_arm: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for row in snapshot["completed"]:
        key = row["fold"], row["seed"]
        if key in by_arm[row["arm"]]:
            raise ValueError("Duplicate arm/fold/seed score")
        by_arm[row["arm"]][key] = row["wp"]
    reference = "baseline" if snapshot["experiment_id"] == "PR15" else "wide"
    reference_rows = by_arm.get(reference, {})
    lines = [
        "| Arm | Scored / planned | Mean WP¹ | Paired Δ¹ | Matched cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, values in sorted(by_arm.items()):
        folds = defaultdict(list)
        for (fold, _), score in values.items():
            folds[fold].append(score)
        mean = sum(sum(v) / len(v) for v in folds.values()) / len(folds)
        paired = defaultdict(list)
        for key in values.keys() & reference_rows.keys():
            paired[key[0]].append(values[key] - reference_rows[key])
        delta = sum(sum(v) / len(v) for v in paired.values()) / len(paired) if paired else None
        delta_text = "—" if arm == reference or delta is None else f"{delta:+.8f}"
        lines.append(
            f"| {arm} | {len(values)} / {snapshot['expected_per_arm']} | {mean:.8f} | "
            f"{delta_text} | {sum(map(len, paired.values()))} |"
        )
    if not by_arm:
        lines.append("| No fixed-checkpoint official endpoints yet | 0 | — | — | 0 |")
    lines.append(
        "\n¹ Equal mean of the available fold means. A delta uses only matching fold/seed cells. "
        "Until all four folds and the complete frozen matrix are present, it is a partial statistic. "
        "This table is not a confidence interval, significance test or public-score forecast."
    )
    return "\n".join(lines)


def report(snapshot: dict, *, new_rows: list[dict] | None, evidence_url: str) -> str:
    """Render a contract-scoped update with original evidence and explicit omissions."""
    validate_snapshot(snapshot)
    experiment = snapshot["experiment_id"]
    kind = "status" if new_rows is None else "results"
    digest = "" if new_rows is None else canonical_sha(new_rows)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    complete = len(snapshot["completed"]) == snapshot["expected_cells"]
    state = (
        "All fixed-checkpoint endpoints are ready; inferential review remains pending"
        if complete
        else "Partial / in progress"
    )
    lines = [
        marker(experiment, kind, digest),
        f"### tabdata {experiment}: {kind} update — {stamp}",
        f"**Status:** {state}. Scored {len(snapshot['completed'])}/{snapshot['expected_cells']}; "
        f"in-flight cells (including delivery) {snapshot['active']}; unstarted {snapshot['pending']}.",
        snapshot.get("contract_note", ""),
        "**Frozen comparison / provenance:** strict FP32, autocast and TF32 off; unchanged "
        "`global_wp_clip2_v1`; fixed checkpoint; original panel, folds and fitting seeds. "
        "Profiles, inference aliases and recovered exports are not additional training replications.",
        "```json\n" + json.dumps(snapshot.get("source", {}), sort_keys=True, indent=2) + "\n```",
        summary_table(snapshot),
        "**Validation and decision:** these are exposed development folds. Training-seed uncertainty, "
        "component uncertainty and selection/public-transfer uncertainty are distinct. Complete CP30 "
        "fits do not by themselves establish statistical significance. Preserve negative and "
        "inconclusive results; no automatic merge, pruning, checkpoint selection or model promotion.",
        f"**Evidence / contract trace:** {evidence_url}. Large weights and supplied data remain in "
        "the approved private server storage; this reporter does not upload them.",
    ]
    if new_rows is not None:
        lines.extend(
            marker(
                experiment,
                "endpoint",
                canonical_sha({"id": row["id"], "receipt_sha256": row["receipt_sha256"]}),
            )
            for row in new_rows
        )
        lines += [
            f"**New or corrected scored endpoints in this update:** {len(new_rows)}.",
            "<details><summary>Exact endpoint identities and hashes</summary>\n\n```json\n"
            + json.dumps(
                [
                    {k: v for k, v in row.items() if k in PUBLIC_RESULT_FIELDS}
                    for row in new_rows
                ],
                sort_keys=True,
                indent=2,
            )
            + "\n```\n</details>",
        ]
    if snapshot.get("errors"):
        lines += [
            "**Retained failures / rejected evidence:**",
            *[f"- {str(e)[:600]}" for e in snapshot["errors"][:20]],
        ]
    lines.append(
        "Automated reporting only: no fitting, retries of scientific fits, export or grading was launched."
    )
    return "\n\n".join(lines)
