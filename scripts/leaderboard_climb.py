from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
OUTPUT = CAMPAIGN / "eda/leaderboard_climb"


def read_json(path: Path) -> dict:
    """Read saved evidence without fitting, scoring new predictions or using the network."""
    return json.loads(path.read_text())


def pooled_wp(stats: list) -> float:
    """Reconstruct clipped global WP from archived class/target sufficient statistics."""
    correlations = []
    for target in range(2):
        n, sy, sp, syy, spp, syp = [sum(group[target][i] for group in stats) for i in range(6)]
        covariance = syp / n - sy * sp / n**2
        correlations.append(
            covariance / ((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)) ** 0.5
        )
    return sum(correlations) / len(correlations)


def build_history() -> dict:
    """Combine portal results, the historical table and explicitly scoped local evidence."""
    registry_path = ROOT / "research/lb/registry.json"
    history_path = CAMPAIGN / "LEADERBOARD_HISTORY_METADATA.json"
    stats_path = ROOT / "research/lb/holdout_stats.json"
    fixed_path = CAMPAIGN / "checks/e450_full_valid_fixed_models_v1.json"
    card_path = CAMPAIGN / "checks/e459_validation_scorecard_v1.json"
    cv_path = CAMPAIGN / "checks/combo_merged_all_folds_report.json"
    registry, metadata, historic, fixed, card, cv = map(
        read_json, (registry_path, history_path, stats_path, fixed_path, card_path, cv_path)
    )
    if sum(historic["classes"]) != 368:
        raise ValueError("historical local series is not the documented 368-sequence panel")
    rows = [
        {
            "kind": "own",
            "time": entry["t"],
            "name": entry["name"],
            "public": entry["s"],
            "approx": True,
            "what": metadata["changes"][entry["name"]],
            "evidence": metadata["historical_source"],
        }
        for entry in metadata["early_entries"]
    ]
    for entry in registry["subs"]:
        row = {
            "kind": "own",
            "time": entry["portal_time"].replace(" ", "T"),
            "name": entry["name"],
            "public": entry["public"],
            "approx": False,
            "what": metadata["changes"][entry["name"]],
            "id": entry["submission_id"],
            "runtime_seconds": entry["runtime_seconds"],
            "memory_mb": entry["memory_mb"],
            "zip_sha256": entry.get("zip_sha256"),
            "evidence": entry.get("latest_portal_evidence", entry.get("evidence")),
        }
        if entry["name"] in historic["subs"]:
            row["validation_368"] = pooled_wp(historic["subs"][entry["name"]]["stats"])
            row["validation_source"] = (
                "Archived clipped pooled moments; research/lb/holdout_stats.json"
            )
        rows.append(row)
    names = {row["name"]: row for row in rows}
    old = names["D256_single_distilled"]
    old["validation_full"] = card["scores"]["D256_public_0.6641"]["all_supplied_validation"]["wp"]
    old["validation_368"] = card["scores"]["D256_public_0.6641"]["excluded_from_final_fit"]["wp"]
    for label, key in (
        ("c20261004_combo_s3_i16", "COMBO_S3"),
        ("c20261004_combo_s4_i16", "COMBO_S4"),
    ):
        names[label]["validation_full"] = fixed["scores"][key]["all_supplied_validation"]["wp"]
        names[label]["validation_368"] = fixed["scores"][key]["excluded_from_final_fit"]["wp"]
        names[label]["validation_source"] = (
            "Fixed FP32 cached predictions, 1873 supplied /368 excluded sequences; E450 saved scorecard"
        )
    names["c20261004_combo_s3_i16"]["cv_recipe"] = cv["pooled"]["exact_global_wp"]["COMBO"]["wp"]
    for label in ("c20261004_combo_head_s3_i16", "c20261004_combo_parent_s3_i16"):
        names[label].update(
            {
                "expected_public": 0.665,
                "expected_low": 0.664,
                "expected_high": 0.666,
                "expectation_note": "Historical 23:41 Kyiv shortlist heuristic; range was illustrative, not a calibrated interval",
            }
        )
    names["c20261004_combo_s4_i16"].update(
        {
            "expected_public": 0.6655,
            "expectation_note": "Historical seed-3 incumbent anchor; no calibrated seed-4 forecast or interval",
        }
    )
    for entry in metadata["team_entries"]:
        rows.append(
            {
                "kind": "team",
                "time": entry["t"],
                "name": entry["name"],
                "public": entry["s"],
                "approx": True,
                "what": entry["what"],
                "evidence": metadata["historical_source"],
            }
        )
    for entry in metadata["historical_forecasts"]:
        note = "Historical stated range; calibration not independently established."
        if entry["name"].startswith("S4D"):
            note += " Its historical page timestamp is later than the portal upload: not proof of a preregistered forecast."
        rows.append(
            {
                "kind": "forecast",
                "time": entry["t"],
                "name": entry["name"].replace("(not uploaded)", "(archived forecast)"),
                "expected_public": entry["s"],
                "expected_low": entry["lo"],
                "expected_high": entry["hi"],
                "approx": True,
                "what": entry["what"],
                "expectation_note": note,
                "evidence": metadata["historical_source"],
            }
        )
    for entry in registry["failed_submissions"]:
        rows.append(
            {
                "kind": "error",
                "time": f"{entry['date']}T{entry['portal_time']}",
                "name": entry["name"],
                "id": entry["submission_id"],
                "what": entry["error"],
                "evidence": entry["evidence"],
                "zip_sha256": entry["zip_sha256"],
                "public": None,
            }
        )
    rows.sort(key=lambda row: row["time"])
    best = None
    for index, row in enumerate(rows):
        row["index"] = index
        row["best_before"] = best
        if row["kind"] in ("own", "team"):
            row["delta_best_before"] = row["public"] - best if best is not None else None
        if row["kind"] == "own":
            row["new_record"] = best is None or row["public"] > best
            best = max(best, row["public"]) if best is not None else row["public"]
        row["running_best"] = best
    return {
        "schema": "connectome-leaderboard-history-v1",
        "metric": "global_wp_clip2_v1",
        "time_zone": "Europe/Kyiv",
        "portal_clock_assumption": "Portal timezone was not explicitly supplied; displayed clock values are interpreted as Kyiv per user request.",
        "historical_leader": {
            "score": 0.6677,
            "scope": "Historical October 2 review reference; not a live leaderboard lookup",
        },
        "target": 0.6660,
        "best_public": best,
        "research_status": "paused",
        "rows": rows,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (registry_path, history_path, stats_path, fixed_path, card_path, cv_path)
        },
        "validation_scope": "Full validation includes 1505 fitted sequences; 368 excluded sequences are historically exposed. Raw local scores are not public expectations. Missing observations remain missing. CV is the seed-3 recipe, not CV of the final seed-4 checkpoint.",
    }


def main() -> None:
    """Write an offline interactive HTML and its machine-readable evidence table."""
    history = build_history()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "leaderboard_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n"
    )
    template = (ROOT / "research/visual/leaderboard_climb.template.html").read_text()
    embedded = json.dumps(history, ensure_ascii=False).replace("<", "\\u003c")
    html = template.replace("__HISTORY_JSON__", embedded)
    if "__HISTORY_JSON__" in html:
        raise ValueError("unresolved HTML data placeholder")
    (OUTPUT / "leaderboard_climb.html").write_text(html)
    print(
        json.dumps(
            {
                "html": str(OUTPUT / "leaderboard_climb.html"),
                "rows": len(history["rows"]),
                "best": history["best_public"],
            }
        )
    )


if __name__ == "__main__":
    main()
