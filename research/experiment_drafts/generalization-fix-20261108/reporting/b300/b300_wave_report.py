from __future__ import annotations

from b300_wave_evidence import SOURCE, SPLIT

from scripts.sequence_progress_report import _finite


def score(values: dict, prefix: str) -> str:
    """Format one actual official WP with its two targets, without forecasting gains."""
    keys = (prefix + "wp", prefix + "wp/t0", prefix + "wp/t1")
    if not all(_finite(values.get(key)) for key in keys):
        return "pending"
    return f"{values[keys[0]]:.6f} ({values[keys[1]]:.6f}, {values[keys[2]]:.6f})"


def render(value: dict) -> str:
    """Keep every arm/seed and label fixed4E endpoints separately from historical30E work."""
    rows = value["cells"]
    fits, grades = sum(r["fit_complete"] for r in rows), sum(r["endpoint_complete"] for r in rows)
    lines = [
        f"B300 ensemble-objective wave: {fits}/4 fits complete; {grades}/4 fixed4E mixture grades recorded.",
        f"Generated {value['observed_at_utc']}; cached source data and ages are explicit below.",
        "Four fresh paired-seed fits; fixed4E/3,280 updates. Earlier30E attempts remain historical/partial. Treatment changes objective curvature, correlation surrogate and prediction-std floor0.01; official evaluation remains unchanged.",
        "",
        "| Cell | Last observed phase | Successful updates | Epochs | Full-valid looks | Mixture WP (t0, t1) | Standalone z WP (t0, t1) | Source age |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        age = row.get("source_age_seconds")
        source_age = (
            "unknown"
            if age is None
            else (f"{age:.0f}s" if age >= 0 else f"clock ahead {-age:.0f}s")
        )
        epoch = f"{row['epoch']:.3f}" if _finite(row.get("epoch")) else "unknown"
        metrics = row.get("latest_valid", {})
        mixture = score(metrics, "valid/")
        if row.get("fixed_grade"):
            final = row["fixed_grade"]
            mixture = f"{final['wp']:.6f} ({final['t0']:.6f}, {final['t1']:.6f}) fixed grade"
        lines.append(
            f"| {row['cell']} | {row['phase']} | {row.get('updates', 'unknown')} | {epoch}/4 | {row.get('valid_count', 0)}/4 | {mixture} | {score(metrics, 'valid/z/')} | {source_age} |"
        )
    lines.extend(
        [
            "",
            f"Training source `{SOURCE}`; split `{SPLIT}`; frozen receiver-plan SHA `{value['panel']['manifest_sha256']}`.",
        ]
    )
    for row in rows:
        if not row.get("ids"):
            continue
        counters = (
            ", ".join(f"{k}={v:g}" for k, v in sorted(row["nonfinite"].items()))
            or "not separately recorded"
        )
        metrics = row.get("latest_valid", {})
        wmse = (
            ", ".join(
                f"{key}={metrics[key]:.6g}"
                for key in ("valid/mixture/wmse", "valid/z/wmse")
                if _finite(metrics.get(key))
            )
            or "pending"
        )
        lines.append(
            f"- {row['cell']}: config `{row['ids']['config']}`; clipping {row['clipping']}; nonfinite {counters}; same-forward WMSE {wmse}; source UTC {row.get('source_observed_utc') or 'unknown'}."
        )
        if row.get("checkpoint_sha256"):
            lines.append(f"  Producer checkpoint SHA256 `{row['checkpoint_sha256']}`.")
    lines.extend(
        [
            "",
            "Producer checkpoint/grade hashes are recorded; artifact upload and independent local checkpoint/prediction byte verification remain pending.",
            "All official validation is development-exposed. Reported z diagnostics and fixed0.9TL3+0.1z mixtures use declared weights; no optimization, winner promotion, significance or public-score forecast is implied. Scientific interpretation remains with ROOT.",
        ]
    )
    return "\n".join(lines)
