from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any  # Ledger rows have intentionally heterogeneous JSON schemas.

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "results_dashboard_template.html"
OUTPUT = HERE / "RESULTS_DASHBOARD_20261012.html"
PUBLIC_ROSTER = (
    ROOT / "research/campaign_2026-10-12/public_score_regression_r1/raw_portal_roster.tsv"
)
ENDPOINTS = (
    ROOT / "research/campaign_2026-10-12/public_score_regression_r1/endpoint_coverage_r1.csv"
)
LATEST_PUBLIC = (
    ROOT / "research/campaign_2026-10-12/public_runtime_quality_r1/PUBLIC_RUNTIME_QUALITY_R1.csv"
)
EXPERIMENTS = ROOT / "research/experiments.jsonl"
HYPOTHESES = ROOT / "research/hypotheses.jsonl"
REPOSITORY_PATH_PATTERN = re.compile(r"(?:research|submissions|specs|docs)/[^\s\"'<>`,;]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a complete JSONL ledger and fail with the exact malformed row."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file with its header names intact."""
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def find_ids(value: Any, pattern: re.Pattern[str]) -> list[str]:
    """Return unique stable IDs found in JSON-like values or legacy text."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def ci_fields(value: Any, path: str = "") -> list[dict[str, Any]]:
    """Collect explicitly named CI and interval values from nested results."""
    fields: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            normalized_key = str(key).lower().replace("-", "_")
            is_interval = (
                normalized_key in {"ci", "ci95", "ci_95", "ci_lower", "ci_upper"}
                or bool(re.search(r"(?:^|_)ci(?:_|\d|$)", normalized_key))
                or "confidence_interval" in normalized_key
                or "credible_interval" in normalized_key
                or "interval95" in normalized_key
            )
            if is_interval:
                fields.append({"path": child_path, "value": child})
            else:
                fields.extend(ci_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(ci_fields(child, f"{path}[{index}]"))
    return fields


def repository_file_links(value: Any) -> list[dict[str, str]]:
    """Turn existing repository paths recorded in a ledger row into links."""
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    links: list[dict[str, str]] = []
    seen: set[Path] = set()
    for match in REPOSITORY_PATH_PATTERN.finditer(serialized):
        raw_path = match.group(0).rstrip(".,:;)]}")
        candidate = (ROOT / raw_path).resolve()
        if candidate in seen or not candidate.is_file():
            continue
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            continue
        seen.add(candidate)
        links.append({"href": relpath(candidate) or "", "text": raw_path})
    return links


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def relpath(path: Path) -> str | None:
    if not path.exists():
        return None
    return Path(os.path.relpath(path, start=HERE)).as_posix()


def experiment_rows() -> list[dict[str, Any]]:
    h_pattern = re.compile(r"H-\d{3,4}")
    rows: list[dict[str, Any]] = []
    for source in load_jsonl(EXPERIMENTS):
        related = [source.get(key) for key in ("hypothesis_id", "hypotheses", "related_hypotheses")]
        h_ids = list(
            dict.fromkeys(
                identifier for value in related for identifier in find_ids(value, h_pattern)
            )
        )
        selected = {
            key: source[key]
            for key in (
                "id",
                "date",
                "kind",
                "status",
                "result",
                "claim",
                "verdict",
                "hypothesis_id",
                "hypotheses",
                "receipt",
                "artifact",
                "evidence",
                "scope",
                "exposure",
                "metric",
                "data",
                "model",
                "design",
                "limits",
                "scientific_fits_completed",
                "reproducible",
                "official_public_score",
                "public_score",
            )
            if key in source
        }
        historical_forecasts = {
            key: source[key]
            for key in ("expected_public_score", "expected_public_gain")
            if key in source
        }
        if historical_forecasts:
            selected["historical_forecast_not_actual_public_score"] = historical_forecasts
        result = source.get("result")
        ci = ci_fields(source)
        result_summary = as_text(result)
        text_for_search = (
            " ".join(
                as_text(source.get(key))
                for key in ("id", "date", "kind", "status", "claim", "verdict", "result")
            )
            + " "
            + " ".join(h_ids)
        )
        rows.append(
            {
                "id": str(source.get("id", "")),
                "date": str(source.get("date", source.get("created_utc", "")))[:10],
                "kind": as_text(source.get("kind")),
                "status": as_text(source.get("status")),
                "claim": as_text(source.get("claim")),
                "verdict": as_text(source.get("verdict")),
                "hasResult": result is not None and result != "",
                "resultSummary": result_summary,
                "ciFields": ci,
                "hypothesisIds": h_ids,
                "sources": {
                    key: source[key] for key in ("receipt", "artifact", "evidence") if key in source
                },
                "fileLinks": repository_file_links(source),
                "details": selected,
                "searchText": text_for_search,
            }
        )
    return rows


def hypothesis_rows() -> list[dict[str, Any]]:
    e_pattern = re.compile(r"E-\d{3,4}")
    rows: list[dict[str, Any]] = []
    for source in load_jsonl(HYPOTHESES):
        evidence_ids = find_ids(source.get("evidence", []), e_pattern)
        rows.append(
            {
                "id": str(source.get("id", "")),
                "family": as_text(source.get("mechanism_family", "")),
                "status": as_text(source.get("status", "")),
                "statement": as_text(source.get("statement", source.get("name", ""))),
                "evidenceIds": evidence_ids,
                "parents": find_ids(
                    source.get("parents", source.get("parent_ids", [])), re.compile(r"H-\d{3,4}")
                ),
            }
        )
    return rows


def public_rows() -> list[dict[str, Any]]:
    endpoint_rows = {row["submission_id"]: row for row in read_csv(ENDPOINTS)}
    public: list[dict[str, Any]] = []
    with PUBLIC_ROSTER.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="|"):
            endpoint = endpoint_rows.get(row["submission_id"], {})
            package = row["zip_filename"]
            zip_path = endpoint.get("zip_path", "")
            local_path = ROOT / zip_path if zip_path else Path("/missing")
            public.append(
                {
                    "time": row["display_timestamp"],
                    "id": row["submission_id"],
                    "name": row["zip_filename"],
                    "package": package,
                    "packagePath": relpath(local_path),
                    "status": row["status"],
                    "score": float(row["score"]) if row["score"] else None,
                    "runtime": row["duration_mm_ss"],
                    "localMetric": float(endpoint["local_metric"])
                    if endpoint.get("local_metric")
                    else None,
                    "protocol": endpoint.get("local_metric_protocol", ""),
                    "latestLabel": "",
                    "isLatest": False,
                    "jitter": 0,
                }
            )
    latest_names = {
        "A": "H390 s3/s4 equal blend (50/50)",
        "B": "H390 s3 + Z4 equal blend (50/50)",
        "C": "H390 s3/s4/Z4 blend (25/25/50)",
    }
    latest_times = {"A": "2026-10-12 15:06", "B": "2026-10-12 15:06", "C": "2026-10-12 12:51"}
    for row in read_csv(LATEST_PUBLIC):
        label = row["candidate"]
        package_path = ROOT / "submissions/10-12-best-r1" / row["submission_filename"]
        public.append(
            {
                "time": latest_times[label],
                "id": row["portal_submission_id"],
                "name": latest_names[label],
                "package": row["submission_filename"],
                "packagePath": relpath(package_path),
                "status": "SCORE · user-transcribed 2026-10-12 portal result",
                "score": float(row["public_score"]),
                "runtime": row["public_runtime_display"],
                "localMetric": None,
                "protocol": "No matched local CV metric for this all-data endpoint",
                "latestLabel": label,
                "isLatest": True,
                "jitter": {"A": 6, "B": -6, "C": 0}[label],
            }
        )
    public.sort(key=lambda row: (row["time"], row["id"]))
    time_counts: dict[str, int] = {}
    for row in public:
        time_counts[row["time"]] = time_counts.get(row["time"], 0) + 1
    for row in public:
        row["ambiguousTime"] = time_counts[row["time"]] > 1
    return public


def headline_findings() -> list[dict[str, Any]]:
    return [
        {
            "kind": "public",
            "label": "Observed public gain",
            "scope": "single upload per blend",
            "title": "The best observed climb is C at 0.6671",
            "result": "A (H390 s3/s4, 50:50) scored 0.6656 in 24:31. B (H390 s3 + Z4, 50:50) scored 0.6668 in 25:56. C (H390 s3/s4/Z4, 25:25:50) scored 0.6671 in 36:36. A→B is +0.0012; B→C is +0.0003.",
            "interpretation": "These packages show that the larger heterogeneous blend reached our highest recorded public result. Their mixtures change together, so these scores do not isolate the effect of adding a model or extra runtime.",
            "uncertainty": "One portal outcome per package; no public CI. Scores were transcribed from the user-provided portal view. Package/runtime checks passed, but this is not a causal comparison.",
            "links": [
                {
                    "href": "../../campaign_2026-10-12/public_runtime_quality_r1/README_R1.md",
                    "text": "A/B/C scores, runtimes and limitations",
                },
                {
                    "href": "../archive/2026-10-12/pages/03-submissions.md",
                    "text": "package members and hashes",
                },
            ],
        },
        {
            "kind": "neg",
            "label": "Transfer mismatch",
            "scope": "exposed fold 1 · Q8 package scores",
            "title": "The scratch Muon endpoint and its local blend gain did not transfer",
            "result": "Local exposed F1 predictions: distilled student 0.67103, Muon scratch 0.67195, 50:50 pair 0.67329. Public outcomes: 0.6652, 0.6616 and 0.6650 respectively.",
            "interpretation": "The pair's +0.00226 local gain over the student became −0.00020 on public. The scratch single also scored 0.0036 below the student publicly. These are exact package-related F1 receipts, but the fold was exposed and repeatedly selected; they warn against trusting this local slice as a transfer forecast.",
            "uncertainty": "No paired CI was recorded for these public differences. The local predictions are pre-native-Q8 arrays and the three packages share components and ancestry.",
            "links": [
                {
                    "href": "../../campaign_2026-10-12/public_validation_plot_r2/README.md",
                    "text": "protocol-separated local/public comparison",
                },
                {
                    "href": "../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md",
                    "text": "public-gap review",
                },
            ],
        },
        {
            "kind": "uncertain",
            "label": "Seed diversity",
            "scope": "exposed F0 · 3 teacher seeds",
            "title": "Averaging teacher seeds helped the teacher pool; endpoint value was tiny",
            "result": "Three teacher seeds scored 0.67898918, 0.67836691 and 0.67969849 (mean 0.67901819, sample SD 0.00066626). Their equal trio scored 0.68090199, +0.00188379 over the mean of singles. Half old student / half trio scored 0.68316743 versus 0.68314181 for the old student alone: +0.00002562.",
            "interpretation": "Seed averaging raised this weaker teacher family, but the measured conditional value over the stronger student was near zero.",
            "uncertainty": "The conditional 95% paired-component interval was [−0.00138485, +0.00139836] (141 exposed components, 800 draws). Three teacher seeds are not three folds; one student pipeline seed and one exposed fold.",
            "links": [
                {
                    "href": "../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md",
                    "text": "teacher trio result and receipts",
                },
                {
                    "href": "../../campaign_2026-10-12/SOTA_SUBMISSION_REFLECTION_20261012_R2.md",
                    "text": "final diversity review",
                },
            ],
        },
        {
            "kind": "uncertain",
            "label": "Conditional blend",
            "scope": "exposed F0 · fixed predictions",
            "title": "The old/new student half blend had a small positive point estimate",
            "result": "Fixed half blend 0.68325495 versus the old student 0.68314181: +0.00011314 on 468 groups / 141 components.",
            "interpretation": "This is not enough evidence to call the newer student a better endpoint or to claim public improvement. Its component value remains a plausible interaction worth distinguishing from recipe and precision differences.",
            "uncertainty": "Paired component-bootstrap 95% CI [−0.00128701, +0.00151656], n=141 components, 800 draws. Old FP16-forward and new FP32-forward; recipes and ancestry differ.",
            "links": [
                {
                    "href": "../../campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/README_R1.md",
                    "text": "fixed student contribution screen",
                }
            ],
        },
        {
            "kind": "neg",
            "label": "Blend result",
            "scope": "exposed F0 · H512 source",
            "title": "The tested 75% student / 25% H512 blend was lower",
            "result": "The fixed mixture scored 0.68230811 versus 0.68314181 for the student, Δ = −0.00083370.",
            "interpretation": "Do not promote this fixed mixture from its current evidence. The result closes this exact weighting and evaluated source pool; it does not rule out every H512 descendant or other H512 role.",
            "uncertainty": "Paired 95% component interval [−0.00228225, +0.00038515], 141 exposed F0 components, 800 draws, one seed per source recipe.",
            "links": [
                {
                    "href": "../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/e529_cp2_fixed_blend_grade_20261012_r2/FIXED_BLEND_GRADE_R2.json",
                    "text": "frozen H512 blend grade",
                },
                {
                    "href": "../../campaign_2026-10-12/SOTA_SUBMISSION_REFLECTION_20261012_R2.md",
                    "text": "interpretation",
                },
            ],
        },
        {
            "kind": "uncertain",
            "label": "Weight fitting",
            "scope": "exposed components · one partition",
            "title": "Learned blend weights did not beat equal averaging out of sample",
            "result": "On calibration, fitted shared weights scored 0.67835353 versus 0.67832145 equal weighting (+0.00003207). On the separate fixed evaluation components they scored 0.76624067 versus 0.76636087 (−0.00012020).",
            "interpretation": "The calibration lift did not carry to the evaluation partition. Equal weighting remains the better-supported default for this three-teacher pool; this is one exposed split, not a global optimum result.",
            "uncertainty": "32 calibration components / 110 groups and 16 evaluation components / 52 groups. No interval was recorded for these two fixed-partition differences.",
            "links": [
                {
                    "href": "../../campaign_2026-10-12/joint_endpoint_optimization_r1/J1_EXECUTION_HANDOFF_R1.md",
                    "text": "frozen weight fit/evaluation",
                }
            ],
        },
        {
            "kind": "neg",
            "label": "Quantization",
            "scope": "10-sequence gate screen",
            "title": "Naive Q4 rounding gave back more quality than Q8",
            "result": "On the same small gate, Q8 differed from FP32 by about 0.00007090 in score; naive Q4 rounding lost a further 0.00180125 relative to Q8.",
            "interpretation": "The tested Q8 route is far better supported for current candidates. The Q4 test used a Q8 backend to evaluate rounded weights; it did not establish a native INT4 implementation or speed gain.",
            "uncertainty": "Small gate only; no CI and no full-fold Q4 score. Quantization-aware fitting and mixed precision were proposed, not established by this test.",
            "links": [
                {
                    "href": "../archive/2026-10-12/pages/06-inference-and-quantization.md",
                    "text": "precision and serving evidence",
                },
                {
                    "href": "../archive/2026-10-12/pages/03-submissions.md",
                    "text": "Q8 package and serving review",
                },
            ],
        },
        {
            "kind": "pos",
            "label": "Inference efficiency",
            "scope": "local CPU · bitwise parity",
            "title": "Sharing an identical frozen backbone reduced callback cost",
            "result": "Shared-backbone inference measured roughly 49–51% lower local callback time with bit-identical outputs. The full-fold native Q8 control scored F0 0.68349444 and F1 0.67002463 (one pipeline per fold).",
            "interpretation": "Exact reuse can buy practical serving headroom without changing predictions. It does not add model quality by itself, and the measured machine was not the competition AMD EPYC environment.",
            "uncertainty": "No final A/B/C endpoint was benchmarked on the target EPYC. The one-seed F0/F1 quality split is substantial and does not establish cross-fold stability.",
            "links": [
                {
                    "href": "../archive/2026-10-12/pages/06-inference-and-quantization.md",
                    "text": "native serving findings",
                },
                {
                    "href": "../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_shared_cleanfold_q8_r2/quality_output_r2/QUALITY_RESULT_R2.json",
                    "text": "full-fold Q8 receipt",
                },
            ],
        },
        {
            "kind": "neg",
            "label": "Public calibration",
            "scope": "retrospective · dependent uploads",
            "title": "The public-score ridge was worse than a recent-five baseline",
            "result": "On 17 chronological, component-purged rows across 9 declared source groups, corrected ridge MAE was 0.00097353; recent-five MAE was 0.00078941. Only 11 of the 38 earlier scored submissions had any local metric, across incomparable protocols.",
            "interpretation": "There is no defensible numerical public forecast for new packages from this regression. It remains useful to show where local/public discrepancies exist, not to convert local deltas into public points.",
            "uncertainty": "Retrospective, exposed rows; score-availability times are unknown and declared source groups do not establish independent ancestry. No new candidate forecast.",
            "links": [
                {
                    "href": "../../campaign_2026-10-12/public_score_regression_r1/ROOT_SYNTHESIS_R1.md",
                    "text": "regression results",
                },
                {
                    "href": "../../campaign_2026-10-12/public_score_regression_r1/PUBLIC_REGRESSION_INDEPENDENT_REVIEW_R1.md",
                    "text": "independent review",
                },
            ],
        },
    ]


def main() -> None:
    experiments = experiment_rows()
    hypotheses = hypothesis_rows()
    public_events = public_rows()
    endpoint_rows = read_csv(ENDPOINTS)
    local_public_csv = (
        ROOT
        / "research/campaign_2026-10-12/public_validation_plot_r2/PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.svg"
    ).read_text(encoding="utf-8")
    runtime_svg = (
        ROOT
        / "research/campaign_2026-10-12/public_runtime_quality_r1/PUBLIC_RUNTIME_VS_SCORE_R1.svg"
    ).read_text(encoding="utf-8")
    svg_pattern = re.compile(r"<svg[\s\S]*?</svg>", re.I)
    local_svg = svg_pattern.search(local_public_csv)
    runtime_figure = svg_pattern.search(runtime_svg)
    if local_svg is None or runtime_figure is None:
        raise ValueError("Could not locate a complete embedded SVG in the frozen plot artifact")
    data = {
        "bestPublic": max(row["score"] for row in public_events if row["score"] is not None),
        "publicEvents": public_events,
        "experiments": experiments,
        "hypotheses": hypotheses,
        "findings": headline_findings(),
        "counts": {
            "scoredPublic": sum(row["score"] is not None for row in public_events),
            "experimentsWithResult": sum(row["hasResult"] for row in experiments),
            "experimentsWithCI": sum(bool(row["ciFields"]) for row in experiments),
            "hypothesesWithEvidence": sum(bool(row["evidenceIds"]) for row in hypotheses),
            "previousVisibleEvents": len(public_events) - 3,
            "previousPortalEventsMissing": 3,
            "knownPortalSubmissions": len(public_events) + 3,
        },
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__DASHBOARD_JSON__", payload)
    html = html.replace("__LOCAL_PUBLIC_SVG__", local_svg.group(0))
    html = html.replace("__RUNTIME_SVG__", runtime_figure.group(0))
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    OUTPUT.write_text(html, encoding="utf-8")
    logging.info(
        f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes): "
        f"{len(public_events)} portal events, {data['counts']['scoredPublic']} scores, "
        f"{len(experiments)} experiments, {len(hypotheses)} hypotheses, "
        f"{data['counts']['experimentsWithCI']} experiments with explicit CI fields; "
        f"{len(endpoint_rows)} endpoint-map rows."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
