from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Review-derived mechanisms: each row retains a distinct scientific question.
# Raw results, not these descriptions, determine the subsequent disposition.
PROPOSALS = [
    (
        "eligibility",
        "Evidence eligibility and archive lineage",
        "Stream archive/model lineage; distinguish fitted, excluded and public comparisons",
        "metadata",
        "astra.md;SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md",
        "cpu",
    ),
    (
        "calibration",
        "Refresh public transfer calibration",
        "Rebuild current registry statistics; grouped-family and chronological sensitivity",
        "measurement",
        "INDEPENDENT_ELEPHANTS_2026-10-05.md;astra.md;freebuff.md",
        "cpu",
    ),
    (
        "headroom",
        "Class-by-age headroom and mixture sensitivity",
        "Stream exact weighted moments on fitted/excluded panels and multiple mixture scenarios",
        "measurement",
        "ELEPHANTS_REVIEW_2026-10-05b.md;INDEPENDENT_ELEPHANTS_2026-10-05.md",
        "cpu",
    ),
    (
        "anchored_screen",
        "Parent-preserving residual fitting",
        "Parent/zero-init/capacity/planted correction before scientific residual fitting",
        "instrumentation",
        "astra.md;SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md;55.md",
        "local_small",
    ),
    (
        "feature_audit",
        "Reproduce completed E415/E418 arithmetic",
        "Audit saved scores and retain both damaged controls and unchanged parent",
        "measurement",
        "ELEPHANTS_REVIEW_2026-10-05b.md;mimo2.md",
        "cpu",
    ),
    (
        "event_semantics",
        "Causal event definitions and a6 identity",
        "Compare OR/AND/bid-only/ask-only clocks, raw code equality and transition alignment",
        "measurement",
        "INDEPENDENT_ELEPHANTS_2026-10-05.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md;freebuff.md",
        "cpu",
    ),
    (
        "activation_mixture",
        "Fixed GELU/LeakyReLU readout transfer",
        "Complete unchanged strong recipe versus fixed 50/50 readout mixture",
        "readout",
        "astra.md;muse.md;sol.md",
        "kaggle_gpu",
    ),
    (
        "candidate_shape",
        "Bounded recurrent candidate transfer",
        "One candidate placement versus slope-matched linear and cubic controls",
        "recurrence",
        "muse.md;SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "duration_fp32",
        "E489 FP32-Muon shape-by-duration",
        "Original six trajectories and 18 fixed snapshots after strict reviewed real-device probe",
        "optimization",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "reanchor",
        "Original-label reanchoring on surviving spans",
        "Fixed .1 original-label contribution versus stopped-bootstrap and bootstrap-off",
        "supervision",
        "astra.md;55.md;SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "metric_objective",
        "Clipped pooled-WP fine-tuning",
        "Identical strong checkpoints: unchanged, original-label WMSE, clipped WP, fixed hybrid",
        "objective",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md;mimo2.md",
        "kaggle_gpu",
    ),
    (
        "normalizer",
        "Distillation denominator controls",
        "Historical per-update mass versus fit-frozen denominator versus shuffled factor",
        "objective",
        "muse.md;sol.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "selection",
        "Label-aware selection weights",
        "Cross-component feature/label selector calibration and weighted residual moments before paired student",
        "supervision",
        "astra.md;55.md;SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md",
        "cpu_then_gpu",
    ),
    (
        "target_balance",
        "Equal-target objective weighting",
        "Measure fixed-batch gradient mass then equal-target versus historical loss",
        "objective",
        "RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "gradients",
        "Main/auxiliary/EMA gradient attribution",
        "Fixed fitting packets with separate gradients, alignment and stopped-span slices",
        "measurement",
        "RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "teacher_crossfit",
        "E442 teacher overlap and cross-fitting",
        "All learned-ancestor exclusions; cross-fitted versus historical teacher targets",
        "supervision",
        "muse.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "teacher_bag",
        "Generation3 and bag-as-teacher",
        "Same causal student with historical teacher versus fixed combo/bag teacher",
        "supervision",
        "muse.md;sol.md",
        "kaggle_gpu",
    ),
    (
        "privileged",
        "Origin-aligned future8 teacher",
        "Equal-width current/future teachers predict original Y_t; cross-fit causal students",
        "privileged_information",
        "astra.md;sol.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "privileged_moments",
        "Privileged conditional-moment teacher",
        "Estimated E[W|G], E[WY|G] versus origin-label teacher and causal control",
        "privileged_information",
        "astra.md",
        "local_small",
    ),
    (
        "direction_aux",
        "Current-target direction auxiliary",
        "New relational/teacher mechanism with and without direction supervision",
        "supervision",
        "RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "regimes",
        "E432 warmup regime heads/intercepts",
        "First99-only routed heads versus parameter-matched universal head",
        "representation",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "clocks",
        "E433 target-specific histories",
        "Shared two-clock versus asymmetric adapters versus swapped assignment",
        "representation",
        "muse.md;sol.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "activity",
        "E434 activity-by-history",
        "Causal rate/variance interactions versus stale/shuffled and additive controls",
        "representation",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "age",
        "E434 event-age-by-history",
        "Event age modulates retention versus stale age and additive age controls",
        "memory",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "i1_innovations",
        "E435 i1 conditional innovations",
        "Lagged i1 residual versus identical i0 lags and shifted-i1 sham",
        "representation",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "transitions",
        "E435 anonymous transition structure",
        "Statistical transitions/products without oracle physical decoding",
        "representation",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "cycle8",
        "E436 cycle8 then EMA",
        "Cycle8 versus equal-horizon row EMA versus cycle7 placebo",
        "memory",
        "muse.md;sol.md",
        "kaggle_gpu",
    ),
    (
        "t1_offset",
        "E437 nonlinear t1 correction",
        "Component-disjoint strong-state offset versus identity and global affine",
        "readout",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "cpu",
    ),
    (
        "volume_inputs",
        "E413 history/current/no-volume",
        "Three matched input arms plus another seed/fold and complete final heads",
        "representation",
        "muse.md;mimo2.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "volume_gate",
        "E427 volume update gate",
        "Aligned/stale/noop recurrent gate after full-shape resource and streaming checks",
        "recurrence",
        "mimo2.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "branch_u",
        "E431 end-to-end branch/U readout",
        "Causal categorical analytic readout versus clean WMSE and capacity control",
        "readout",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "joint_support",
        "Joint support with continuous escape",
        "Train-only joint atoms/residual versus marginals/shuffled support/pure regression",
        "readout",
        "muse.md;sol.md;astra.md",
        "local_small",
    ),
    (
        "sign_amplitude",
        "Conditional sign/amplitude posterior",
        "Complete strong-state distribution head versus equal-capacity unconstrained residual",
        "readout",
        "mimo2.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "diagonal_memory",
        "Learned residual diagonal/LRU memory",
        "Learned decay versus fixed decay versus matched static branch beside strong GRU",
        "memory",
        "astra.md;muse.md",
        "local_small",
    ),
    (
        "spline",
        "Spline residual readout",
        "Train-only fixed knots versus matched hinge/ReLU head and no residual",
        "readout",
        "muse.md",
        "local_small",
    ),
    (
        "event_quiet",
        "Event/quiet level-jump posterior",
        "Zero-init state branch versus unrestricted/no-event/stale alignment controls; quiet updates stay",
        "memory",
        "INDEPENDENT_ELEPHANTS_2026-10-05.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "class3",
        "Class3 difference/long-age specialists",
        "Strong-state residual versus equal-capacity target-rotation/control and second component panel",
        "readout",
        "INDEPENDENT_ELEPHANTS_2026-10-05.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "content_memory",
        "E444 content-addressed event memory",
        "Causal query retrieval versus position/decay memory and stale address",
        "memory",
        "muse.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "delta_memory",
        "Erase/write Delta-rule memory",
        "Event-keyed error-driven writes versus additive diagonal and position-only controls",
        "memory",
        "muse.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "two_block",
        "E444 stable two-block state",
        "Matched state/parameter-budget stable blocks versus ordinary state/content branch",
        "recurrence",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "kaggle_gpu",
    ),
    (
        "cadence",
        "E430 cadence/mass/credit",
        "Fresh chunk1000, accumulated2000-detach1000, connected2000; matched row EMA",
        "optimization",
        "muse.md;sol.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "random_origins",
        "Random-origin prefix views",
        "Fixed/random/discarded-prefix views with matched supervised mass",
        "augmentation",
        "muse.md;sol.md;freebuff.md",
        "kaggle_gpu",
    ),
    (
        "detach_jitter",
        "Truncation-boundary jitter",
        "Fixed origin with jittered detach boundaries versus fixed boundaries",
        "optimization",
        "muse.md;INDEPENDENT_ELEPHANTS_2026-10-05.md",
        "local_small",
    ),
    (
        "thinning",
        "E438 exact thinning/view averaging",
        "Recomputed causal features and exact labels versus retained-mass control",
        "augmentation",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "refresh",
        "E445 causal incoming-state refresh",
        "Refresh versus equal-work repeated-operation sham",
        "optimization",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "state_drift",
        "Fixed-packet state-drift response",
        "Offsets0/8/32/96/256/end; bounded whole/group gradient/update response",
        "measurement",
        "muse.md",
        "cpu",
    ),
    (
        "handoff",
        "E439 late optimizer handoff",
        "Muon to AdamW at75% versus no handoff and module-clipping contrasts",
        "optimization",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "artbp",
        "E440 ARTBP finite-window credit",
        "Exact-gradient oracle then unclipped SGD/momentum versus matched truncation",
        "optimization",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "weight_decay",
        "Weight-decay0.1 modern contrast",
        "Matched modern complete recipe versus unchanged decay, fresh seed/component",
        "regularization",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "ema",
        "Aggressive EMA horizon",
        "Matched work/row clock versus unchanged horizon with separate teacher consistency",
        "regularization",
        "INDEPENDENT_ELEPHANTS_2026-10-05.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "antithetic",
        "Antithetic sequence ordering",
        "Seeded forward/reverse order with identical normalization/tails/eligible rows",
        "optimization",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "tier_aux",
        "Class-tier auxiliary",
        "Modern complete model with and without bounded class-tier CE auxiliary",
        "supervision",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "width",
        "W256/W320/W384 by optimizer",
        "Streamed single-worker production-shape pair, actual exclusions, missing replication",
        "capacity",
        "astra.md;muse.md;sol.md;INDEPENDENT_ELEPHANTS_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "lstm",
        "E441 LSTM diversity partner",
        "Same-cost GRU pair versus fixed LSTM/GRU blend including actual native time",
        "diversity",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "mlp_encoder",
        "MLP encoder diversity",
        "Matched strong pipeline with MLP encoder, singles and fixed blend",
        "diversity",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "two_timescale",
        "Two-timescale student",
        "Modern matched two-timescale representation versus equal-state ordinary student",
        "memory",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "local_small",
    ),
    (
        "sentinel",
        "E443 sentinel neutralization",
        "Safe neutral code plus missingness flag versus unchanged and capacity control",
        "representation",
        "muse.md;RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "kaggle_gpu",
    ),
    (
        "spread_aux",
        "E443 current-spread auxiliary",
        "Causal observed-column auxiliary versus equal-work control, no physical oracle",
        "supervision",
        "muse.md;ELEPHANTS_REVIEW_2026-10-05b.md",
        "kaggle_gpu",
    ),
    (
        "mixstyle",
        "Nuisance-coordinate style augmentation",
        "Identified fit-only nuisance statistic versus identity/sham preserving raw-code equality",
        "augmentation",
        "RESEARCH_DIRECTION_AUDIT_2026-10-05.md",
        "local_small",
    ),
    (
        "native_engine",
        "E429 L2 native engine",
        "Actual patch-build parity and one-CPU D256-pair/W384 callbacks; preserve fallback",
        "deployment",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "cpu",
    ),
    (
        "quantization",
        "INT8/QAT diversity enabler",
        "Exact repaired archive parity, own float reference and actual one-CPU runtime",
        "deployment",
        "astra.md;muse.md;INDEPENDENT_ELEPHANTS_2026-10-05.md",
        "cpu_then_gpu",
    ),
    (
        "seed_bag",
        "Exact seed3/seed4 fixed bag",
        "Actual fixed member archive callback timing, pooled scores and no time governor",
        "diversity",
        "ELEPHANTS_REVIEW_2026-10-05b.md;muse.md",
        "cpu",
    ),
    (
        "weight_soup",
        "E448 compatible tensor soup",
        "Identical tensor lineage: single, fixed prediction blend and weight average",
        "diversity",
        "ELEPHANTS_REVIEW_2026-10-05b.md",
        "cpu",
    ),
    (
        "older_finals",
        "Older finals/all-valid transfer survey",
        "Identity/exposure audit then fixed diversity tests and genuinely excluded recipe retraining",
        "diversity",
        "55.md;freebuff.md;INDEPENDENT_ELEPHANTS_2026-10-05.md",
        "cpu_then_gpu",
    ),
    (
        "tpu_recovery",
        "E463/E465 actual TPU recovery",
        "Startup breadcrumbs, real-device parity, full-shape FP32/BF16 profiles before fits",
        "infrastructure",
        "SECOND_PASS_PRIORITY_TRIAGE_2026-10-05.md;muse.md",
        "kaggle_tpu",
    ),
]


def append_unique(path: Path, records: list[dict[str, object]]) -> None:
    """Append new records without duplicating a completed initialization."""
    existing = {
        row.get("continuation_key"): row
        for line in path.read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }
    pending = [row for row in records if row["continuation_key"] not in existing]
    with path.open("a") as output:
        for row in pending:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Materialize the review coverage and register its unchanged hypotheses."""
    target = HERE / "coverage.json"
    if target.exists():
        raise FileExistsError("coverage is already initialized; preserve existing state")
    timestamp = datetime.now(UTC).isoformat()
    ledger = ROOT / "research/hypotheses.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    first_id = max(int(row["id"].split("-")[-1]) for row in rows) + 1
    coverage: list[dict[str, object]] = []
    hypotheses: list[dict[str, object]] = []
    for offset, (key, title, test, family, sources, placement) in enumerate(PROPOSALS):
        hypothesis_id = f"H-{first_id + offset:03d}"
        references = [f"reflect_05_day/{source}" for source in sources.split(";")]
        source_hashes = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in references
        }
        coverage.append(
            {
                "key": key,
                "hypothesis_id": hypothesis_id,
                "title": title,
                "review_refs": references,
                "review_sha256": source_hashes,
                "mechanism_family": family,
                "cheap_test": test,
                "placement": placement,
                "status": "registered",
                "experiment_ids": [],
                "artifacts": [],
                "next_action": "source/design and independent review before admission",
            }
        )
        hypotheses.append(
            {
                "id": hypothesis_id,
                "continuation_key": key,
                "parent_ids": [],
                "mechanism_family": family,
                "statement": title,
                "representation_change": title,
                "causal_mechanism": test,
                "assumptions": ["causal available inputs", "competent matched controls"],
                "predicted_observable": "paired matched-control effect or decisive diagnostic",
                "predicted_direction_or_delta": "unknown magnitude; positive effect if mechanism helps",
                "cheap_test": test,
                "falsifier": "invalid invariants or reproducible absence of the predicted effect in this tested protocol",
                "known_failure_modes": [
                    "exposed-panel selection",
                    "degraded control",
                    "resource-censored execution",
                ],
                "estimated_cost": {"placement": placement, "admission": "requires measured probe"},
                "proposed_by": "supplied October5 reviews; controller index",
                "committed_at": timestamp,
                "status": "proposed",
                "review_refs": references,
            }
        )
    append_unique(ledger, hypotheses)
    target.write_text(
        json.dumps({"version": 1, "created_at": timestamp, "items": coverage}, indent=2) + "\n"
    )
    header = "# Review coverage\n\nEvery distinct mechanism stays visible; registered is not completed.\n\n"
    table = (
        "| Hypothesis | Mechanism | Placement | Status | Minimum test |\n|---|---|---|---|---|\n"
    )
    table += "".join(
        f"| {row['hypothesis_id']} | {row['title']} | {row['placement']} | {row['status']} | {row['cheap_test']} |\n"
        for row in coverage
    )
    (HERE / "COVERAGE.md").write_text(header + table)
    evidence = {
        "id": "E-490",
        "date": timestamp[:10],
        "kind": "continuation_contract_and_review_coverage",
        "status": "registered_no_scientific_fits",
        "scientific_fits_completed": 0,
        "continuation_key": "night_coverage_initialization",
        "hypotheses_registered": len(coverage),
        "contract": str((HERE / "CONTRACT.md").relative_to(ROOT)),
        "coverage": str(target.relative_to(ROOT)),
        "public_anchor": 0.666,
        "target_public_score": 0.7,
        "local_start_before": "2026-10-06T07:00:00Z",
        "notes": "User approved implementation; metadata registration is not an experiment result.",
    }
    append_unique(ROOT / "research/experiments.jsonl", [evidence])
    print(
        json.dumps(
            {
                "registered": len(coverage),
                "first_id": first_id,
                "last_id": first_id + len(coverage) - 1,
            }
        )
    )


if __name__ == "__main__":
    main()
