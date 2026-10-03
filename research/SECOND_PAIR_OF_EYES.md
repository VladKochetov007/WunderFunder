# Independent research review: start here

Use the root project on `main`. This guide describes the paused October3,2026
checkpoint. No training, new experiment or submission upload is authorized by
this review request. Do not unpause launch guards. The separate untracked
`wunder/` project is outside this review.

## What the competition is about

The repository describes the Wunder Connectome sequence-prediction task:
predict two future-price indicators, `t0` and `t1`, for instrument `i0` from
112 anonymized book/trade/additional features covering instruments `i0` and `i1`.
Sequences contain20,000 rows; process causally in order, reset state between
sequences, and produce predictions from step99. The repository metric is pooled
Global Weighted Pearson, averaged over targets, using the scored-row mask and
clipping targets/predictions to[-2,2]. Economic interpretations of anonymous
columns are hypotheses requiring evidence.

The deployed predictor must run offline with one CPU,16GB RAM,60minutes and a
ZIP<=20MB. Local research training has different limits:12CPU,36GiB RAM, and
roughly5–5.5GB observed VRAM. The practical research objective is to improve
public score with causal, reproducible inference that fits deployment limits.
Best recorded public score is0.6641. Neither0.666 nor0.667 is established.

Read these local specifications first:

| File | Purpose |
| --- | --- |
| `README.md` | Task, feature count, data dimensions, metric, inference interface and limits. |
| `METRIC.md` | Scoring definition; distinguish pooled WP from historical block scores. |
| `docs/data_overview.md` | Columns, sequences and feature schema. |
| `docs/submission_guide.md` | Packaging and execution contract. |
| `docs/rules.md` and `docs/faq.md` | Repository copies of competition rules and clarifications. |
| `utils.py` | Actual callback types and feature ordering. |
| `src/wnn/gmetric.py` | Implemented global evaluator. |

## Read in this order

1. `research/campaign_2026-10-03/continuation/RESUME.md`: current scientific
   conclusions, honest limits, paused work and next questions.
2. `research/lb/registry.json`: reported public outcomes and archive associations.
3. `research/experiments.jsonl` and `research/hypotheses.jsonl`: durable experiments
   and hypothesis history, including negatives. E403–405 record final reviews/wrap.
4. `research/campaign_2026-10-03/continuation/RESEARCH_SYNTHESIS.md`,
   `COHORT1_REVIEW.md`, `COHORT2_REVIEW.md`: controller synthesis of original
   research framings. These are historical intermediate syntheses; use final
   RESUME and later results for terminal status.
5. `research/campaign_2026-10-03/scores/combo_matched_2seed_report.json`,
   `scores/combo_fold23_prereg.json`, `scores/combo_fold23_prereg_report.json`,
   and `continuation/FOLD23_DECISION.json`: strongest local candidate and its
   follow-up. Paths in this item are relative to `research/campaign_2026-10-03/`.
6. Relevant original agent reports, raw small results and independent critics
   listed below. Compare split identities, controls, source versions and plots.

## Today's ten independent research framings

Each directory is under `research/campaign_2026-10-03/eda/`. Start with its
`REPORT.txt` and `STATE.json`; read the JSON results and PNGs in that same subtree.
The complete index enumerates exact filenames, including embedded notebook plots.

| Directory | Research framing / question |
| --- | --- |
| `hft/` | Book/trade identities, spreads, causal sign and scale, economic interpretation. |
| `linear/` | Linear/ridge controls, EMA histories, phase and state calibration. |
| `nonlinear/` | Nonlinear/threshold alternatives and conditional recalibration. |
| `olympiad/` | Algebraic structure, constant-label runs, current-price anchors and clocks. |
| `signal/` | Filtering, temporal clocks and signal representation. |
| `statphys/` | Memory, dwell age, event/row clocks and state dynamics. |
| `fractal/` | Scaling/DFA, stationary-level controls, multiscale representations. |
| `econometrics/` | Clock/hazard models, sample composition, t0 versus t1. |
| `puzzle_a/` | Repeated-state counterexamples and price/target structural laws. |
| `puzzle_b/` | Periodicity, phase and lag-recurrence consistency. |

Optimization proposals: `research/campaign_2026-10-03/opt2/REPORT.txt`,
`PLAN.txt`, `FLAGS.txt`, `STATE.json` and patch files. Their independent review
is in `continuation/opt2_review/`; strong-model falsification of the calibration
lead is in `continuation/reproduction/REPORT.txt`. Proposed patches were not
automatically promoted to the production trainer.

## Later actor/critic rounds today

All directories in this section are under
`research/campaign_2026-10-03/continuation/`.

| Subtree | Purpose |
| --- | --- |
| `round2/` | Objective coefficients, first-passage laws, strong-GRU history usage, specification and streaming controls. Read `SYNTHESIS.md` and each task REPORT/STATE/results. |
| `round3/` | Raw decoder audit, coordinate identifiability, hybrid paths, matched history usage, washout, teacher/objective gates, observable simulator and independent reviews. |
| `round4/` | State forgetting, teacher lineage, capture instruments and corrections, repeated parity tests, combo confirmation review. Preserve v1/v2/v3 failures when reviewing v4. |
| `round5/` | Mechanism preregistration/instruments/reviews, credit controls, observable support; distinguish tested instruments from approved actual analysis. |
| `round6/agents.json` | Final five agent IDs, stopped status and completed versus unfinished work. |
| `round6/capture_measurement_review/REPORT.md` | Independent review of actual v4 warm capture. |
| `round4/gpu_capture_v4_default_2120r/results.json` | Actual synthetic/real warm capture results and provenance. |
| `round6/state_drift_v2/REPORT.md` and `real_probe_2222r/comparison.json` | Corrected instrument and actual state-drift comparison; detectable but subthreshold. |
| `round6/state_drift_result_review/REPORT.txt` and `STATE.json` | Final independent state-result interpretation. |
| `round6/amp_linearity_audit_v2/actual_discrepancy_2222.json` | Actual two-case descriptive component-gradient residuals. |
| `round6/amp_linearity_result_review/REPORT.md` | Final independent AMP result interpretation; no decomposition bound or score benefit. |
| `round6/artbp_credit/PREREG.md`, `REPORT.md`, `STATE.json` | Exhaustive fixed-weight finite-graph gradient oracle, including nonlinear-update counterexamples. |
| `round6/artbp_critic/REPORT.md` | Blind critic; subsequent actor-result cross-review was not started. |
| `round6/state_drift_response/REPORT.md` and `STATE.json` | Paused before implementation or execution; not a result. |
| `round6/common_macrostep_credit/PLAN.md` and `STATE.json` | Partial paired-macrostep design, not executed. |
| `round6/common_macrostep_credit_critic/REPORT.txt` | Partial blind source premortem, not executed. |
| `round6/month_research_options.md`, `state_drift_next_options.md`, `MEMORY_LITERATURE_ADDENDUM.md` | Prospective directions, falsifiers and source-linked literature. |
| `runtime_incidents/` | Diagnosed launch failures and corrections; scientific evidence must keep failed attempts distinct. |

Frozen reports are snapshots. An earlier pending/BLOCK state is not erased by a
later correction. Review exact version bindings and later result reviews. In
particular, `round6/REPORT.md` and `landed_evidence_index.json` predate some final
probes; do not treat them as the final campaign status.

## Earlier research and agents

| Files / subtree | Purpose |
| --- | --- |
| `research/README.md`, `frontier.md`, `tree.md`, `runs/INDEX.md` | Ledger conventions, historical frontier, hypothesis structure, per-run navigation. |
| `research/findings-*.md`, `2026-09-*.md`, `reflection-*.md`, `plan-*.md` | Earlier findings, decisions, retrospectives and failed directions. Exact matches in full index. |
| `research/ideas/`, `research/panel/`, `research/lit/` | Previous independent idea framings, skeptical reviews, mechanism proposals and literature. |
| `research/artifacts/`, `research/visual/`, `research/runs/` | Earlier small result JSONs, figures and run reports. |
| `research/campaign_2026-09-29/PLAN.md`, `HANDOFF_2026-09-30.md`, `contract.json`, `state.json` | Previous campaign setup and handoff. |
| `research/campaign_2026-09-29/independent_eda/`, `independent_ideas/`, `reports/`, `provided_decode/`, `artifacts/` | Earlier agents' investigations and decoding evidence. |
| `research/campaign_2026-09-30/` | September30 investigations, runtime work and supporting results. |
| `research/campaign_2026-10-01/REPORT.md`, `IDEAS.md`, `TEAM_INTAKE.md` | October1 campaign synthesis and agent intake. |
| `research/campaign_2026-10-01/creative/`, `critic/`, `data_eval/`, `packages/`, `plots/` | Independent proposals, critics, evaluations and figures. |
| `research/campaign_2026-10-02/REPORT.md`, `RESEARCH_RESET.md`, `UPLOAD_PLAN.md` | October2 findings, public-score lessons and historical upload plan. |
| `research/campaign_2026-10-02/dtw/`, `similarity/`, `lit/`, `opt/`, `tcn/`, `twoclock/`, `eda_cells/`, `plots/` | Teacher/path objectives, recurrence/similarity, optimization, architecture/clock tests and figures. |
| `research/campaign_2026-10-02/packages/*_report.json` | Local student, teacher, ensemble, architecture and holdout comparisons. |
| `research/campaign_2026-10-02/engine/` | Native inference speed/precision evidence; deployment runtime matters. |
| `research/imported_agent_branches/{online_norm,conv}/` | Exact archived old agent patches, source files and commit provenance; current code evolved separately. |

Implementation to inspect when assessing claims: `scripts/train_rnn.py`,
`scripts/predict_run.py`, `src/wnn/nn.py`, `src/wnn/optim.py`, `src/wnn/features.py`,
`src/wnn/resources.py`, `submission/solution_multi.py`, `scripts/package_submission.py`
and `native/src/`. These are code, so the evidence-format index does not list them.

## Complete filename lists and availability

- `research/REVIEW_FILE_INDEX.md`: every tracked research MD, JSON/JSONL, image,
  PDF, TXT report, notebook and HTML at the index's recorded base commit;
  grouped by parent directory, with links and file roles.
- `research/REVIEW_FILE_INDEX.json`: the same exact paths, sizes, categories and
  roles for scripted traversal or context batching. Role labels are filename
  classifications, not scientific conclusions.
- `continuation/LOCAL_ARTIFACT_INVENTORY.json`: numerical outputs intentionally
  kept local, not Git; paths and sizes. Check availability before replication.
- `continuation/CURRENT_RECOVERY.md` and `RESTORE.md`: additional local recovery
  archives. Private transcripts and duplicate restored trees are not in the index.
- `continuation/SUBMISSION_ARCHIVES.json`: existing candidate identities and
  hashes. Alternatives are exploratory; one reference already scored0.6641.

The `continuation/` paths in the last four items are relative to
`research/campaign_2026-10-03/`.

## Prompt to give the second reviewer

> Review the root Connectome research on main, starting with this guide and
> RESUME.md. Work read-only; do not start experiments, alter frozen evidence,
> unpause queues or upload. Independently reconstruct the task and metric, then
> assess the strongest positive findings and rejected mechanisms against their
> raw small results, split/provenance declarations, plots and critic reports.
> Distinguish public scores, exposed local validation, warm/synthetic diagnostics,
> instrument checks and unexecuted ideas. Identify contradictions, leakage or
> selection risks, missing controls and opportunities that survive prior negatives.
> Return a claim-by-claim verdict with exact file/evidence references, followed by
> prioritized future experiments and explicit falsifiers. Do not infer public
> improvement from weak-control or proxy-only results. Flag unavailable large
> artifacts and ambiguous source versions rather than assuming them resolved.
