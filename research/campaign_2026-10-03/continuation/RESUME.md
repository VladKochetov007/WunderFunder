# Start here: paused research checkpoint

The user stopped research on 2026-10-03 at22:34 Kyiv and requested consolidation
on `main`. All five Codex agents are stopped. No experiment, training run, or
scientific analysis is running. Queues and the minute watcher were stopped;
`research/campaign_2026-09-29/STOP_NEW_JOBS` and
`research/campaign_2026-10-01/QUEUE_PAUSE` intentionally block new launches.
Do not remove those markers or start a run without explicit resumption.

## What is saved

- The ten original research perspectives, plots, runnable probes and results are
  in `../eda/`; optimization proposals and unapplied patches are in `../opt2/`.
- `round2/` through `round6/` preserve successive actor/critic versions, hypotheses,
  preregistrations, failed controls, source bindings, results and final reviews.
- Git preserves code, small result JSONs, plots, notebooks and evidence manifests.
  Large numerical arrays and state packets remain local, with recovery archives
  and an exclusion inventory. No Git LFS or DVC is required. Copy those local
  dependencies separately when moving the research to another machine.
- `research/experiments.jsonl`, `research/frontier.md`, the training scripts and
  resource/admission wrappers provide the shared history and implementation.
- Original older agent ports are retained under `research/imported_agent_branches/`
  with commit identities and exact patches/source snapshots. They are historical
  implementations, not permission to replace the current native predictor.
- `CURRENT_RECOVERY.md` and `RESTORE.md` describe additional actual paused restore
  checks. Private Claude transcripts/context and duplicate recovery archives stay
  local. Full competition datasets and external training/prediction caches remain
  dependencies; this is not a standalone mature-training resume.

## Strongest results and their limits

The best recorded public score is **0.6641**, the distilled D256 single model
listed in `research/lb/registry.json`. The Muon/chunk1000/stopped-bootstrap recipe
has replicated local gains near0.0018, including fixed fold2/3 confirmation.
Those folds were previously explored. Full-data training and a verified submission
of that recipe remain unfinished. A public forecast near0.666 is an optimistic
straight-line extrapolation, not a calibrated expectation;0.667 is unproven.

The v4 default capture passed exact recorded warm normal/repeat/original/forced-
skip comparisons and independent review. The real panel starts after four
completed updates with no AMP skips, not a recovered mature training checkpoint.

The real state-drift binary probe finished in12.56s with2.00GB observed VRAM.
Independent review supports **DETECTABLE_SUBTHRESHOLD**: hidden relative change
0.236%, prediction RMS1.63e-5, gradient relative change0.0118%, actual update
relative change0.871%. No materiality group passed. Only incoming detached student
hidden state changed; this does not establish predictive benefit or a burn-in length.

The independently reviewed descriptive AMP audit measured separate component-
selfgradient sum versus whole preclip-gradient residuals: real relativeL2
0.00044967, synthetic0.00019014, using8MiB conversion workspace. No scientific
decomposition bound, conflict attribution, optimizer-component attribution or
public gain follows from these different-operation residuals.

The tiny ARTBP-inspired fixed-weight finite-graph oracle exhaustively enumerated
seven cases, at most1024 masks, and checks unbiased compensated gradients.
Clipping and Adam-like transforms break expected-update equivalence. Its blind
critic is saved, but actor cross-review was not started. This is not a full-paper
implementation or an approved real-training method.

## Pending work, saved without execution

Existing full-data submission archives and their byte identities are listed in
`SUBMISSION_ARCHIVES.json`; copies are local under
`artifacts/submission_quota_2026-10-03/`. Upload ZIPs individually. Equivalent
tar.gz payloads are also available. No new training or inference was run to
produce them. The reference is already scored; alternatives are exploratory.

1. `round6/state_drift_response/`: fixed-lag and whole-gradient/update summaries
   were not implemented or executed. Its STATE lists every unfinished item.
2. `round6/common_macrostep_credit/` and `_critic/`: partial two-chunk design and
   source premortem only; no new tests, forwards or GPU work ran.
3. `round6/artbp_credit/` versus `artbp_critic/`: independently cross-review the
   already saved finite-graph evidence before considering a training proposal.
4. `round6/month_research_options.md` and `state_drift_next_options.md` contain
   prospective falsifiers. Keep their original thresholds and negative evidence.

## Conditions for a later resumption

Read the relevant frozen STATE/report/preregistration before using an artifact.
Receipts bind exact historical source/runtime hashes and an absolute2026-10-03
deadline; they are evidence, not reusable launch authorization. A later run needs
a new dated reviewed contract. Preserve source versions when making corrections.

Keep root<=12CPU,36GiB RAM/swap0; GPU family8CPU/24GiB; four CPU slots1CPU/2GiB.
GPU allocator4.8decimalGB, observed device ceiling5.5decimalGB. Revalidate those
limits, dependencies and the watcher before unpausing the queue. Diagnose failures
before retries and preserve each failed job. Two corrected launch incidents
(false timeout predicate and missing executable permission) are retained under
`runtime_incidents/`; neither caused a new OOM.
