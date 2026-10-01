# Resume audit, 2026-09-30

Scheduler repairs and five durable CPU stop probes passed. Root may review the
two scheduler scripts and intentionally remove `STOP_NEW_JOBS` to launch a new
controls queue job. This reviewer launched no GPU job and no controls queue.
The original marker bytes are restored. Historical queue/job records and logs
are unchanged (56 preexisting files checked bytewise).

## Stop behavior

- SIGTERM/SIGINT persist the launch barrier, stop the child process group, stop
  the named scope for either CPU or GPU jobs, and finalize `interrupted` records.
- Cleanup also checks the group and scope after an apparently successful leader
  exit, so surviving descendants cannot silently escape the job.
- The service gives the monitor 15 seconds for cleanup and applies explicit
  `KillMode=control-group`. The GPU cap and `remaining - 15` allowance are unchanged.
- Launchers check the barrier immediately before service creation; service startup
  checks it again. GPU lock acquisition now polls the barrier while waiting.
- The queue checks the marker before every launch and in both record-appearance
  and completion polling. Direct queue signals also create the marker. A queue
  barrier exits scheduling without terminating separately launched training jobs.
- `--threads` (1..12, default 12) caps service CPU quota and worker thread pools.
  The probes used one thread and one-core scopes in `wnn-campaign.slice`.

`stop_probe_results.json` records fresh unique durable jobs for actual service
SIGTERM, monitor SIGINT, a scoped leader exiting with a SIGTERM-ignoring child,
a queue wait with no record, and a queue wait with a running record. All passed;
the two signal cases also proved followup launches were rejected. Scope and
process checks found no surviving executable workers. Some killed descendants
may briefly remain kernel zombies; the assertions explicitly permit that state.
Signals completed accounting/cleanup in about 3.51 seconds. Queue polling noticed
the marker within its existing 2/5-second polling cadence. The CPU/GPU branches
share cleanup code, but live GPU signal cleanup was intentionally not exercised.

Production files changed only in `scripts/campaign_job.py` and
`scripts/campaign_controls_queue.py`; `ruff format` and `ruff check` passed.
The executable regression probe is `stop_probe.py`. Its driver is only safe while
the root coordinator keeps GPU scheduling paused, since it temporarily removes
and restores the campaign barrier. It must not run concurrently with a queue.

## Completed controls and artifact provenance

`provenance_audit.json` verifies every listed checkpoint byte against its manifest
for nine completed controls: fold-0/1 propensity, four parent seed/fold controls,
and three completed fine-tunes. All exit 0. All split hashes, fixed checkpoint
policies, and `no_eval_during_fit` flags match. Parent propensity checkpoint
bindings match. Both fine-tune stages match their exact input checkpoint bytes;
both stages use FT seed 3 even for the seed-4 parent. FT stage epochs are fixed 0;
parent/propensity epochs are fixed 3.

Fold-1 has 10,533 train and 1,416 valid fitting groups, with 457 evaluation groups.
Fold-0 has 10,535 train and 1,405 valid fitting groups, with 468 evaluation groups.
This audit checks recorded manifests and checkpoint bindings; the previous
independent adjacency-traversal split audit remains the evidence for component
exclusion, and these clean recipe analogs are not the public incumbent weights.

The fold-1 seed-3 FT valid prediction array matches its handoff hash:
`03810002ea2296ca63ec9f67b7771d71180bb09499afb50254ae1bac2554e6f6`.
The fold-1 seed-4 parent matches all checkpoint manifest entries, including final
`95682ebffcc01970ffd72ea0a369852def3cdd8d1e1e7f046363a12365274faf`.
Its FT remains absent, so the frozen queue should resume at that step.

## Metric and research status

All six evaluator/documentation files match the frozen evaluator manifest.
The official metric is `global_wp_clip2_v1`: select `need_prediction AND
is_scored`, clip targets/predictions to [-2,2], weight by absolute clipped target,
pool rows, and average the two target correlations. Confirmation uses centered
component moments and component bootstrap. The earlier failed nearconstant
review artifacts are historical; the latest independent release review recorded
nine focused checks passing and the frozen centered evaluator hashes now match.
This reviewer did not edit or rerun the evaluator.

The incumbent ZIP matches
`7990bafd721b09bde2670bc9281fdb0913456a7e2caf046d894fed31faae10f7`.
Its public 0.6622 association remains user-reported. Fold-0 bilinear screen is
negative: average 0.67781291 versus bilinear 0.67212585, paired delta -0.00568705,
95% component-bootstrap interval [-0.01136381,-0.00102736]. No promotion is
supported. Completed fold-1 FT predictions are not a promoted candidate.
Neither existing public-calibration scenario supports expected public >=0.6660.
Verification remains tier C for public transport, with deterministic checks for
scheduler behavior and artifact identity. Keep the later grader independent.

Root reported that runtime slice properties were lost during reboot and restored
them before these probes. A live read confirmed CPU quota 12 cores, MemoryMax
38,654,705,664 bytes (36 GiB), and MemorySwapMax 0. New job services also set an
explicit 36 GiB hard memory limit and scopes retain `scripts/limited` limits.
GPU charged budget stays at 5,153.59 / 28,800 seconds because these probes are CPU.

## Subsequent independent work after root resumption

New data-derived cards are reviewed in `dataderived_cards_critique.md`:
D1 conditionally proceeds to CPU proof with modulo128 aliases and joint
write/decay interpretation disclosed; D2 is deferred behind label-only screen
with branch/class loss ambiguity resolved before implementation; D3 is blocked.
`inverse_clock_prereg_review_v2.md` and independent raw replay record its frozen
TRAIN effective-cell failure3.73283<4 despite improved heldout agreement.
Actual replay peak1.051GiB was below2GiB, but individual hard2GiB was not
enforced; outer/shared hard36GiB was respected. No replay-only rerun or
post-result threshold change was performed.

`spread_aux_spec_review.md`, `spread_aux_independent_checks_v1.json` and
`spread_source_commitment_review.json` record conditional matched label-only
screen approval: independent loss/gradient oracle, eleven provenance/cache
failure checks,512 selection reconstruction, all cache hashes/ranges, six raw
feature cache replays and nineteen source commitments passed. Scope properties
proved hard2GiB, one core and zero swap; actual peak1.674GiB. No full raw512-group
target-vote reconstruction is claimed. Exact FT-stage/source/score designs and
successful-update parity remain required before evaluation. Scorer unchanged,
trainer patch unapplied, no new candidate score or GPU job.

The preceding stop/budget statements describe the initial audit snapshot. Root
subsequently archived/removed the restored stop marker and resumed its GPU work;
this reviewer has not mutated the marker since the five stop probes. Controls
child launches now explicitly use `--threads10`, leaving the remaining aggregate
slice capacity for the notebook/data probes. The 16 frozen control step recipes,
timeouts and expected artifacts match after plan-schema normalization, recorded
in `scheduler_source_manifest.json`. Two initial plan comparison fixture mistakes
(wrapper/list and field-name schema) are recorded there; neither changed a recipe.

Further independent results are indexed here:

- `architecture_critique.md`: A/B/C specification concerns and matched known-code
  spread-label continuation conditions; no architecture authoring.
- `pooled_independent_replay.json`: all1,026 pooled code/class/spread entries,
  training vote counts and all32 test-group counts reproduced from raw TRAIN.
- `campaign_score.py`, `campaign_score_freeze.json`, `scorer_checks_v3.json`:
  reusable metric grader frozen after synthetic arithmetic/provenance checks,
  with zero-copy stored NPZ targets and one-sequence prediction handling.
- `campaign_score_README.md`, `reference_fourfold_design.json`,
  `build_reference_spec.py`: reference-only fourfold design and durable scoring
  commands for root after dependencies finish. F0/F1 exact parent/both-stageFT/
  propensity/1873-row sidecar bindings passed without prediction scoring.
- `structured_state_grade_v1.json`: preserved long-horizon arithmetic failure;
  `structured_state_grade_v2.json` and `structured_state_review.md`: corrected
  actor source passes all8 independent CPU cases, unchanged numeric tolerances.
- `quote_endpoint_independent_replay.json`, `quote_endpoint_review.md`: raw
  feature-only endpoint counts independently reproduced, origins and three
  class1 quiet exceptions separated; physical movement/time remain unidentified.
- `inverse_clock_critique.md`: conditions for a bounded nondegenerate feature-only
  inverse-clock probe, including observed-support scale, per-bin holdout checks
  and an explicit limitation of row-age agreement.

All reviewer runs were CPU-only with at most one thread and 2GiB scopes, and
no GPU initialization for the structured-state grade. Competition scorer and
dependencies still match the frozen bytes. No candidate prediction scoring,
GPU model fitting, native port or public-score promotion was performed here.
