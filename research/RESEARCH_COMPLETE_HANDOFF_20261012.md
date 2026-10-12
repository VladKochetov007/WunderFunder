# Challenge 2 research handoff — 2026-10-12

**Archive state: pre-drain draft.** This handoff is based on Main snapshot
`3b8c5fb296371d693d5703b02f3e37fb67eaabad`, including the latest stop
and memory-margin instructions and the documentation/evidence cleanup commits.
It intentionally leaves the two already admitted logical runs pending. Root
must add their terminal outcomes and artifact inventory, refresh current-state,
index, and generated-tree snapshots, and mark the archive final only after
those runs are reconciled. This document does not admit, recommend, or launch
work.

The governing record is
[STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json](campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json).
New research admission is closed. The only work permitted by that record is
draining (a) the already admitted clean-F0 seed-4 student continuation through
its fixed CP5 export and official grade and (b) already admitted Kaggle R7
terminal collection/grading. A scheduled status receipt says R7 is provider
RUNNING, but its worker phase, heartbeat, fitting, export, and outputs had not
been observed at this snapshot. Provider status is not evidence that training
started. See
[R7_PROVIDER_RUNNING_R1.json](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/monitor_20261012_1358_r7_running_r1/R7_PROVIDER_RUNNING_R1.json).
Final drain must distinguish successful, partial, and failed terminal
outcomes and preserve originals. Do not infer missing scores.

The archive snapshot has 479 hypothesis-ledger lines, 795 experiment-ledger
lines, 158 configured research-tree nodes across 11 families, and 133 journal
events. The ledgers are append-only and can contain repeated IDs for revisions,
retries, or additional events; line count is not unique-experiment count. The
generated navigation snapshot is older than these source ledgers and must be
rebuilt after the final drain. This draft adds one archive-only event to the
working-tree journal; it is not a scientific result. See
[the Main status inventory](campaign_2026-10-12/research_archive_inventory_r1/MAIN_STATUS_INVENTORY_R1.json),
SHA-256 `7395df8a06ff7a53767301d642e7012806516d34d226b953a33e6198e0a820de`.
It is a point-in-time, metadata-only inventory from the earlier archive
preparation snapshot at `62378c04`; it is not the post-cleanup Main
inventory. It records status/path counts and file-size classes; it omits
credential-like, provider-raw, large-JSON path details and all file contents.
It is an aid for root's explicit-file cleanup, not authority to stage or
delete. The cleanup commit b607ca0e tracked useful documentation/evidence
files and removed fourteen weight-bearing ZIPs from Git tracking without
deleting their local Main copies or rewriting history. The current source
snapshot also includes later worker/stdout and skill-link updates. Main
status has since changed; refresh inventory only as part of root's final drain.

## Research objective and decision boundary

The active goal remains a valid deterministic Challenge 2 submission that can
reach top-1 and public score above 0.700. Those objectives have not been met.
The strongest available public observation in this archive is the user-reported
0.6671 for candidate C. No numerical public forecast is supported. Existing
local folds, public rows, and submission packages have been repeatedly inspected
and are exposed development evidence. None is a fresh holdout.

Choose complete endpoints using source-qualified local evaluation across
folds and distinct fitting seeds, with the exact served package checked for
metric, precision, determinism, size, and runtime. Public feedback confirms
transfer; it does not replace local comparison, and a local delta is never
added to a public score. A component with lower standalone quality can still
contribute conditionally to a fixed mixture, but that contribution must be
measured on the declared evaluation units. A negative result closes only the
exact tested recipe or combination at that evidence level.

Project-wide binding rules and rationale are in
[the research objective](../specs/research-objective.md),
[research methods](RESEARCH_METHODS.md),
[ADR0031 cross-fold top-tier release](../specs/adr/0031-cross-fold-top-tier-submission-release.md),
[ADR0030 local-evidence selection](../specs/adr/0030-local-evidence-selects-submissions.md),
[ADR0021 producer/consumer validation](../specs/adr/0021-validate-consumers-against-producer-artifacts.md),
[ADR0023 faithful recipe reconstruction](../specs/adr/0023-faithful-method-reconstruction-before-variants.md),
and [current state](../specs/current-state.md). Teammate methods may be read
and independently implemented; teammate weights and copied source code are
not part of our submission lineage.

## Data, metric, and validation

The task is causal sequential prediction over supplied time-series sequences.
Predictions must be deterministic and reset state between sequences. The
official implementation is
[src/wnn/gmetric.py](../src/wnn/gmetric.py); the metric identifier is
global_wp_clip2_v1. It applies the fixed score/prediction mask, casts
predictions and labels through float32, clips each target to ±2, weights each
selected row by the absolute clipped true label, computes weighted Pearson
correlation pooled over selected rows separately for both targets, and averages
the two target correlations. Warm-up rows are excluded by the official mask
(99 rows for the scored sequence setup). Training-target clipping and teacher
mixing are separate recipe choices; they never change official scoring.

Groups are identified by source as well as row-group identity and connected
components preserve related sequences. Splits and bootstraps operate on whole
groups/components, not rows. This prevents within-sequence history or related
group leakage, but repeatedly inspected F0/F1/F2/F3 panels remain exposed.
Bootstrap intervals on fixed predictions measure evaluation-component
sampling only; they exclude fitting-seed variation, recipe variation, endpoint
selection exposure, and public-transfer uncertainty.

The clean-F0 student fit used 10,535 train-source groups and 1,405 valid-source
fit groups, with 468 valid-source evaluation groups excluded from fitting. The
separate prediction export covers all 1,873 valid-source groups, while the
official F0 score uses its fixed 468-group subset. F0 score receipts refer to
141 connected components. F1 comparison receipts use 457 groups and 149
components. Full sample membership, fingerprints, prediction hashes, and
official-scoring commands are in the endpoint-specific manifests linked below.
The final all-data H390/Z endpoints used validation rows for training or
head-fitting and therefore cannot be assigned held-out quality from those
all-data weights.

## Best current packages and serving evidence

The create-only package records are in
[submissions/10-12-best-r1](../submissions/10-12-best-r1/README_R1.md) and
[submissions/10-12-best-r2](../submissions/10-12-best-r2/README_R1.md).
Candidate order B, A, C reflects runtime-risk preference, not a local quality
ranking. Public observations below were transcribed by the user from the portal
and were not independently browser-verified.

| Candidate | Fixed endpoint | ZIP bytes | ZIP SHA-256 | User-reported portal result |
|---|---|---:|---|---|
| B | 0.5 H390 s3 + 0.5 exact merged Z4 | 9,053,484 | `486a34c46989361428aa2e34f1c43dd246e2b7293f6dfcb5dc5031c989420440` | 0.6668, 25:56, 634 MB, portal `09853lyw` |
| A | 0.5 H390 s3 + 0.5 H390 s4 | 8,556,250 | `8834774c58ff31a77547fc8ec59988aef61e1bd1e7198c4c52d1d3835c43e6e5` | 0.6656, 24:31, 632 MB, portal `0rg9ujj5` |
| C | 0.25 H390 s3 + 0.25 H390 s4 + 0.5 exact merged Z4 | 12,680,868 | `6013d20a7ff26ce6ef46821f80ed01da848ef4da565e2cd91656417ddd5186ea` | 0.6671, 36:36, 642 MB, portal `1ya21vpd` |

Each ZIP is under 20 MB, has root-level solution.py, and contains the
precision-aware native Q8VNNI scorer and dispatched library tiers. Exact ZIP
extraction tests ran with CPython 3.11.16 and Q8VNNI mode. Each callback
profile covered ten unchanged complete 20,000-row validation sequences, two
reset-separated passes, 400,000 callback calls, None during warm-up, two
finite values on required rows, and bitwise-identical reset/replay outputs.
Package bindings, ABI checks, prediction receipts and independent review are
linked from each package README and
[the independent package review](../artifacts/submission_review_10_12_best_r1/SUBMISSION_PACKAGE_REVIEW_R1.json).

The H390 source members are
vt_all_h390_ds_homogeneous_s3_r3@ep5 and
vt_all_h390_ds_homogeneous_s4_r3@ep5, with distinct seeds and checkpoint
hashes. Their lineage records since_weight=32, since_floor=0.25, and
10,606 train plus 1,873 validation sequences in fit membership; they are
all-data endpoints. Their Muon parent lineage differs from the strong F1
student reference. The Z4 member is the unchanged merged endpoint
bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1 paired
with z4_seed99_student_hft_alldata_v2@ep0, not a bare Z4 trunk. Its NPZ
member hash is preserved in the package receipts.

Local callback timing came from an Intel i9-11900H and is capacity evidence,
not an AMD EPYC portal guarantee. Initial A/B/C callback means were 44.97,
49.82, and 83.33 microseconds per row; concurrent CPU activity made those
unpaired observations noisy. A later sequential reference-C-reference-C
probe measured C at about 1.44 times reference callback cost, with substantial
pass variation. Portal elapsed times are separately observed end-to-end
package runs. Neither local timing nor the three public points establish a
linear runtime-quality law.

Two further native Q8VNNI packages are technically reviewed and runnable, but
their quality is unmeasured:

| Candidate | Fixed mixture | ZIP bytes | ZIP SHA-256 | Local callback |
|---|---|---:|---|---:|
| D | 0.25 H390 s3 + 0.50 Z4 + 0.25 merged Z5 | 13,178,260 | `bf9f48dbb05668181038d28386a076774d41146efb7c4f1680fcd724a303df09` | 113.60 µs/row |
| E | 0.1875 H390 s3 + 0.1875 H390 s4 + 0.375 Z4 + 0.25 merged Z5 (0.75*C + 0.25*Z5) | 16,805,643 | `2aed2137c347344146e3481c83388d64bd5060b1c13bb66554b0bf9cc0027024` | 152.62 µs/row |

The source-qualified Z5 pair is
bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2@ep1 paired
with z5_seed9_student_hft_alldata_v2@ep0. Its head fitting used 1,505
supplied validation groups. A targeted search found no matched clean-CV grade
for that Z5 endpoint. Its older i16 ZIP has a user-reported 0.6664 result, but
this does not establish Q8 quality because served predictions change with
quantization. The exact Q8 package technical review covers package/source/
runtime only; no official metric has been computed for D or E. E's paired
local capacity heuristic is around 50.9 minutes when scaled from C's portal
duration; this is not a measured portal runtime and is not evidence E will
complete under the 60-minute limit. See the
[D/E package review](campaign_2026-10-12/independent_reviews/submissions_10-12-best-r2/D_E_TECHNICAL_REVIEW_R1.json)
and [paired C/E timing receipt](campaign_2026-10-12/z5_blend_runtime_pair_r1/README_R1.md).

The exact native full-fold scorer is a separate, previously exposed Q8 result:
F0 0.6834944365, F1 0.6700246323, one pipeline seed per fold. It processed
about 9.36 million and 9.14 million rows, with measured callback totals of
201.729 and 198.042 seconds and a 512.34-second worker wall time including
decode and scoring. It has local bit-parity and source/membership receipts,
but does not establish AMD target runtime or the quality of A/B/C/D/E. The
terminal record is
[QUALITY_RESULT_R2.json](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_shared_cleanfold_q8_r2/quality_output_r2/QUALITY_RESULT_R2.json).

An earlier Q4 conversion lost 0.00180125 against Q8 on its ten-sequence
screen; Q8 differed from FP32 by 0.00007090 on those same inputs. This is a
screen, not full validation. No useful native INT4 package was established.
Keep Q4/mixed precision/QAT open only for a changed method with measured
quality, size, and runtime; the tested naive Q4 conversion does not justify
defaulting to Q4.

## Model families, training, and source-qualified findings

The core family is recurrent GRU sequence modeling with independently trained
teachers and students. Training and distillation use supplied allowed data;
the all-data final is distinct from a held-out evaluation sibling. Teacher
targets use the declared clipped mixture
(1 − alpha) * clip(y, −2, 2) + alpha * clip(teacher, −2, 2) where specified,
while primary MSE continues to use clipped true-label magnitude weights and
the existing auxiliary objectives. Alpha 0.9 has a favorable result in the
tested one-seed ladder; it is an exploitation setting, not a universally
established optimum. A strict label-free objective was not run.

The clean-T3 CP5 control is the most recent completed local student grade in
this snapshot. The independent review reports 0.6811553603333363 for seed 3,
fixed epoch 5, on 468 exposed F0 groups / 141 components. The source-qualified
fit used 10,535 train groups plus 1,405 valid fit groups, excluding 468 F0
evaluation groups. It starts from own checkpoint
c20261003_combo_s3_f0@ep3; its three own H384 Muon teachers are seeds 3/4/5
at epoch 2. The full-valid FP32 prediction export is finite and hash-bound.
The actual homogeneous cache target is Tstored = float32(0.8125 *
float32_mean(three teacher predictions)), without cache clipping. Training
then uses 0.1*clip(y,−2,2) + 0.9*clip(Tstored,−2,2). Therefore the nominal
pre-clipping contribution is 0.24375 per teacher; provenance rows labelled
0.1 are source-parent metadata, not the numeric mixture weights. Clipping
makes the final operation nonlinear.

The closest old Z4 student score is 0.6831418134, from a different
teacher/initializer lineage, recipe and FP16 forward inference, versus clean
T3 FP32. This endpoint gap cannot isolate target family, precision, or
architecture. The full static review, hashes, terminal, teacher cache, and
grade links are in
[the CP5 grade review](campaign_2026-10-12/clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md)
and its JSON receipt.

Clean T3 teacher evidence on the same exposed F0 panel:

| Endpoint | F0 WP | Scope |
|---|---:|---|
| H384 Muon teachers, seeds 3/4/5 | 0.67898918 / 0.67836691 / 0.67969849 | three fit seeds, same exposed F0 panel |
| Teacher-seed mean | 0.67901819 | sample SD 0.00066626 |
| Fixed equal teacher trio | 0.68090199 | +0.00188379 vs seed mean |
| Old student + teacher trio, 50:50 | 0.68316743 | +0.00002562 vs old student; CI [−0.00138485, +0.00139836] |

These records support that this three-seed teacher trio is useful for this
fixed F0 target and mixture; they do not establish that any weaker teacher
family is useful in every student or fold. See
[F0_EQUAL_TEACHER_TRIO_GRADE_R1.json](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json).

The preregistered fixed half screen combined the new clean-T3 CP5 student
with the old Z4 student and compared it only against that old student. Blend
WP was 0.6832549498 versus 0.6831418134, delta +0.0001131364, with
whole-component 95% interval [−0.00128701, +0.00151656] over 141 components
(800 bootstrap replicates). This is a small uncertain conditional result on
exposed F0. Both arrays are float32 storage, but new inference was FP32 and
old inference FP16; ancestry and recipe also differ. There is no recipe-winner,
public-transfer, or cross-fold claim. See
[the preregistration and result handoff](campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/README_R1.md).

On F1, old/new student half scored 0.67182 versus new student 0.67103,
one pipeline seed and 457 exposed groups/149 components, with a broad interval
crossing zero. A completed F1 HFT comparison gave 0.67009 for the half blend
versus 0.67038 for the student; an extra-head blend scored 0.67188 versus
0.67182, a tiny uncertain difference. The direction changes across folds.
These are not independent lineage replication and not evidence to select from
F1 alone. See [F1 fixed half](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_new_f1_fixed_half_20261012_r1/RESULTS_R1.md)
and [F1 HFT comparison](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_hft_export_review_20261012_r1/RESULTS_R1.md).

The H382 cross-fold epoch work shows a fold-dependent four- versus three-epoch
effect: approximately +0.00078 in F0, about −0.0007 in F2, and F3 0.70865 at
epoch 3 versus 0.70657 at epoch 4; F1 comparisons also vary by fixed mixture.
Each fold/recipe has one seed except the separately reported F1 seed set, and
epoch changes also change total OneCycle schedule horizon. Results are from
exposed panels and do not imply a universal best epoch. Primary evidence is
[H382 cross-fold grade evidence](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H382_CROSSFOLD_GRADE_EVIDENCE_R1.json).
The H384 F1 three-seed teacher mean was 0.67068046, equal trio 0.67254162,
and half H256 plus H384 trio 0.67264009; the incremental +0.00009847 is
exposed fixed-prediction evidence, not a new training seed or fresh fold.
Related source records are in
[IDEA_LEDGER_20261011.md](IDEA_LEDGER_20261011.md) and H384 receipts in the
October-11 operations continuation.

The large-teacher/native Q8 endpoint, clean-T3 pipeline, all-data H390 package
members, and historical old Z4/Z5 students are different source families.
Their scores should not be pooled without matched ancestry and precision.
Specific fit membership, epoch, parent checkpoint, teacher and initializer
provenance are in linked receipts rather than reconstructed from names.

## Fixed endpoint-weight optimization

The small numerical primitive implements centered prediction covariance,
regularized linear initialization, exact real-arithmetic clip-2 weighted
Pearson objective evaluation, and a local SLSQP search with bounded starts.
Because official scoring casts synthesized predictions to float32 before
clipping, the proxy gradient is not the exact computational derivative of
the float32 scorer. Candidate values are re-scored through the unchanged
official metric; the method makes no global-optimum claim.

J1 compared three own H384 CP2 teacher-seed outputs over 48 previously
exposed F0 connected components, split before optimization into 32 calibration
components / 110 groups and 16 evaluation components / 52 groups. Calibration
selected approximately [0.39956, 0.24662, 0.35382], scoring 0.6783535252
versus equal blend 0.6783214526 on calibration. Those weights were frozen
before evaluation access. On evaluation they scored 0.7662406664, below the
frozen equal mixture 0.7663608706 by 0.0001202042; calibration lift did not
carry to that evaluation split. This is one exposed component partition, not a
fresh fold, global optimum, or public predictor. There is no post-hoc selection
of a different pair. No model fit or prediction generation occurred. Receipts:
[J1 execution handoff](campaign_2026-10-12/joint_endpoint_optimization_r1/J1_EXECUTION_HANDOFF_R1.md),
[root admission](campaign_2026-10-12/joint_endpoint_optimization_r1/ROOT_J1_ADMISSION_R1.json),
[fixed component manifest](campaign_2026-10-12/joint_endpoint_optimization_r1/J1_COMPONENT_PANEL_MANIFEST_R2.json).

## Public calibration and plots

The public-history diagnostic found 42 visible upload events, 38 scored rows,
three timeouts and one check error; three of the portal's 45 events were not
visible. Only 11 scored rows have any local metric, spread across incomparable
precision, fold, package, and legacy protocols; 27 scored rows lack a
comparable local feature. A corrected component-purged ridge had chronological
MAE 0.00097353, worse than a recent-five baseline 0.00078941. The rows also
share ancestry and are not independent. This bounded fit does not support a
current package forecast. The
[public regression synthesis](campaign_2026-10-12/public_score_regression_r1/ROOT_SYNTHESIS_R1.md)
and [independent review](campaign_2026-10-12/public_score_regression_r1/PUBLIC_REGRESSION_INDEPENDENT_REVIEW_R1.md)
record precision and protocol qualifications.

The validation/public plot keeps exact F1 receipts, one package-reported value,
five legacy source-reported values, and two conditional EMA6 reconstructions
in separate facets. The two reconstructions are not exact served endpoints.
The three-point A/B/C portal runtime-quality plot is descriptive only:
A→B adds 85 seconds and 0.0012; B→C adds 640 seconds and 0.0003; A→C adds
725 seconds and 0.0015. Blend compositions differ, n=3, with no regression,
correlation coefficient, confidence interval, or forecast. D/E local timing
heuristics are not observations in that plot. See
[protocol-separated validation/public figure](campaign_2026-10-12/public_validation_plot_r2/README.md)
and [runtime-quality figure](campaign_2026-10-12/public_runtime_quality_r1/README_R1.md).

## Serving engineering, compression, and package validity

All final Q8 ZIPs use an independently built and reviewed precision-aware
native runtime and callback wrapper. The serving path resets recurrent state,
returns None during warm-up, emits the two required finite outputs for scored
rows, and replays deterministically. Exact archive tests are compatibility
checks; they do not replace full official scoring of the exact package.

The reviewed native single-member full-fold Q8 control completed F0/F1 quality
evaluation as listed above. Earlier packet generations found real blockers:
Python 3.11 dependencies missing in the system interpreter, absolute paths
bound to old worktrees, incomplete footer identity enforcement, and CLI source
transport that did not include sidecars. These were preserved and repaired in
identified revisions; static preparation passes were not treated as runtime
passes. The exact R6 admission and native source receipts are indexed under
[native single-member clean-fold Q8 R6](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6/ROOT_ADMISSION_R6.json).

October-12 native CPU work tested callback-preserving changes. Activation
fusion and scaled projection precomputation passed output/reset parity but
paired timing directions disagreed; PGO was slower in paired checks. Shared
frozen backbones with multiple heads substantially reduced local callback cost
with bit-identical outputs; their quality remained fold-conditional, so the
default is off and this did not independently qualify a final package.
Raw-prefix repetition screening found zero exact adjacent repeats in 199,990
pairs across ten sequences, which defers only that exact cache opportunity.
The current instruction is to exploit actual Q8 headroom before Q4; no E
runtime guarantee follows from Intel callback timing. Complete methods,
artifacts, independent reviews, and unstarted native ideas are in the
[Zen4/native serving research index](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/INDEX.md).

## Ideas tested, deferred, and closed at their exact scope

The full append-only sources remain the record:
[hypothesis ledger](hypotheses.jsonl), [experiment ledger](experiments.jsonl),
[idea ledger](IDEA_LEDGER_20261011.md), [campaign loop log](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md),
[active allocation/status](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md),
[older hypothesis register](campaign_2026-10-06/bandit_continuation/IDEA_REGISTER.md),
and [research-tree journal](bandit_tree/journal.jsonl). Preserve original rows
and claim scopes; do not rewrite historical negatives into universal family
conclusions.

Useful mechanism families explored include recurrent width/depth and
training-horizon ladders; independently seeded teacher/student ensembles;
heterogeneous Muon/AdamW parents; distillation target alpha and clipping;
teacher scale and target normalization; auxiliary objectives; HFT readout
adaptation; fixed blending and component-cross-fitted weight selection;
native quantization and recurrent-kernel speedups; long-range causal memory,
regime/age representations, event and multi-timescale signals; cross-fold
robustness; and public-transfer diagnostics. Complete source cards, original
falsifiers, exact closures and no-fit proposals remain in the linked ledgers
and generated tree.

Specific negative or low-value results include output-space smoothing,
hysteresis, class/age calibration, duplicate-row shortcuts and lattice
probes, windowed GBM, exact raw-prefix cache applicability, tested bootstrap
student recipes, and the listed naive Q4 conversion. Some target/crop,
teacher, epoch, width, and fixed-blend comparisons were near noise or varied
by fold. These exact treatments may be deferred or closed for opportunity
cost; they do not disprove broader mechanisms such as causal persistent-state
representations, LUPI/lookahead targets, hidden-state residuals, low-rank
distillation, or changed native kernels. The latter remain proposed or
unstarted unless a receipt says otherwise. No research sweep is authorized
now because of the explicit stop.

Older research reports contain historical policies and scores that later ADRs
superseded. For example, October-10 notes prioritized public leaderboard
selection and cite an older 0.6668 best; the current rule uses valid local/CV
evidence for selection and records 0.6671 as the latest user-reported result.
Historical information remains useful for lineage but should not override the
current objective and stop receipt.

## Execution, incidents, and resource controls

No new rental, top-up, or billing change was authorized in this archive.
Vast lane B's last evidence showed zero credit/no owned instance, with later
read-only access unavailable; lack of a refreshed account view is not proof of
remote teardown. Kaggle runs require phase-level receipts, actual mounted input
hashes, worker resource checks, terminal output collection and quota
reconciliation. RUNNING alone does not prove a worker phase.

The strongest current support admission target is 17 decimal GB aggregate
working memory, below the existing 18 GB high and 20 GB hard caps; swap stays
zero and aggregate CPU stays five cores. When an active worker is included,
the memory arithmetic is parent current plus the worker's unused reserved
headroom plus the new bounded task. See
[ADMISSION_MEMORY_MARGIN_20261012_R1.json](campaign_2026-10-12/ADMISSION_MEMORY_MARGIN_20261012_R1.json)
and [KILL_PREVENTION_PLAN_20261012.md](campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md).

The first failed metadata scan treated NPY magic bytes as version/length
fields, then attempted an unbounded read and hit its 650 MiB scope. The
follow-up command also failed and its exact source/command mismatch is
preserved; it is not evidence that a second full metadata scan ran. A later
attempt staged a 1.69 GB cache into tmpfs and was OOM-killed. The source,
commands, exits and cgroup diagnoses are preserved. No model or scientific
comparison completed in those incidents.
Correct readers verify NPY magic and version, cap header length (existing
bound below 64 KiB), account for each payload plus header, and budget dtype
conversion, masks, labels, gradients, library workspaces and transients.
Large frozen artifacts stay on disk or are memory-mapped; never retry a
malformed reader by raising its cap. The fixed F0 blend scorer completed at
1,402,112 KiB under a 1.5 GB limit, with little headroom. Reuse its result;
an equivalent future analysis needs approximately 1.9 GB or a streaming/smaller
panel design and fresh aggregate admission.

Kaggle R4/R5/R6 export repairs failed before valid worker scoring due to mount,
host-admission, timeout-boundary, or source-identity issues. Their preserved
packets are execution evidence rather than negative model results. The R7 run
at this archive snapshot is still only known as provider RUNNING. Root must
classify its worker and outputs from actual terminal receipts. The user-issued
STOP supersedes older open allocations and “keep compute busy” instructions
for new work; healthy admitted runs drain normally.

## Final drain checklist for root

This draft should become final only after:

1. Seed-4 logical run is terminal through the admitted P1/P2/P3, fixed epoch-5
   checkpoint export, and official fixed-468 F0 grade. Record actual unit/exit
   reason, selected epoch, exact source/contract/manifest, fit and score
   membership, prediction/checkpoint hashes, official score and terminal
   resources. A partial result is recorded as partial; do not manufacture CP5
   from an earlier checkpoint.
2. Kaggle R7 has terminal provider status and complete receipt inventory.
   Record whether a real worker phase began, session/version, mounted hashes,
   per-worker results/resources, output coverage and official grade if valid.
   Preserve incomplete output if it fails; classify infrastructure separately
   from science. Reconcile campaign and provider quota without treating the
   initial RUNNING receipt as a completed run.
3. Root updates current state, durable stop, final archive status, research
   indexes, final inventory and generated research-tree snapshot. Preserve
   original ledgers and all raw run receipts.
4. Root confirms all active delegated roles and admitted jobs are terminal.
   Only then close the archive and report the final Main commit/hash.

## Evidence catalog

This map is a navigation layer, not a replacement for raw receipts. Prefer the
specific source/manifest/terminal/grade tuple for each claim. Large model,
prediction, cache, and provider payload binaries are intentionally not copied
into this handoff.

| Evidence area | Authoritative entry points |
|---|---|
| Project objective, methods, and decision rules | [Objective](../specs/research-objective.md), [methods](RESEARCH_METHODS.md), [spec index](../specs/README.md), [current state](../specs/current-state.md), [ADR index](../specs/adr/README.md) |
| Append-only research record | [Hypotheses](hypotheses.jsonl), [experiments](experiments.jsonl), [curated tree config](bandit_tree/config.json), [journal](bandit_tree/journal.jsonl), [tree README](bandit_tree/README.md), [generated navigation](bandit_tree/generated/INDEX.md), [generated integrity snapshot](bandit_tree/generated/integrity.json) |
| Original idea and chronology records | [Idea ledger](IDEA_LEDGER_20261011.md), [October-6 campaign index](campaign_2026-10-06/INDEX.md), [diversity-stack continuation](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/INDEX.md), [loop log](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md), [frontier](campaign_2026-10-06/bandit_continuation/FRONTIER.md), [exploitation plan](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md), [external ideation synthesis](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/external_ideation_20261010/SYNTHESIS_20261010.md) |
| Campaign stop and resource boundaries | [Stop and finalization gate](campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json), [memory margin](campaign_2026-10-12/ADMISSION_MEMORY_MARGIN_20261012_R1.json), [kill prevention](campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md), [metadata-only worktree inventory](campaign_2026-10-12/research_archive_inventory_r1/MAIN_STATUS_INVENTORY_R1.json) |
| Clean-T3 CP5 student / seed confirmation | [CP5 independent grade review](campaign_2026-10-12/clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md), [review JSON](campaign_2026-10-12/clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.json), [grade handoff and trainer records](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/CONTROL_CP5_FULL_VALID_GRADE_HANDOFF_R1.json), [seed-4 launch contract](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/FIT_CONTRACT_R1.json), [seed-4 launch manifest](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/LAUNCH_MANIFEST_R1.json) |
| Teacher cache and conditional values | [Three-teacher F0 grade](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json), [fixed half prereg/result](campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/README_R1.md), [fixed half result JSON](campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/RESULT_HANDOFF_R1.json), [F1 fixed half](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_new_f1_fixed_half_20261012_r1/RESULTS_R1.md), [F1 HFT comparison](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_hft_export_review_20261012_r1/RESULTS_R1.md) |
| Complete native Q8 quality | [Native Q8 quality result](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_shared_cleanfold_q8_r2/quality_output_r2/QUALITY_RESULT_R2.json), [R6 admission](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6/ROOT_ADMISSION_R6.json), [serving methods and negatives](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/INDEX.md) |
| A/B/C submission artifacts and user reports | [A/B/C README](../submissions/10-12-best-r1/README_R1.md), [public results receipt](../submissions/10-12-best-r1/PUBLIC_RESULTS_20261012_R1.json), [package manifest](../submissions/10-12-best-r1/PACKAGE_MANIFEST_R1.json), [readiness state](../submissions/10-12-best-r1/READY_R1.json), [root upload release](../submissions/10-12-best-r1/ROOT_UPLOAD_RELEASE_R1.json), [independent byte/runtime review](../artifacts/submission_review_10_12_best_r1/SUBMISSION_PACKAGE_REVIEW_R1.json) |
| D/E runnable but ungraded artifacts | [D/E README](../submissions/10-12-best-r2/README_R1.md), [D/E package manifest](../submissions/10-12-best-r2/PACKET_MANIFEST_R1.json), [technical review](campaign_2026-10-12/independent_reviews/submissions_10-12-best-r2/D_E_TECHNICAL_REVIEW_R1.json), [paired runtime review](campaign_2026-10-12/z5_blend_runtime_pair_r1/README_R1.md) |
| Component-panel optimizer | [Plan and method](campaign_2026-10-12/joint_endpoint_optimization_r1/PLAN_R1.md), [numerical primitive](campaign_2026-10-12/joint_endpoint_optimization_r1/NUMERICAL_PRIMITIVE_R2.md), [admission](campaign_2026-10-12/joint_endpoint_optimization_r1/ROOT_J1_ADMISSION_R1.json), [calibration/evaluation handoff](campaign_2026-10-12/joint_endpoint_optimization_r1/J1_EXECUTION_HANDOFF_R1.md), [evaluation JSON](campaign_2026-10-12/joint_endpoint_optimization_r1/J1_EVALUATION_RESULT_R1.json) |
| Public calibration and comparison plots | [Regression synthesis](campaign_2026-10-12/public_score_regression_r1/ROOT_SYNTHESIS_R1.md), [independent regression review](campaign_2026-10-12/public_score_regression_r1/PUBLIC_REGRESSION_INDEPENDENT_REVIEW_R1.md), [protocol-separated plot](campaign_2026-10-12/public_validation_plot_r2/README.md), [runtime/quality plot](campaign_2026-10-12/public_runtime_quality_r1/README_R1.md) |
| Kaggle R7 status and producer/consumer review | [R7 review](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r7_export_recovery_20261012_r1/INDEPENDENT_REVIEW_R1.md), [R7 current provider status](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/monitor_20261012_1358_r7_running_r1/R7_PROVIDER_RUNNING_R1.json), [R4/R5/R6 independent review](campaign_2026-10-12/independent_reviews/clean_t3_f0_pair_kaggle_r4_r5_r1/INDEPENDENT_REVIEW_R6_R1.md) |
| Older methods, replication, and ideation | [Current best explained](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md), [Maxim replication reflection](campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md), [external ideation synthesis](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/external_ideation_20261010/SYNTHESIS_20261010.md), [October-10 notes index](RESEARCH_NOTES_INDEX_20261010.md) |

Several links deliberately point into the historical continuation tree; that
tree preserves attempt versions, raw logs, root releases, independent reviews
and terminal evidence side by side. It also contains duplicate snapshots and
monitor outputs. The claim-bearing artifact is the explicitly named receipt,
not the newest filename by assumption. Generated tree summaries are
navigational snapshots; append-only ledgers and source-bound result receipts
remain authoritative. Raw provider output, private credentials, challenge-3
materials and model/data binaries are not embedded or exposed here.
