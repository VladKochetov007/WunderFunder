# Unified research library

## 4 November: objective/clipping comparison on existing L4

[The completed fixed-endpoint report](../experiment_drafts/l4-objective-20261104/RESULTS_20261104_R1.md)
finds no useful gain in either arm, so no optional students are launched. Actual
six candidate/control histories, all-epoch train/valid and gradient figures are
retained. Clip5 reduced clipping markedly without improving WP; late auxiliary
annealing is negative at both seeds. The rows below describe the earlier launch.

[PR62 execution](../experiment_drafts/l4-objective-20261104/EXECUTION_20261104_R1.md)
records four finite actual profiles and four fresh paired GRU256x2 fits,
comparing clip5 and late auxiliary annealing against reused controls. The
diagnostic mask failure is retained without a scientific conclusion; the
predeclared prior fallback chose the second arm. All sources, official
train/valid membership, fixed checkpoints and conditional student spending
rules are frozen. No new quality/public/serving result is established yet.

## 31 October: submission-only full-corpus preparation

[Final TL3 + LSTM packet](../experiment_drafts/final-submit-all-labeled-20261031/README.md)
and [ADR0045](../../specs/adr/0045-submission-only-all-labeled-final-fit.md)
record the user's explicit all-labeled final fit. Exact12480wholefitsequences,
no independent eval or same-valid score claim; research chronology stays intact.
The17TL3dependencyfits+oneLSTM retain prior endpoints and coefficients. Source
audit and scoped CPU checks are complete; native package/profile/full-fit
execution evidence is still pending. Historical results below remain unchanged.

## 31 October: approved batch/TBPTT continuation

[The five-fit screen and fixed TL3 mixtures](../experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1/README.md)
follow the completed multistage study. Three arms share GRU384x2 and fixed4E;
two seeds per arm include one exact reused control. Six prior-model mixtures
are complete; best observed90%TL3+10%LSTM is0.692502063 versus0.692377425.
This small exposed-valid gain is not public confirmation. The execution packet
binds actual profiles, source/config identities and bounded final collection.

The [completed17-row comparison](../experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1/RESULTS.md)
has5/5 new fixed4E fits, exact reused A/1301 and all expected composites.
B-A is negative at both seeds; C-B is mixed. All six new GRU-pool/TL3 mixtures
are negative. Five configured-final weights/receipts are published and verified;
owned GPU commands exited, no successor queued, foreign NIER/MPS preserved.

## 30 October: heterogeneous teachers and12 fixed ensembles completed

[Six-fit execution](../experiment_drafts/heterogeneous-tl3-20261030/README.md)
completes one LSTM, one causal TCN and four paired GRU students, plus the original
W18 recovery. [The full table](../experiment_drafts/heterogeneous-tl3-20261030/results/R3/COMPARISON.md)
uses canonical fresh TL30.6923774246. Best tested90%TL3 +10%GALAM is0.6924964468
(+0.0001190223). Heterogeneous students improve at both matched seeds
(+0.0006193485/+0.0002192739); their ensemble gains0.0004057735 over controls,
but its20%addition to TL3 is negative. All25model/ensemble outputs independently
agree within5.55e-15. Full official valid remains exposed; no fitted coefficients,
public result or significance claim. Our GPU workers have exited; exact seven-model
checkpoint/source archives and source-qualified failures are retained in the packet.

## 29 October:24 model checkpoint artifacts published

[Checkpoint publication](../experiment_drafts/checkpoint-publication-20261029/README.md)
delivers8 PR46 and16 PR49 model bundles through private GitHub releases, linked
from their experiment PRs as SemyonEpanov. Best-observed and fixed TL3 endpoints
remain distinct; W18 is untrained/absent. All source hashes, CPU payloads and
roundtrip downloads were verified, with independent selection/ancestry review.
This changes no fitting recipe or scientific result.

## 29 October: heterogeneous teachers and reported TL3 endpoints

[Heterogeneous-teacher proposal](ideas/HETEROGENEOUS_TEACHERS_20261029_R1.md)
turns the user's multi-architecture pool into a proposed6-new-fit comparison,
with existing ancestors reused and the student's WP as the primary endpoint.
It records PR49's terminal report:16 fits complete, best tiedlead WP0.691415,
W18 blocked on cache_unavailable and all6 fixed fine-tune endpoints below
their base parents. Scores are from the artifact-verified publisher; the current
SSH reconnection timed out, so live state and the cache's raw error are unverified.
The proposal changes no frozen source or running allocation.

## 28 October: fresh TL3 train-only restoration

The [TL3 restoration packet](../experiment_drafts/tl3-chronological-20261028/README.md)
recovers the three-member endpoint and17-predictor dependency graph, with every
ancestor retrained on official train. Historical valid-fitted propensity is
replaced explicitly by neutral weighting; unknown optimizer facts stay unknown.
The first scratch GRU256x2 teacher completed all4 epochs: fixed-endpoint full-valid
WP0.6882596674 versus the exposed plain-GRU benchmark0.6778561232. All4 train/valid
WMSE and WP points are in the existing dashboard (run tl3c_fin256_s3); seven original
GALAM fits continue. Native-Muon, causal tiedlead and
cache-provenance adapters and the fixed endpoint scorer have focused checks.
The [R8 receiving packet](../experiment_drafts/tl3-chronological-20261028/receiving/R8/README.md)
records verified first-teacher adoption, completed prediction export and38
passing receiving checks. All17 model entries are visible in the dashboard;
fin256_s4 is fitting and15 predictors are queued.
[Draft PR49](https://github.com/VladKochetov007/WunderFunder/pull/49) is published
as SemyonEpanov with automatic hourly/fit-terminal updates. A report-only final
blend dashboard view is wired to verified terminal scoring, which remains pending.
Full TL3 metrics remain pending; old public0.66832 is
not an independent score on our current valid.

## 28 October: authorized fixed eight-L4 launch

The [current launch packet](../experiment_drafts/galam-l4-quality8-20261028/launch/README.md)
supersedes the earlier six-fit and v1 preparation panels with the user's exact
plain-GRU benchmark plus seven GALAM configurations. [Draft PR46](https://github.com/VladKochetov007/WunderFunder/pull/46)
is published as SemyonEpanov. All eight full fits now run after complete data
preparation and independent checks of eight actual-source profiles. Common
GALAM batch384 and plain-GRU32 are frozen. [GRU epoch1](../experiment_drafts/galam-l4-quality8-20261028/launch/FIRST_EPOCH_RESULT.json)
reports full-valid WP0.6681328342 on all1873 valid sequences; it is an intermediate
single-seed result with exposed validation. The node posts hourly and terminal
updates independently of SSH; source/evidence archives and ownership are linked
from the packet. No final model ranking or public-score claim is made.

## 28 October: GALAM local monitoring and L4 preparation

The [implementation packet](../experiment_drafts/galam-l4-monitoring-20261028/README.md)
tracks actual train/valid telemetry, a local selectable dashboard and explicit
recurrent/branch adaptations. Its [live L4 preflight](../experiment_drafts/galam-l4-monitoring-20261028/L4_PREFLIGHT.json)
finds8GPUs but full data disks and an unverified partial train file; no fit or GPU
profile was launched. Original GALAM identity and training quality stay unverified.
[Validation](../experiment_drafts/galam-l4-monitoring-20261028/VALIDATION.json)
records92 targeted passes/1skip, independent criticism and live browser checks;
the local demo is explicitly synthetic. Six runnable recipes remain unadmitted
until actual data/headroom/runtime/profile checks pass.

## 27 October: chronological validation and GALAM reconstruction

[ADR0044](../../specs/adr/0044-chronological-primary-validation.md) makes the
Predictorium-style train/valid boundary mandatory for future primary comparisons.
The [source-traced GALAM packet](../experiment_drafts/galam-reconstruction-20261027/README.md)
provides reconstructed visible blocks,32→32/112→2 example configurations and
canonical training integration. [Sources and unknowns](sources/galam-20261027/README.md)
separate original diagram evidence from assumptions. CPU checks pass; no new fit,
GPU profile, actual-data chronology audit or exact reference identity is claimed.

PR32 R4: [batch96/64 selection, batch128 measurement and CPU/transfer costs](../experiment_drafts/b300-transformer-distillation-runtime/execution_20261024/BATCH_THROUGHPUT_20261024.md).

PR32 execution handoff: [measured Blackwell recipe, startup repair and four-teacher allocation](../experiment_drafts/b300-transformer-distillation-runtime/execution_20261024/README.md).

October 24 PR32 execution: [Blackwell throughput amendment and measured profiles](../experiment_drafts/b300-transformer-distillation-runtime/BLACKWELL_THROUGHPUT_20261024.md).
Four 30-epoch teachers are authorized on eight B300s; precision/backend/layout
comparisons are engineering work and do not count as completed scientific fits.

Start with [current state](../../specs/current-state.md) and the
[accepted local/Kaggle plan](../../specs/plans/2026-10-20-plateau-review-local-kaggle.md).
The user requested implementation of the October 20 cycle with 15 local GPU-hours
and verified available Kaggle quota, without a mandatory 30-run matrix.
Preparation is active; the [campaign frontier](../campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
records actual progress. The [October 18 stop](operations/RESEARCH_STOP_20261018_R1.md)
remains historical authority for unrelated work. This index distinguishes
measured results, implementation drafts and historical plans.

Latest policy and design: [fast screens / large teacher ADR](../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md)
sets4–6 total new fits per initial idea comparison. The
[large causal Transformer proposal](ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
specifies0.68B/1.65B capacity recipes, optional3.17B scaling, dynamics embeddings,
FP32 Muon/WSD/DDP, causal long memory and measured GRU transfer. Sources and
independent design criticism are indexed; no new fit, B300 profile or gain exists.

Latest broad campaign closeout: [Online-features closeout](../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md)
— summarizes causal-state, Kalman, gating, model-blend, submission and local code-draft evidence from the closed campaign. The new plan contains no additional quality result yet.

The October 19 [B300 PR review](reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
records partial width/depth and geometry scores from stopped research-team jobs.
It finds a small directional advantage for the wide GRU, a negative partial
geometry bundle, and no scored age/innovation pairs. These exposed-fold partials
do not promote a model or establish transfer.

Previous: [No-fit research reset and 25-idea portfolio](../campaign_2026-10-17/research_reset_r1/README.md)
[Latest bounded idea screen, R15](../campaign_2026-10-19/idea_screens_r15/RESULTS.md) trains explicit sparse/dense dictionaries with retained residuals. The sparse pipeline gains0.002568WP, below its fixed0.005 spending gate; both targets positive but component/drop-one signs are mixed. Four756-update fits complete on the exposed R3 panel; no factor-identity/GRU/public claim. All R1–R15/F3 are finished; [continuation](../campaign_2026-10-19/idea_screens_r15/CONTINUATION.md) records no R16 admission.

[Latest bounded idea screen, R14](../campaign_2026-10-19/idea_screens_r14/RESULTS.md) tests matched-pole/span conditioning directly. Orthonormal impulse coordinates leave larger128-step objective gaps for both targets; secondary WP is slightly higher, so optimization and prediction evidence remain distinct. All R1–R14/F3 are finished; see [current continuation](../campaign_2026-10-19/idea_screens_r14/CONTINUATION.md).

[Synthesis after thirteen screens](ideas/IDEA_SCREEN_SYNTHESIS_20261019_R1.md) retains three conditional leads, closes only exact negative variants and preserves untested mechanisms. No R14 job or model is admitted; this evidence review adds no data/model execution or empirical result.

[Latest bounded idea screen, R13](../campaign_2026-10-19/idea_screens_r13/RESULTS.md) closes one immutable first99-row mean side-context recipe: WP0.462646 versus current bypass0.499683 and zero0.507387, same initialization/updates. All R1–R13 and original F3 are finished; no new GPU or old-queue cell. See the [current continuation](../campaign_2026-10-19/idea_screens_r13/CONTINUATION.md).

[Current user-authorized EDA/simple-model screen](../campaign_2026-10-19/idea_screens_r1/RESULTS.md):48whole sequences, nine fixed ridge views, component-aware interpretation and a status map for all25cards. EMA's pooled gain reverses after dropping the largest evaluation component; i0-only remains worth a fit-only regularization/data-size check. Existing GRUs are much stronger on the same selected rows, with different fit budgets. This supersedes the earlier discussion-only hold for bounded EDA/simple probes, not for a full campaign.

[Current user direction: no-fit discussion of 25 hypotheses](../campaign_2026-10-19/no_fit_ideation_r1/README.md). Independent generation, prior-overlap criticism and mathematical/source checks replace automatic historical reconstruction and quantization work. The cards are unmeasured proposals, with close refinements and dependencies marked. Existing F3 has maintenance-only authority; no new fit, data replay or code implementation was performed in this session.

[Current continuation,18 October21:03UTC](../campaign_2026-10-18/quality_gap_r1/CONTINUATION_20261018T2105Z.md) records healthy F3 wide epoch4, reviewed/staged terminal wrappers and the isolated tiedlead engineering publication. Modified scientific fits and F3 quality remain pending; no new public score.

[Current continuation, 18 October20:31UTC](../campaign_2026-10-18/quality_gap_r1/CONTINUATION_20261018T2035Z.md) separates healthy F3 training and accepted stage1 artifacts from the newly published strict-propensity V3 engineering capability. F3 quality and the modified clean comparison remain pending; no new leaderboard score is claimed.

[18 October measured strong-control comparison](../campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r4/records/RESULTS.md): matched-row F0 WP 0.68115416 versus Q8 0.67382636; conditional delta interval [0.00238446,0.01352521]. [Public incumbent trace](../campaign_2026-10-18/quality_gap_r1/public_submission_trace_r1/README.md) resolves tl3 /1lmehfi3 /0.66832; [clean-counterpart plan](../campaign_2026-10-18/quality_gap_r1/TL3_CLEAN_COUNTERPART_PLAN_20261018.md) remains unexecuted; the bounded historical search is closed and a separately declared modified clean reconstruction is being prepared.

Start with [current state](../../specs/current-state.md). The October 17 research
reset authorizes analysis, EDA and engineering; the separately admitted geometry
pilot has its own bounded fitting authority.
This index distinguishes measured results, implementation drafts and historical plans.

Latest research-reset follow-up: [R2 measured exception impact and prediction dynamics](../campaign_2026-10-17/research_reset_r2/README.md)
— full census, fixed-output interventions, mathematical decomposition and25
revised decisions. No training or new inference in this packet.

Earlier: [No-fit research reset and 25-idea portfolio](../campaign_2026-10-17/research_reset_r1/README.md)
— deployed prediction/cache fixes, a 249.6M-row narrow census, sampled full-feature
and event-lag EDA, primary literature, and independent criticism. No new fits.

## October 18 direct quality-gap investigation

[18:40 continuation](../campaign_2026-10-18/quality_gap_r1/CONTINUATION_20261018T1840Z.md): F3 wide stage1 independently accepted; deep stage1 healthy. [W18 teacher/parent metadata](../campaign_2026-10-18/quality_gap_r1/tl3_w18_checkpoint_execution_r1/README.md) recovers eight exact saved configs; [remaining closure](../campaign_2026-10-18/quality_gap_r1/TL3_CLOSURE_NEXT_STEPS_20261018T1840Z.md) separates known production ancestry from unknown propensity/cache provenance. This is metadata progress, not a new quality score.

The [100-row public submission trace](../campaign_2026-10-18/quality_gap_r1/public_submission_trace_r1/README.md) identifies `tl3.zip` / `1lmehfi3` as the user-reported team best, 0.66832, and binds its name to the archived three-model production recipe. Portal bytes remain unverified. The [quality-gap packet](../campaign_2026-10-18/quality_gap_r1/README.md) separates that recipe from the cold rank1 width/depth control, restores a historical F0 control, and records source-bound residual diagnostics. [F1](../campaign_2026-10-18/priority25_execution_r4/receipts/F1_COMPLETION_20261018T1500Z.md) is complete and accepted; F0/F1 are two paired cells, not CP30.

## Read first

| Need | Start here |
|---|---|
| Established findings and remaining unknowns | [Repository audit](reviews/REPOSITORY_AUDIT_20261017.md) |
| Current models, lineage and packages | [Model catalog](models/CURRENT_MODEL_CATALOG.md) |
| Metric, panels and comparable scores | [Validation protocols](validation/VALIDATION_PROTOCOLS.md) |
| Local, GitHub, beleriand and private archives | [Storage map](operations/STORAGE_MAP.md) |
| Code layout and routine checks | [Architecture](../../docs/ARCHITECTURE.md), [development](../../docs/DEVELOPMENT.md) |
| Retained metadata and exact source identities | [Audit receipts](reviews/audit-2026-10-17/README.md) |

## Browse by topic

| Topic | Contents |
|---|---|
| [Models](models/README.md) | Architecture, optimization, distillation and ancestry |
| [Validation](validation/README.md) | Membership, official scoring, uncertainty and public transfer |
| [Submissions](submissions/README.md) | Releases, package identities and recorded portal outcomes |
| [Inference](inference/README.md) | Native runtime, quantization and serving measurements |
| [Operations](operations/README.md) | Execution, pauses, incidents and preservation |
| [Reviews](reviews/README.md) | Audits, independent assessments and source reconciliation |
| [Methods](methods/README.md) | Metric, research protocol and implementation contracts |
| [Ideas](ideas/README.md) | Existing hypotheses and proposed comparisons |
| [Sources](sources/README.md) | External literature and imported writeups |
| [Private model archive](models/huggingface/README.md) | Checkpoint and run lookup, transfer provenance |
| [October 12 archive](archive/2026-10-12/README.md) | Preserved dated handoff |
| [Templates](templates/README.md) | New narrative records and experiment drafts |

## Reading and recording evidence

Follow a topic page to a synthesis, then to the source-qualified receipt, exact
artifact identity and H/E records. Compare scores only within a stated protocol.
The append-only [hypothesis](../hypotheses.jsonl), [experiment](../experiments.jsonl)
and [decision](../bandit_tree/journal.jsonl) ledgers retain their original rows;
rows and repeated IDs do not count independent completed fits.

Every useful research item needs a descriptive link in its domain README, and
new folders need an indexed README. Use the [templates](templates/README.md),
preserve frozen campaign sources and keep large private payloads at their
recorded storage locations. See [AGENTS.md](../../AGENTS.md#research-note-organization).

<details>
<summary>Dated index preserved before the October 17 maintenance</summary>

The following text is retained for provenance. Its execution/authorization and
pending-status claims are historical; [current state](../../specs/current-state.md)
and later receipts supersede them.

This is the maintained, human-facing research map. It joins different result types through shared topic pages while preserving the original campaign records and evidence paths.

The [critical assessment and separate public and future private leaderboard strategies](reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md) audits the previous approach and sets out ranked experiments, model and representation scaling, and selection for the public leaderboard versus the user's two-to-four-week future-data scenario. Its proposals are now being implemented under the [subsequent explicit user resume](../campaign_2026-10-13/strategy_execution_r1/README.md).

Independent second review: [first-principles verdict and plan](reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md), merged from `claude/stoic-euler-70wff3`. This static critique proposes different priorities; it contains no new fit or recomputed score. The [evidence assessment and experiment cards](ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) distinguish useful compact-model, averaging, symmetry and cross-fitting experiments from unsupported capacity ceilings and policy proposals. The user's existing objective and authorization remain active.

## Start by topic

- [Ideas and hypotheses](ideas/README.md)
- [Methods, sampling, and metric](methods/README.md)
- [Reviews and audit receipts](reviews/README.md)
- [Imported source writeups](sources/README.md)
- [Model families and training](models/README.md)
- [Private HF model archive](models/huggingface/README.md) — checkpoint/run lookup, descriptions, exact local-to-Hub paths and private transfer evidence.
- [Validation and public transfer](validation/README.md)
- [Interactive results dashboard](validation/RESULTS_DASHBOARD_20261012.html)
- [Submission choices and scores](submissions/README.md)
- [Inference, quantization, and runtime](inference/README.md)
- [Execution, incidents, and preservation](operations/README.md)
- [Dated October 12 archive](archive/2026-10-12/README.md)
- [Shared templates](templates/README.md) — narrative notes and experiment draft PR descriptions.

## How records fit together

The JSONL ledgers remain the structured source of truth: add hypotheses to `research/hypotheses.jsonl`, experiments to `research/experiments.jsonl`, and allocation or archive decisions to `research/bandit_tree/journal.jsonl`. A Markdown note gives readers a concise explanation and points to those stable IDs, the frozen recipe, and the result receipts. Do not rewrite past ledger rows or overwrite dated archives.

Use the shared template for new narrative notes. Set its `kind` and `status`, list related hypothesis and experiment IDs, and link the source receipts.

Every useful research item must be linked and described in its corresponding folder's README: reviews, plans, ideas, data and literature sources, evidence, results, analyses, plots, artifacts and operational findings. Explain its contents, relevance, finding or status, provenance and material limitations. Add this entry as part of the same change or handoff, including older unindexed sources when they are used. Each new subfolder needs a README linked from its parent. The full rule is in [AGENTS.md](../../AGENTS.md#research-note-organization).

Frozen campaign artifacts stay in their original locations and are linked here. For large or private data and weights, link their provenance manifest and approved access location rather than copying payloads. The October 12 stop is preserved as history; the subsequent strategy campaign records the current user authorization, allocation and admission checks.

## Reading order

1. Read [current state](../../specs/current-state.md) for live status and constraints.
2. Open the relevant domain README above.
3. Follow its links to a dated synthesis, then to the source-qualified receipt or raw artifact.
4. Check the linked H/E record and validation exposure before repeating a claim.

The dated archive preserves the October 12 handoff as it stood when all admitted work finished, including its historical path inventory. Domain pages provide maintained links to current locations. Reviews, idea artifacts, methods references, and imported source writeups have canonical homes under this tree; use those paths directly.

## Current execution

[October13 strategy implementation and experiments](../campaign_2026-10-13/strategy_execution_r1/README.md) records the subsequent explicit user resume, every actionable review item, full Kaggle/local GPU authority, accepted private-risk preference, owners and evidence. The earlier stop/archive paragraphs describe historical scope; ADR0034 now governs new research. No new model result at admission.

</details>


## 2026-11-04: completed W18 eight-L4 quality screen

[Results](../experiment_drafts/w18-l4-quality-20261104/RESULTS_20261104_R1.md) ? [PR60](https://github.com/VladKochetov007/WunderFunder/pull/60). Eight matched fits completed; causal innovations and alpha0.5 worsened singleton WP, family0.01 had negligible effect. Largest observed fixed-blend gain+0.000051582 on exposed development valid; no promotion/public forecast. All own GPU work finished, weights and evidence published.
