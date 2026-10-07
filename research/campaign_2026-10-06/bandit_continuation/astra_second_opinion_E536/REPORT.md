# E536 independent second opinion

Review owner: native Astra, requested xhigh by the user for this independent critic. Research basis: root Challenge 2; own weights and independently written production implementation; top-1 / >0.700 remains the objective. Root retains decisions. This report proposes work; it does not authorize spending or change a running recipe.

Read-only source/receipt investigation, with report-only changes in isolated worktree `research/astra-second-opinion-E536`, based at `9a6f1bd2`. Live evidence was read from the main workspace because many newest receipts are untracked there. No fits, rentals, credential reads, provider mutations, delegation, or prediction regrading were performed. Skills applied: conduct-creative-research and empirical-validation, research-recipe-audit, research-evidence-review, wnn-connectome-top1. Verification is source inspection / recorded empirical evidence; public transfer remains tier C. Review cutoff is the newly landed 2026-10-07 23:30:34 UTC matched-control grade, plus the inspected H232 V4 packet. Later runtime state must be reconciled separately.

The strongest recommendation is to deliver a small number of interpretable models through one warm execution path. There is enough ideation. The current bottlenecks are a concrete admission bug, fit-scale preparation overhead, confounded architecture comparisons, and insufficiently discriminating tests of teacher quality and optimization. None establishes a performance ceiling.

## Immediate P0: H232 V4's gate contradicts its packaged feature schema

**Confirmed source defect, conditional on the inspected packet being the executing artifact.** In [V4 profile_worker.py](../continuous_execution_E529/h232_conditional_v4/packet/profile_worker.py), `evaluate_conditional_fit_gate`, line 928, rejects `n_in != 160` with `profile_input_width_is_not_full_feature_160`. Its actual `source_archive.tar` contains:

- `src/wnn/data.py:17`: `N_FEATURES = 112`.
- `src/wnn/features.py:24`: event features count `22 + 22 + 22 + 8 + 3 + 22 = 99`.
- `scripts/train_rnn.py:1828`: input width is raw plus event features, plus only explicitly enabled extras. The packet requires `raw_only=False`, `heuristics=False`; its ordinary full feature width is **211**.

The historical [E521 provenance](../../../../artifacts/runs/bd26_w384raw_s3_f0/provenance.json) independently records normalization shape 211. This is not a preference about features: a correct 211-wide profile cannot pass the inspected 160-wide gate. No claim is made that the live worker has reached that line yet.

Inspected SHA256: worker `0bbebb928a2a628cf4e953b8c088223f01bc57af206a7251eec5f5f2062cf20d`; source archive `afde296e1c5e2f0c47a6fb352c2dea8e0a3df75900f53c26f103b3921df1f565`.

**Smallest correction:** cloud operator independently verifies the exact pushed packet, preserves the current attempt, and prepares an identified successor whose expected input schema matches the actual packaged feature names/count. Test the gate against metadata produced by the assembled trainer, rather than a handcrafted fixture that repeats 160. Preserve all other scientific fields. Do not modify the live packet or suppress the check.

Reproduction, without CUDA or reading competition arrays:

```sh
rg -n 'n_in != 160|profile_input_width' research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet/profile_worker.py
tar -xOf research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet/source_archive.tar src/wnn/data.py
tar -xOf research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet/source_archive.tar src/wnn/features.py
tar -xOf research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet/source_archive.tar scripts/train_rnn.py
```

## What the newest scientific evidence says

| Comparison | Actual evidence | Allocation implication |
|---|---|---|
| H230 family-on vs exactly matched family-off | 0.681719502 vs 0.681658487; **+0.000061015**, component bootstrap SD 0.000079291, 95% interval [−0.000091445, +0.000210497]; 468 sequences, 141 components, one paired training seed | Small positive exploratory effect. Keep the mechanism, but give a new teacher or meaningful optimization change priority over an identical family-head seed repeat. |
| H230 vs same-seed warm parent | +0.000778295 on the same exposed fold | Useful continuation result; does not identify heterogeneity or family supervision as its cause. |
| H230 vs same-seed C1 | +0.000101360 | Descriptive only: target/source/head paths differ. It is not an isolated heterogeneity estimate. |
| H230 vs reconstructed incumbent | Incumbent fitted 374/468 evaluated validation groups; H230 and eleven audited teacher ancestors fitted zero | The −0.008115 local ranking cannot establish held-out inferiority. Preserve the number with its invalid comparison label. |
| Actual public endpoints | Own best 0.6660; C1 singles 0.6647/0.6654; H230 single 0.6649; teammate pair 0.6674 | These are real endpoint observations. They do not identify why the methods differ or establish that independently reproduced all-data teachers/students fail. |

Sources: [matched grade](../wide_baseline_heterogeneous_H230/H230_MATCHED_CONTROL_GRADE_V1_20261008.json), [candidate grade](../wide_baseline_heterogeneous_H230/H230_SEED3_INDEPENDENT_GRADE_V1.json), [independent membership review](../wide_baseline_heterogeneous_H230/H230_MEMBERSHIP_INDEPENDENT_REVIEW_V1_20261008.md), [H230 public receipt](../wide_baseline_heterogeneous_H230/H230_PUBLIC_RESULT_20261008_V1.json), [C1 public receipts](../submission_C1_E524/E524_PUBLIC_RESULTS_20261007_V1.json). I inspected these results; I did not independently rerun their numerical grading. Matched-grade SHA256: `82aa53f4eafc26dffe286bae9ab2d3ef4bac5189246f3bf1dcdb04b11cbc7db9`.

The family contrast's small magnitude matters more for allocation than whether its interval crosses zero. The interval conditions on these fitted models; it does not estimate training or public-transfer uncertainty. A second seed could be useful later, but does not address the currently larger uncertainty about teacher signal and final all-data delivery.

## Stop conflating useful teacher production with controlled architecture evidence

H232's [original hypothesis](../ideation/tcn_teacher_recipe_grok_20261007/H232_CAPACITY_EVENT_FEATURE_INTERACTION_V1.md) removes only raw-only from the E521 recipe. Actual E521 seed 3 used batch 16, four epochs, 29,840 attempted updates, 5,169.2 fitting seconds. The [H232 AMP4096 successor](../continuous_execution_E529/recipes/H232_FULLFEATURE_GRU384_STAGED_V1_AMP4096_S3.json) uses batch 8, four epochs and a changed AMP initialization. For 11,940 fitting sequences, batch 8 implies 1,492 outer batches per epoch versus 746 at batch 16: approximately **twice as many optimizer attempts**, with a different OneCycle schedule trajectory and random batching. Full feature effects cannot be isolated from that difference.

Let the already admitted useful teacher run reach an artifact after the gate repair. Label it a full-feature recipe candidate. If a feature-effect conclusion is needed, add the raw-view B8 control under the same source/numerical policy, or run a matched B16 pair after device admission. A width-by-feature interaction requires the compact pair as well; H239 already states this. Do not claim a width effect merely because a GRU384 model exists.

H217's [TCN/GRU recipe](../continuous_execution_E529/recipes/H217_TCN_RAW_SCRATCH_STAGED_V1.json) is better scoped: it explicitly says architecture parameter count and compute are unmatched while membership, target, schedule and batch choice match. Keep that caveat. Four identical epochs are a reasonable screening allocation, but are not evidence that either architecture has been adequately optimized. Preserve the full causal sequences and export both endpoints, including a fixed contribution test to the strong anchor. A weaker standalone TCN may still be the more useful teacher.

## Objective and data semantics deserve more attention than another family label

The official metric is correctly pooled, clipped, target-wise WP in [gmetric.py](../../../../src/wnn/gmetric.py). The actual trainer optimizes a surrogate with several independent decisions. In the frozen [H230 trainer](../continuous_execution_E529/h230_source_v2/scripts/train_rnn.py):

- `main_row_weights` at lines 568–587 uses `p(x)^2 + 0.03` for the current recipe, then replaces weights on supplied-validation fitting sequences by a batch-normalized true-mask rule.
- Distilled loss at lines 2185–2218 clips original labels, clips the already-combined main teacher target, weights by original clipped-label magnitude, and normalizes by the current chunk's total weight across both targets.
- Family outputs receive raw targets through a separate average loss. A numerical coefficient of 0.1 does not establish its actual gradient share.

These choices are allowed and largely faithful to the inspected reference. They are not consequences of the official metric. Per-chunk normalization gives chunks equal update opportunities despite different scored-weight mass; a shared target denominator is not the same as averaging two target correlations. Propensity weighting is a learned approximation to hidden scoring membership, and the true-mask rescaling changes the fit distribution again. The old H029 register contains evidence favoring power 2 over power 1 on older models; **do not rediscover that sweep as a new idea without checking E039/E048**. The current transferable question is whether the new raw/full teacher family and longer training benefit from the already successful weighting treatment, not whether power 2 has never been tried.

H217 and H232 deliberately omit propensity and auxiliaries. Their teacher diversity is legitimate, but raw/all-row training underperformance would not show that the architecture cannot solve the scored task. Add one matched scoring-aware teacher treatment before rejecting a promising new family. Keep ancestry clean for the propensity source; true validation masks are allowed only on fitting groups.

The heterogeneous target also has explicit shrinkage: teacher coefficients sum to 0.8125, alpha is 0.7. In the unsaturated equal-teacher thought experiment, target scale is `0.3 + 0.7 × 0.8125 = 0.86875`. That scale can affect optimization and clipping even though unclipped correlation is invariant to a common positive scale. This is reference fidelity, not a bug. [distillation.py](../continuous_execution_E529/h230_source_v2/src/wnn/distillation.py) already constructs source-scale, normalized and scale-matched homogeneous targets. The existing C1 command also uses 0.8125; do not claim its comparison is confounded by a missing coefficient sum. Future teacher tests should keep this scale fixed, or name normalization as the intervention.

Before commissioning further distillation variants, compute a compact fitting/evaluation diagnostic for the actual teacher mixture: per-target mean/variance, clipping fraction, error by causal event-age regime, mixture residual association conditional on the clean strong anchor, and loss/gradient shares by family. Use it to choose one discriminating student test. Disagreement alone is not useful complementary information. Evaluation-label-derived next-change distance is diagnostic only, never a serving gate.

## Preparation has consumed scientific-fit-scale resources

[E520 V10 reconciliation](../own_baseline_E520/E520_PROFILE_V10_FINAL_QUOTA_RECONCILIATION_20261007_R1.json) records **6,778.546 GPU-quota seconds** spent on recovery through V10, with no fit, export or grade. Later [V12 profile](../continuous_execution_E529/downloads/e520_profile_v12/e520_profile_v11/profile_result.json) took **977.1 worker seconds**, still profile-only; its median loader wait was 3,247 ms against a 3,399 ms inter-start interval. These timings must not be added as independent stage medians. For scale, [H230's complete two-epoch history](../../../../artifacts/runs/bd26_h230_hetero_s3_f0/history.json) records **1,983.4 training seconds** and 14,914 completed updates. Device and quota units differ, so this is not a cost-equivalent speedup calculation; it is sufficient evidence that repeated preparation is a first-order bottleneck.

Vast A2 transferred part of a 37.1 GB payload until a 1,800-second timeout, then produced no fit; [terminal spend receipt](../continuous_execution_E529/vast/H217_A2_TERMINAL_FAILURE_SPEND_RECONCILIATION_V1_20261008.json). A3D's [admission](../continuous_execution_E529/vast/H217_VAST_ADMISSION_V2_20261008_A3D.json) reserves $2.80 including $0.59 upload and allows 5,400 seconds staging. A queue that warms multiple useful recipes on one verified dataset can therefore matter more than nominal GPU price. Do not repeatedly rent cold replicas before measuring this worker's useful end-to-end rate.

**Minimum delivery change:** extend the existing reviewed packet path, not a new framework:

1. Run assembled entrypoint imports, final argv serialization, helper closure, path resolution and the actual feature-schema/gate check **before** expensive data hashing or rental. Test packaged bytes, not just source modules.
2. Keep one worker alive through verified input identity → bounded actual-shape profile → conditionally admitted fit → fixed validation export → frozen metric handoff. Future packet review covers this complete path once; narrow repairs receive a diff review. Existing ancestry, causality, resource, budget and instance-ownership checks remain binding.
3. Freeze two or three diverse successors before staging a paid worker. Reuse verified immutable input bytes and existing loader/cache primitives within that worker. Reprofile shape-changing recipes; do not repeat full identity work for unchanged inputs merely because the run name changes.
4. Make a result manifest the handoff: checkpoint, predictions, group mask/ancestry, metric, phase times and terminal outcome. H232 V4 explicitly ends at checkpoints, so assign its export/grade successor now. CPU evaluation may run outside the GPU process under its own bounded allocation.
5. Report completed mechanism comparisons, accelerator work, preparation fraction and ready-queue depth. Do not count profile success as an idea screen. Keep live frontier short and link the dated status archive; its currently repeated contradictory snapshots increase controller reconstruction work.

The current generic dispatcher and many hash receipts have real safety value, especially for strict Vast ownership. Removing them indiscriminately would be wrong. The failure is repeated manual recomposition and incomplete end-to-end tests, not the existence of provenance.

## Replace the zero-AMP-skip rule prospectively

[H232 V1 diagnosis](../continuous_execution_E529/H232_PROFILE_V1_FAILURE_DIAGNOSIS_20261008_R1.json) records 230 finite losses, 229 completed optimizer updates and one skip, and marks throughput invalid. H230's delivered fit records **six** skips over 14,920 attempts; E521 seed 3 records **15** over 29,840. A universal zero-skip scientific validity rule is inconsistent with these accepted runs.

PyTorch explicitly describes initial scaler calibration and occasional skipped updates as expected behavior that protects parameters from overflowing gradients. This does not prove a particular skip harmless, but it defeats the blanket inference that any skip invalidates a fit. [PyTorch 2.11 GradScaler source](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/amp/grad_scaler.py).

Keep current frozen policy and receipts unchanged. For the **next** independently reviewed policy, distinguish finite forward loss/parameters, scaler backoff calibration, sustained gradient instability, incomplete update evidence and resource failure. Record attempts, completed updates, skip positions and scale evolution; report throughput per completed update and per processed sequence. A small observed skip rate earns investigation proportional to its effect, not automatic repeated cloud startup. Exact allowable rates/windows should be declared before the next run, with a persistent-instability abort and an FP32/BF16 diagnostic if needed.

The actual H230 trainer calls `sched.step()` after `scaler.update()` unconditionally (line 2313), and increments its update counter even if the optimizer skipped. This is a confirmed schedule-clock difference, not a demonstrated score bug. For faithful reference diagnostics preserve it explicitly. A completed-update schedule is a separately named future treatment; do not silently change it in a repair.

## Proposed 24-hour portfolio and allocation decisions

This is an ordered, resource-conditional plan. Fresh quota, profile and all-in reservations remain root-owned. Do not shorten useful full fits merely to claim 5–6 ideas/hour; first measure warm throughput, then size capacity from measured total work. Preserve the user's target as an operational scaling objective.

| Priority / window | Concrete deliverable | Why this is discriminating |
|---|---|---|
| Now–2 h | Independently verify/fix the H232 160/211 gate under a new attempt; collect actual phase from active lanes; bind H232 export/grade ownership. Integrate the landed H230 family-off grade. | Removes a guaranteed admission failure and closes the already-paid scientific comparison. |
| Parallel, 0–6 h | Freeze the closest independently written all-data heterogeneous recipe and the isolated exact-source diagnostic with fresh own weights and fully enumerated teacher/init ancestry. Start the first runnable admitted stage. | Finalist delivery and implementation diagnosis answer different questions. Do not make the independent endpoint wait for rebuilding the entire diagnostic ancestry graph. The successful source is commit `833aaf54`, `research/iterations/2026-10-05-wave6/scripts/train_rnn.py`, not a Transformer headline. |
| 2–8 h | Complete H217 TCN/GRU predictions and H232 predictions. Use the fixed clean anchor for standalone and predeclared contribution diagnostics. | Produces actual diverse teachers; architecture failure is not inferred from startup failure or standalone rank alone. |
| First warm screening cohort | Three distinct candidates: **scoring-aware training of the new teacher** versus its all-row recipe; **H233 conditional readout** versus its frozen-parent control; **H244/H238 causal fast/slow normalization or innovations** versus a same-dimensional fixed-statistics control. Use existing code/cards and whole-component subsets. | Tests objective, readout and representation mechanisms. Do not add a large menu of near-duplicate feature statistics. Source/recipe audit chooses the runnable version. |
| 4–16 h, separate confirmation lane | One useful full-data/full-budget teacher comparison: matched GRU384 input view and batch, with a **4-epoch versus 12-epoch training-budget plan**, or an independently declared continuation whose optimizer reset is explicit. Record fixed-budget endpoints and learning curves. | Tests whether short schedules are constraining capacity. Do not call a four-epoch model the architecture ceiling. Twelve epochs is a proposed 3× budget experiment, not an inferred optimum or permission to exceed caps. |
| 8–24 h | One truly different sequence family, selected by the user's separate architecture/literature investigation, with a GRU control and a teacher-to-fixed-student path. A width ladder such as 256→384→768 and credit/context ladder 2k→4k/8k are eligible after device profiling. | Larger teachers can exceed serving limits; an architecture needs relevant optimization and causal history before rejection. A tiny-data loser may receive one full-scale test for a concrete scale-dependent reason. |
| After the first useful new teacher lands | One fixed-student compression comparison: current teacher mixture versus new source replacement/addition with unchanged main scale, student initialization, head structure and optimization. | Establishes teacher value in the intended endpoint, rather than assuming teacher WP or disagreement transfers. |

The first warm screen cohort can use the saved 10%/30% nested-component protocol, but measure at least two fidelities for a few families before relying on rankings. Equal epochs on a 10% subset have roughly one tenth the updates; they are not a neutral cheap proxy for full training. Whole histories, fixed evaluation groups and explicit update/row budgets remain mandatory. Existing full-data teacher ancestry makes a subset student screen a student-compression screen, not an all-model small-data experiment.

Protect one longer-budget lane from the hourly idea-count target. Protect one genuinely different family from the current recurrent-distillation lineage. Keep the highest-return ready work running while the next recipe is prepared; do not wait for the entire matrix. Additional owned GPUs are justified by measured warm cost and ready work, never by nominal TFLOPS or the backlog's size alone. The $20 ceiling and unproven-instance prohibition remain unchanged.

## What to stop, and what reopens an arm

- Stop using the contaminated incumbent reconstruction as a held-out gate. Audit exact affected decisions, including ancestor/propensity membership, before reopening them. The correction does not erase clean negatives.
- Stop repeated calibration of the unchanged E514 weak pool. Its [actual result](../conditional_weak_portfolio_E514/REPORT.md) chose zero contribution for all four rules; that is relevant to that pool. A new source, a different conditional role or corrected ancestry can reopen a specified descendant. No inaccessible pristine holdout is a prerequisite for honest exploratory work.
- Stop treating two-epoch family-head continuation as the whole heterogeneous-teacher mechanism. The new control says that head is a small lever in this setting. Reopen its priority when teacher family signal, head location/target, optimization duration or student bottleneck changes concretely.
- Stop universal family rejection from tiny models, wrong features, wrong masks, unmatched schedules, or failed execution. Require a receipt identifying the actual data view, source/ancestry, fitting budget, target and comparator. Infrastructure-censored attempts retain unknown scientific value.
- Stop generating another broad idea list while several distinct implemented arms lack an operator. The user's separate architecture/MI scout should return a short ranked set with runnable experiments; it should not become a prerequisite for current useful execution.
- Stop turning every uncertainty into another compulsory preflight. Keep hard semantic/resource checks; make each extra diagnostic name the failure it can prevent and the decision it can change.

There is no evidence here supporting a public forecast above 0.700, and no evidence that it is unattainable. Reproducing the 0.6674 reference is a useful baseline milestone. The route to a substantially stronger endpoint needs stronger or complementary information, adequate optimization, and measured transfer into the submitted model; matching family labels and accumulating reviewed packets cannot establish that progress.

## Limits and handoff

This critic independently inspected source and original receipts, including the actual packaged H232 trainer/features, the successful reference script, H230 candidate/control commands, metric, optimizer loop, history, quota receipt, and new matched grade. No numerical result was newly reproduced by this report. The latest live provider phase, export completeness, feature EDA and exact-reference runnable diagnostic are owned by other roles. Recommendations remain proposals for root integration; no ADR, budget, scorer, frozen source or campaign ledger was changed.
