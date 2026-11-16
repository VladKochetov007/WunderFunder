# Current project state

**Handoff update, 2026-11-14 21:40 UTC:** Kaggle R2 is now ERROR, not queued. Both started workers stopped at chronological train-parquet identity validation before training; four cells never started. Logs are saved; diagnosis/retry is deferred by the user. Local GPU hold remains, no WNN services running. [Complete session handoff](../research/unified/operations/SESSION_HANDOFF_20261114_R1.md). Earlier queue snapshots below are historical.

## November 14: CPU checks continue; local GPU held for PHCapital

The user's latest instruction reserves GPU for PHCapital. All owned local GPU/thermal services are stopped; launch wrappers respect the [hold](../research/campaign_2026-11-14/quality_r2/LOCAL_GPU_HOLD_R1.json). Existing six-fit Kaggle panel is unchanged and still queued without outputs. No automatic local restart, paid compute or competition upload.

The [fixed equal FBC/TLead-s5 target](../research/campaign_2026-11-14/quality_r2/teacher_pool_r3/TWO_TEACHER_RESULT_R1.md) scores **.692260448**, current FBC **.691420480**, delta **+.000839968** with both targets positive. Complete predictions, exact train/eval membership and checkpoint bytes are bound; official/reference metric agreement passes. Full valid remains exposed. This is teacher-quality evidence only; three-teacher evaluation, train caches and three prospective student fits remain incomplete/unlaunched. No new stronger submission is established. Pure-teacher KD and fixed weight averaging gave no useful gain.

The final team refresh found [new PR99](../research/campaign_2026-11-14/quality_r2/team_refresh_r3/PR99_READOUT_R1.md), a causal Transformer perturbation implementation and ten-fit design with zero measured fits/results. Earlier scouts' no-new-push wording is dated. PR49/54 released artifacts were recovered and audited; PR79 K16 shows paired gains on a different recipe but lacks downloadable weight/prediction payloads. [Current frontier](../research/campaign_2026-11-14/quality_r2/FRONTIER.md). Historical running snapshots follow.

## November 14: quality-first research resumed

Latest user steering prioritizes maximum quality and asks to inspect team GitHub pushes. [Public receipts R2](../research/campaign_2026-11-14/local_r1/PUBLIC_RECEIPTS_R2.json): final KD50 pair .66510 versus teammate fullmix .66543; KD90 single .66193 versus old KD50 single .66108. Three of five slots used in the user's snapshot. Recent receipts do not replace the separately dated historical team-best record. Read-only PR/release scouting runs alongside the existing bounded quality wave; no paid compute or automatic upload is added.

**Precision correction:** an independent audit plus saved optimizer state show that the legacy PR49 Muon constructor used BF16 Newton–Schulz while the model kernels were FP32. The incomplete confirmation and initial automatic delivery are stopped. Corrected source48776fba8 passes real-CUDA checks and a120-update full-recipe profile. [R2 amendment](../research/campaign_2026-11-14/quality_r2/PRECISION_AMENDMENT_R2.json) replaces the panel with six corrected-source cloud fits (.5/.9/.9+boot8 ×s3/s4), a local .9/1.0 seed3 pair and conditional final refits. Old results keep their scores with corrected precision identity; no new quality gain is inferred from this engineering repair.

The [quality wave](../research/campaign_2026-11-14/quality_r2/README.md) is active. Strict KD90/KD100 seed3 and four fixed prediction exports completed; neither pure teacher nor weight averaging earned confirmation. [Teacher-pool R3](../research/campaign_2026-11-14/quality_r2/teacher_pool_r3/README.md) reuses original clean PR49 FBC/TLead tensors, with16 ancestor memberships audited and bitwise tensor preservation. Its fixed target grade is running; prospective3 new matched student fits reuse strict KD90 seed3 and remain conditional on score/review/headroom. The [allocation](../research/campaign_2026-11-14/quality_r2/teacher_pool_r3/ALLOCATION_R3.json) preserves8local GPU-hours/8cloud device-hours and maximum2 final refits, with a minimum-one-final reserve for this different continuation. The six private Kaggle cells remain queued without execution logs. Old automatic R1 delivery remains stopped.

The same-member-count TL3C replacement produced only a +.0000275 proxy delta. Its actual native replay was [deferred for thermal opportunity cost](../research/campaign_2026-11-14/quality_r2/original_fbc_r1/THERMAL_DEFERRAL_R1.json); partial predictions are ungraded and that ZIP is not recommended as an improvement. [Thermal response](../research/campaign_2026-11-14/quality_r2/THERMAL_RESPONSE_R4.json) now controls owned GPU/native work together; active fitting is capped at150% CPU and the hot native job was stopped. No paid compute or automatic competition upload. [Live frontier](../research/campaign_2026-11-14/quality_r2/FRONTIER.md). Earlier idle/paused snapshots below are history.

## November 14: local queue complete, submission collection

All 18 fits from the November 14 queue completed; GPU is idle. User requested status and best submission ZIPs. Fixed train-only kd50 seed pair scores 0.691423 on exposed official valid (+0.000949 over mean single); alpha .9 single scores 0.691442 (one seed). KD improves both seeds; since×KD interaction changes sign, so synergy is not confirmed. Two final-data kd50 models are available for an equal pair and single; validation was fitted and provides parity only. Public kd50 train-only single: 0.66108, 15m27s, user portal receipt 1qn2bx0q. [Handoff/checks](../research/campaign_2026-11-14/local_r1/SUBMISSIONS_R1.md), [campaign](../research/campaign_2026-11-14/README.md), shared draft PR #98. This collection starts no new fits or uploads.

## October 29: paused by user, teacher-track pilots landing

TabM-mini GRU, NorMuon and column view masks screen-failed (12 fits; blend law holds). Validation audit: official valid is a regime-shifted block (adversarial AUC 0.984, multivariate drivers), replicating E-094; protocol in [robust validation](../research/unified/methods/ROBUST_VALIDATION_20261029_R1.md). Teacher track under train-only/full-valid protocol: parent P 0.68671, labels-only child C0 0.67562 (overfits); PR49 teachers reproduced locally; D1/D2 distillation pilots landing. [Campaign](../research/campaign_2026-10-29/README.md), [pause receipt](../research/campaign_2026-10-29/PAUSE_R1.json). No new submission.

## October 25 evening: session wrapped, two jobs landing (user pause)

The user ended the session: research loop deleted, no new admissions. Landing: Muon on the production stage-6 student (local, auto-graded) and the Kaggle R3 independent reproduction (collect manually). Full results, ledger, incidents, new code, handoff commands and ranked next steps: [session synthesis](../research/campaign_2026-10-25/SESSION_SYNTHESIS_R1.md), [pause receipt](../research/campaign_2026-10-25/PAUSE_R1.json). Strongest result: strict-FP32 Muon beats LR-tuned AdamW by +0.0026/+0.0035 on the fast recipe, via orthogonalized recurrent updates that prevent gate collapse. No new submission.

## October 25 (afternoon): Muon mechanism established on the fast recipe

- Strict-FP32 Muon beats LR-tuned AdamW (5e-4) by +0.0026/+0.0035 (F0/F2) and AdamW 1e-3 by +0.0038/+0.0043; removing only the Newton–Schulz orthogonalization returns it to AdamW level (−0.0040 both cells). Mechanism: spectrally equalized recurrent updates prevent AdamW's gate collapse (45–48% saturated update gates, frozen units with τ q99 ~5e5 rows vs Muon 3%, ~170 rows). Chrono long-timescale init hurts (−0.0057). Clipping at 1.0 helps (no-clip −0.0025). [Mechanism panel](../research/campaign_2026-10-25/mechanism_panel_r1/README.md), [diagnostics](../research/campaign_2026-10-25/optimizer_panel_r1/diagnostics/README.md).
- **Policy finding:** torch 2.14 ran cuDNN GRU recurrences in TF32 in every local/B300 'strict FP32' fit (Kaggle T4 runs were true FP32); new `--strict-fp32-kernels` costs no measurable quality (−0.0011/−0.0001) and should be used in new recipes.
- Pending: Kaggle independent reproduction (fresh seeds, T4), saturation panel (spectral cap, Muon larger step, Muon H512×2), AdamW 2.5e-4, Muon on the production stage-6 student. Ops: cpptools capped (5.9 GB → 31 MB), RAM guard, 92 °C thermal guards after three laptop restarts.

## October 25: loss review, class-3 reframe and GPU weighting screen

User session (resources 5 GB VRAM / 30 GB RAM / 10 cores; "utilize gpu, experiment!"; 30-min research loop). Index: [loss review](../research/campaign_2026-10-25/loss_ideation_r1/README.md), [class-3 reframe](../research/campaign_2026-10-25/class3_reframe_r1/README.md), [class-3 weighting screen](../research/campaign_2026-10-25/class3_public_mix_r1/README.md).

- **Loss family diagnosis:** |y|-weighted MSE is already Bayes-aligned with pooled WP, so distance-function changes (NLL, Pearson, Huber, KD variants) cannot move the target; the NLL test was clipping-confounded. Row-measure weighting and distillation are the families that worked. Per-target normalization is a measured no-op (t0 mass share 0.496). External Grok/OpenCode red team claims verified/rejected in the synthesis.
- **PR32 teammate B300 Transformers memorize:** scored-valid weighted variance is 1.80, so val WMSE ~1.55 is weighted R² ~0.14 (train ~0.995). 1.54B val WP fell 0.37→0.27 (F0) epochs 15→21; teammate stopped the fits, epoch-15 checkpoints/predictions retained. Blend-probe predictions and a protocol redesign were requested on PR32.
- **Zero-fit gates:** label change at book events is trivially predictable (AUC 1.0, quote arithmetic) — the hidden quantity is F relative to quotes. A cross-fitted residual-information audit finds causal depletion/refill/rate/i1-lead features add no transferable information beyond the strong GRU (Δ −0.004 all / −0.012 class 3).
- **GPU:** class-3 weight 1.43 (matched to the E-298 public class mix) in the warm LR3e-5 continuation is null on F0 (−0.00006 official, −0.00009 public-mix). The more sensitive matched strict-FP32 6-epoch LR1e-4 pair is also negative (−0.00022 [−0.00046, −0.00002]; class 3 not improved): class-3 up-weighting is closed for the student lineage. The cost-deferred [dynamics-aux four-fit panel](../research/campaign_2026-10-25/dynamics_panel_r1/README.md) ran: aux alone F0 +0.00046 / F2 −0.00016 (mean +0.00015, inconclusive); consistent gains only in rows 99–500 (+0.014/+0.004), i.e. faster post-reset state formation. No new submission; no GPU arm currently has a supported >0.001 hypothesis.

## October 24 evening: independent value review and own-optimizer research

User-requested evening session (continues until explicit stop; resource ceiling raised to **10 CPU cores / 5 GB VRAM / 38 GB RAM**, superseding the earlier one-core/1-GiB operational cap after two laptop freeze/reboots — the strict process-control posture and its gates are recorded in the [operations note](../research/unified/operations/SESSION_FREEZES_STRICT_PROCESS_CONTROL_20261024_R1.md)). Zero competition-data fits, zero uploads.

- **Independent value/blind-spot review** ([folder](../research/unified/reviews/independent_value_review_20261024_r1/README.md), [synthesis](../research/unified/reviews/independent_value_review_20261024_r1/SYNTHESIS_R1.md)): three fresh-context reviewers (endgame/value, elephants red team, questions/literature). Convergent findings: final ranking privately re-scores previously **submitted** solutions (the two unuploaded quota ZIPs have zero placement value — upload them with a pre-declared rule); +0.0003-scale effects sit under the measured 0.0007–0.001 seed-noise floor; TL3's score mass lives in the two FTT/tiedlead members (student slot worth ≈+0.0006 publicly) and the FTT training source plus 52 lead features are the missing ask; PR32 teacher consumption is the only +0.005-scale lever (freeze the distillation packet before artifacts land); public 0.700 is not reachable by Nov 15 via blending. A 10-action endgame plan with Nov 8/13/15 milestones and 20 ranked questions are in the synthesis; seven new hypotheses recorded in the append-only ledger.
- **Own optimizer (Muon-inspired)** ([draft folder](../research/experiment_drafts/own-optimizer-muon-20261024/README.md)): the proposal established our `MuonAdamW` already implements Kimi-style RMS scaling and that its Newton-Schulz **bf16 CUDA cast violated strict FP32 in every historical GPU Muon fit** (now fixed; ruff/structure/14 optimizer tests pass). The synthetic 12-fit screen (strict one-core gated runs) found full-Muon −0.0049 paired vs AdamW, hh-only hybrid at parity, two-stage NS recovering ~⅓; the zero-fit C5 diagnostic on the sha256-verified matched D02 checkpoints shows Muon raises `weight_hh` stable rank 0.069→0.41 **while losing that screen by −0.0054** — the rank mechanism is real but does not convert to quality at small-screen scale. Best remaining local candidate: the C4 batch-mass screen (`loss_norm="running"` branch, coded, never graded).

## October 24 collection close

The bounded cycle has finished its scientific work and closed admission. One requested additional Astra/ultra reframe, all38 source PR inspections, both new diagnostics and two exploratory ZIP handoffs are preserved in [draft PR #41](https://github.com/VladKochetov007/WunderFunder/pull/41). The notebook server is stopped; app and ten figure families remain saved. No new scientific fits or competition upload occurred. Follow the [terminal record](../research/campaign_2026-10-24/resume4h_r1/TERMINAL_R1.json); earlier active wording below is the execution history, not a renewed allocation.

## Active research resume — October 24, four-hour allocation

The user explicitly resumed gh/Kaggle/local research until **14:38:12 UTC**, with new admission closing at14:23:12 UTC. The [contract](../research/campaign_2026-10-24/resume4h_r1/RESUME_R1.json) supersedes the historical stop below. The user allowed ten aggregate CPU cores; the operational cap is one after the thermal report, with1GiBaggregate RAM, no swap and the old production reserve preserved. Follow the [frontier](../research/campaign_2026-10-24/resume4h_r1/FRONTIER.md) and [comprehensive synthesis](../research/campaign_2026-10-24/resume4h_r1/CAMPAIGN_SYNTHESIS_R1.md).

- **Research audit:** all38 PRs inspected, remote heads fetched, and PR32/PR33 engineering archives recovered and verified. New teammate shortscreen results remain reported evidence where raw artifacts are unavailable. No new trained large-teacher endpoint was recovered.
- **Actual diagnostics:** saved F0 online heads all trail their parent. A fixed additional0.1 correction gains+0.000131368 for mean-only A inF0 but is essentially flat at−0.000004696 inF1. The [direction probe](../research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/RESULTS_R2.md) narrows ordinary calibration as the explanation and finds B/C largely share A's direction. Independent point/projection review passes; independent bootstrap reconstruction remains incomplete.
- **Fundamentals:** the explicitly requested one-Astra/ultra review is complete, with seven deduplicated cards and a prospective four-total-fit equal-capacity Z/A screen. The [fitting-weight audit](../research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/RESULTS_R1.md) measures roughly0.1×–9.7×batch reweighting and less concentrated component weights; this does not prove training harm. No new scientific fits ran; the old twelve-fit panel remains incomplete.
- **Submissions:** [two TL3/H390 ZIPs](../research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/RESULTS_AND_HANDOFF_R1.md) pass exact native replay and independent fixed-grade review. Pair-versus-TL3 deltas are+.003742/F0 and+.000205/F2; S3-only deltas+.003513/F0 and−.000029/F2. Both were delivered locally, pair first. H390 used all supplied data and retained TL3 ancestry is unknown: these are fitted sample diagnostics, with no public-gain forecast or upload.
- **Delivery and EDA:** the repaired Kaggle CPU profile times out after855.52seconds without a usable profile; engineering update count/cache completion remain unknown. No GPU retry is admitted. The [live marimo notebook](../research/campaign_2026-10-24/resume4h_r1/eda/README.md) has ten exported figure families, including rare boundary censoring, sampled event clocks, F0/F1 comparisons and both new diagnostics.

Evidence publication is being prepared on a separate sparse branch from research-team. The original source checkout and unrelated changes remain untouched. Older stop/resume entries below are historical.

## Active user stop — October23

The user explicitly orders all experiments stopped/killed. [Stop receipt](../research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json) confirms local research slices stopped and runtime-masked, the remaining collector timer disabled, and no running helpers or matching local workers. Kaggle R2 is terminal ERROR; R3 was never launched. No new submission. Do not resume any research or publication work without an explicit user resume.

## October23: requested independent research review complete; experiments remain stopped

The user authorized ten fresh-context Astra/ultra ideators to examine missing or misleading ideas and save their findings. The [review packet](../research/unified/reviews/astra_ultra_audit_20261023_r1/README.md) contains ten separate reports, a [comprehensive synthesis](../research/unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md), forty hypothesis cards and seventy questions. Root checked historical receipts, corrected a stale projected-mean recommendation and an insufficient permutation control, and recorded all proposals in the append-only ledger. This is source/literature/document work only: zero fits, evaluation queries, profiles, imports/tests, exports or provider launches; no experiment resume, new score or submission. Older allocation/resume entries below remain historical.

## October23: reduce local load after another freeze warning

Local research now uses at most one quarter CPU core, 1 GiB hard RAM and zero swap; full checkouts and local compute are halted. A helper's lingering Git checkout and child were stopped. The [resource correction and remote repair](../research/experiment_drafts/b300-fast-local-kaggle-20261023/RESOURCE_AND_DELIVERY_AMENDMENT_R2.json) charges this incident within the unchanged allocation and prepares only a tiny source update for a final Kaggle CPU profile. No scientific age fits or new submission exist; strongest confirmed public package remains original TL3 at 0.66832.

## October23: first remote age profile fails on an incomplete import bundle

The [portable Kaggle CPU attempt](../research/experiment_drafts/b300-fast-local-kaggle-20261023/README.md) verifies its uploaded archive and declared source pins, then stops before cache reconstruction because `wnn.i1_history` was omitted from the transitive dependency bundle. The remote worker uses 36.874 seconds, 353 MB peak RSS and zero swap; cleanup succeeds. Root owns a bounded source-closure repair and isolated import check before retry. No scientific fit, score or new submission exists. Local disk staging and private upload are complete; the 7,200-second screen and 12-hour production reserve remain unchanged.

## October23: local heat defers the age profile; portable Kaggle CPU delivery under preparation

The [age-screen implementation and independent grader](../research/experiment_drafts/b300-fast-local-kaggle-20261023/README.md) pass their correctness checks. No local profile child started because the laptop stayed above the cooling gate. The [Kaggle CPU amendment](../research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_CPU_DELIVERY_AMENDMENT_R1.json) keeps the six scientific fits and original recipe unchanged while preparing a small suffix transport and exact reconstructed-cache validation. Engineering allowance increases within the existing7,200seconds; no extra total allocation or production spending. No new quality result or submission yet.

## October23: faster B300 ideas selected for local/Kaggle research

The latest user steering authorizes a bounded [field-age readout screen](../research/experiment_drafts/b300-fast-local-kaggle-20261023/README.md): six small heads on two exposed fold/seed cells, zero new GRU or teacher fits. Up to 7,200 inclusive seconds are reserved from unused strong-parent allocation; the 30-hour total and 12-hour production reserve remain fixed. All new storage uses physical disk outside `/tmp` and `/dev/shm`. Reused cache bytes and headers pass verification; runtime admission is pending. Kaggle CPU grading is prepared because fresh unreserved GPU quota is only 1,635.575 seconds. No new quality result or submission yet; the earlier incomplete online panel is unchanged.

## October23: online worker cap reached; incomplete panel

The [terminal receipt](../research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json) records timeout at the original deadline, 03:10:32 Kyiv. Seven of twelve head exports are saved; one head stopped after two unsaved epochs and four never started. Independent metadata review accepts seven exact NPY headers and all frozen source pins, but full-panel grading stays closed. No target-quality result or native release. The proposed one-hour transfer remains unanswered/unapproved; no retry or reallocation occurred. Both parent caches and completed heads are retained, the CPU observer has exited, and the 12-hour production reserve is protected. This is an execution-cost outcome, not evidence against online learning. User research-resume authority remains effective; further spending on this path needs the pending allocation decision.

## October23: online-learning screen admitted and running

Both complete parent caches are published and the twelve-head fitting command is active. The [CPU safety correction](../research/experiment_drafts/online-relationships-20261022/CPU_THERMAL_CORRECTION_R1.json) reduces the worker to one CPU core; a [single-invocation observer](../research/experiment_drafts/online-relationships-20261022/CPU_OBSERVER_EXECUTION_R1.json) adds 97°C/90°C cooling with zero task swap. The original 7,800-second service deadline (03:10:31 Kyiv) and 9,000-second inclusive ceiling remain binding. The [updated timing review](../research/experiment_drafts/online-relationships-20261022/ROLLING_COST_REVIEW_R2.json) projects beyond that ceiling; the proposed one-hour transfer from unused strong-parent allocation is unanswered and unapproved. No partial-panel quality grading or native release is authorized by these operational findings.

The [full twelve-head panel](../research/experiment_drafts/online-relationships-20261022/README.md) passes an independent8,311.440-second inclusive cost gate against9,000seconds. Immutable recipe/source/grader pins are frozen; CPU-only full parent caches and then all12fixedCP2heads are running under two-core/5GBaggregate/noSwap constraints. No scientific quality result yet. Remaining3warm-historyfits are [deferred intact](../research/campaign_2026-10-21/next_cycle_r1/WARM_REMAINDER_COST_DEFERRAL_R1.json), completedcontrol/export preserved, production12hreserve untouched. Fresh reframecards and an independent critique are recorded separately with no extra fits.

## October23: postboot research resume

The explicit user resume supersedes the reboot pause, retaining the original budget. Disk /tmp and SSD swap are verified; zram is absent. The boot-reverted20GB research cap is repaired persistently to5GB/twoCPUs/noSwap before execution. Reviewed B8 online cost profiling and separate fresh-context reframing resume; no new scientific fit admitted yet. [Resume](../research/campaign_2026-10-21/next_cycle_r1/POSTBOOT_RESUME_R1.json), [contract](../research/campaign_2026-10-21/next_cycle_r1/REFRAME_CONTRACT_R1.json).

## October22: research paused for reboot

The latest user stop supersedes the resume. [Pause handoff](../research/unified/operations/REBOOT_PAUSE_20261022_R2.md): no active fit/export/profile, completed F0 CP1 and full CPU prediction export preserved. Research observers/timers are stopped and disabled across reboot. Online R2 costs exceed the ceiling; B8 R3 source is reviewed but its profile never started. Zero online scientific fits; remaining three warm fits unadmitted. Explicit user resume required; production reserve remains 12 hours.

## October22: thermal recovery and compiled online cost profile

The [resume operation](../research/unified/operations/THERMAL_RESUME_20261022_R1.md) preserves the completed warm F0 control and stops its thermally stalled GPU export. The same fixed CP1 now exports on CPU, with GPU hidden and a one-core/4 GB/no-swap bound; full coverage is pending. Remaining three warm fits still need a safe complete-panel projection. Exact compiled online features reduce the one-pass cost projection to 523.40 seconds; full parent-cache/head/grade profiling is pending, with zero online fits admitted. The earlier Python deferral and thermal pause remain history. Production reserve stays 12 hours.

## October22: user resumes experiments after thermal pause

[Resume receipt](../research/campaign_2026-10-21/next_cycle_r1/RESUME_AFTER_THERMAL_R1.json) supersedes priorpause. F0warmcontrol completed2epochs/14920finiteupdates andsavedfixedCP1; rootfirstexportsitwiththermal/memoryguardanddiskTMPDIR. Remaining3warmfits require refreshedcomplete-panelcostadmission withinremaining8hstage. SSDswapactive, tasknoSwap; production12hreserve preserved. Optionalcompiledonlinefeatureengineering hasnoqualityfitadmission.

## October22 user-directed thermal pause after current fit

The user reports laptop freezes/overheating and asks to let the current fit land gracefully, then pause. [Drain receipt](../research/campaign_2026-10-21/next_cycle_r1/THERMAL_GRACEFUL_PAUSE_R1.json): only currentF0warmcontrol drains at10%CPU; its supervisor is suspended and cannot start further fits/exports. A bounded observer records checkpoint hashes afterexit and releasesworker resources. Completion is pending; no automaticresume or additionalheavywork. Researchtask swap is0, buthostswap is7.9/8GB. The online12fitpanel was already deferred by its4.394h featurecostprojection against2.5hcap.

## October22: two-parent public receipt and online amendment

The user reports [two-parent portal0r4h74xw](../research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json):0.66771,28m31s,643MB. This is−0.00061 versus reportedTL3 0.66832 but10m41s faster, and+0.00104 versus the H512 replacement. The exposed local sample advantage did not transfer in this descriptive comparison; TL3 remains the stronger public reference. Portal payload hash remains unverified. The unchanged warm-history panel continues. The [online input-relationship amendment](../research/unified/ideas/ONLINE_RELATIONSHIPS_20261022_R1.md) is now deferred intact: user-requested backward lags1and16 require12readout fits, but measured one-pass featuregeneration projects4.394h against the unchanged2.5h inclusive ceiling, before fitting/export. Core/profile verified, no onlinefit admitted or productionreserve spent.

## October22: immediate exploratory quota package and warm follow-up preparation

The user explicitly requested one submission now for public evidence. The [fixed two-parent TL3 package](../submissions/tl3_two_parents_20261022/README.md) is technically checked and independently reviewed:8.90MB, unchanged native libraries, two teachers at0.5 each, no new fits. Its full-fold quality is unmeasured; the subsequent fixed0.5/0.5 sample grade gains+0.00257354 over sourceTL3 on eight adaptively exposed components, both targets positive. This is ancestry-qualified diagnostic evidence, not a public forecast. No authenticated portal session is available, so it is handed off for human upload; no quota use, portal ID or public score is claimed. This narrow exploratory exception does not promote a champion.

Both real-parent profiles pass1,100finiteupdates with zeroOOM/skips (F0median2.604s/B16, F1median2.376s). Both CPUparent-output gates pass exactparity with sharednormalization/sampler identities. After preserving the metadata-bound and expected-order failures and explicit narrow amendments, the complete four-fit strong-parent history panel is now running: two matched control/history cells, two epochs fixedCP1, full1873valid exports and finalpairedgrade. Inclusive projection5.996h within8h; oneCPU/4GB/noSwap/oneGPUworker, with no quality or production-promotion claim yet. See the [live frontier](../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md). Earlier snapshots below are history.

## October 22: complete i1 screen and saved-model recovery

The six-fit i1 panel is complete. The fixed 90% B / 10% C mixture improves F0/1001 by +0.00035953 and F2/1002 by +0.00027145, averaging +0.00031549. Both targets improve, while C alone loses in both cells. Conditional component 95% intervals are [−0.00001187,+0.00070893] and [+0.00008223,+0.00043398]. A's fixed mixture also helps F2, so complementarity does not uniquely establish a temporal-memory mechanism. The [independent review](../research/unified/reviews/I1_SIX_FIT_HISTORY_GRADE_REVIEW_20261022_R1.md) supports the existing conditional four-fit strong-parent comparison, with initial-function parity, shared normalization and native-serving checks. All folds remain exposed; no stronger complete submission is established.

Saved-artifact recovery is complete without new fits. The [fixed checkpoint mean](../research/campaign_2026-10-21/next_cycle_r1/WEIGHT_MEAN_PROJECTED_GRADE_R1.json) adds only +0.00001096 over the existing oldZ4/WMSE mixture. The [recovered Kaggle CP5/SWA grade](../research/campaign_2026-10-21/next_cycle_r1/B16_LOCAL_FIXED_GRADE_R1.json) shows both fixed mixtures lose about 0.000983 versus their historical FP16 reference; ancestry and missing initial-digest limits remain explicit. No recovered model is promoted.

The user requests continuation until explicitly stopped, within the accepted 30-hour allocation. The corrected dynamics profile passed 110 batches and 1,100 finite updates under 4 GB RAM, no swap and one GPU. Its [inclusive cost projection](../research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_PROFILE_RESULT_R2.json) is 3.366 hours, so the entire four-fit panel is deferred under the three-hour ceiling; no target-quality comparison ran. Warm-history tests pass 12/12 on root and publication sources; the narrow historical output-head restoration also passes all 17 combined warm/head tests on root and publication sources. Actual-parent parity and sustained warm profiles remain before fits. Broader compatibility checks have one recorded preexisting class-monkeypatch failure. The [frontier](../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md) tracks subsequent execution; older snapshots below are history.

## October 22: fixed-history geometry and native precision diagnosis

The first i1 cell’s fixed 90%B/10%C gain is +0.00036506 before clipping and +0.00035953 with official clipping. The correction has positive residual covariance with both targets after accounting for B; clipping slightly reduces the gain. This is descriptive evidence of a complementary prediction direction on the same exposed F0 fits, not independent confirmation or proof that history information caused it. F2/B completed at 0.68410821; A is fitting and C remains pending. [Geometry and limits](../research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/README.md).

[Eight-component precision diagnosis](../research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json) passes same-host original and CP5 reconstruction before targets. The saved FP32 counterfactual improves the complete CP5 endpoint by only +0.00004035 and still trails TL3 by 0.00328280 on this sample. Precision does not explain that sample gap; no full-fold or public extrapolation follows. No new model was fitted.

The dynamics target numerical repair passes six oracle/causality/export tests and six checker tests on merged source; a clean corrected checkout is frozen. Its four-fit panel still requires a sustained full-cost profile. Saved mean and B16 exports remain next after the single active GPU panel. Corrected i1 worker OOM counters remain zero. [Live frontier](../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md).

## October22 00:06 Kyiv: first complete i1 cell and recovered endpoint diagnosis

The accepted30-local-GPU-hour cycle is active. The first F0/seed1001 i1 cell scores B0.65907416, A0.64317272, C0.65553705. History C loses standalone (−0.00353711), but the predeclared90%B/10%C blend gains+0.00035953 on both targets; its conditional component95%interval [−0.00001187,+0.00070893] leaves repeatability unresolved. F2/seed1002 is running under the unchanged six-fit panel; no new OOM events in the corrected worker. [Full interim grade](../research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json).

The [same-runtime native grade](../research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json) leaves CP5 replacement0.00736292 below TL3; the host-versus-legacy score difference is only−0.00000059. Runtime mismatch does not explain this gap, while unequal training exposure/objectives prevent an architecture ranking. TL3 remains an exposed ancestry-qualified diagnostic, not clean CV. The [fixed projected checkpoint mean](../research/campaign_2026-10-21/next_cycle_r1/WEIGHT_MEAN_PROJECTED_DERIVATION_R1.json) passes exact output parity and normalization checks; full-fold quality is pending localGPU availability.

The B16 F1 cloud export hit its700-second limit before observed GPU use, with no prediction arrays or OOM. [Terminal evidence](../research/campaign_2026-10-21/next_cycle_r1/B16_CLOUD_TERMINAL_COLLECTION_R1.json) and [fresh quota](../research/campaign_2026-10-21/next_cycle_r1/KAGGLE_QUOTA_R6.json) are preserved:1635.575seconds unreserved, refreshOctober24. No further provider launch is admitted; root will recover both saved exports locally after the current panel. No new submission is promoted. [Frontier](../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md) owns subsequent state; older snapshots below are history.

## October21: new 30-hour local/Kaggle cycle

The user accepted and requested execution of the [new cycle](plans/2026-10-21-next-local-kaggle-cycle.md): 30 new local GPU-hours plus verified available Kaggle, B300 deferred. Start with saved-artifact recovery and the six-fit i1 screen; dynamics has four fits and strong-parent/production spending is conditional. [Live frontier](../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md). Historical costs and stops below remain history.


## Matched B300 endpoint replication correction

Local H512 F0/seed1003/CP5 reproduces exact historic PR25 score within1.28e−9 (0.6737706291771555 vs0.6737706304503658). Earlier non-replication chat conclusion confused a four-fold mean with this single fold. [Audit](../research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/README.md#correction-matched-b300-endpoint-successfully-reproduced). Postfit grading succeeded; existing oldZ4/WMSE still beats fixed oldZ4/H512. Native replay stopped at coverage guard; no accepted new submission.

## October21 15:21Kyiv: fit completed, postfit recovery active

Local H512 six-epoch fit succeeded at13:29Kyiv. The original launcher failed on Linux page-rounding of the5GBcap; a separate recovery now passes acceptance, has derived SWA and is exporting predictions. Native endpoint waiter is rebound to the exact recovery invocation. See the [recovery packet](../research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/README.md#completed-fit-and-page-alignment-recovery-october21). No validation score or new ZIP yet; earlier ETA missed. Kaggle weights remain collected after a separate producer/validator digest mismatch.

## October21 monitoring and Kaggle terminal recovery

Local H512 continues its fifth of six epochs; automatic postfit/native evaluation waiters remain active under5GBRAM/twoCPU/noSwap. Kaggle B16 completed training and saved checkpoints but failed the postfit initial-parameter identity gate before exports. See the [terminal packet](../research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/README.md#version2-terminal-fit-saved-postfit-provenance-gate-failed). Root is investigating metadata-only recovery; no new fit or provider launch. No new quality grade/submission yet. Older RUNNING snapshots below are historical.

## October21 actual execution at01:13UTC

Local H512 completed its first epoch (14,920 finite-gradient updates, no skips) and continues epoch two of six. Corrected [Kaggle kernel version2](../research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/README.md#corrected-version2-running) is RUNNING and reached input validation after both uploaded files were remotely hash-verified. Version1 failed before profiling due to an omitted manifest; its evidence and separate transport-repair admission are preserved. No Kaggle profile/fit result or new submission quality is established.

## October21 B16 conditional admission

The [separate B16 F1 packet](../research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/README.md) passed independent source/recipe review and root refreshed quota (17,006.922seconds,zero reserved). One profile-first conditional sixepochfit is admitted; exact bundle verification remains beforeupload. B16means44,760updates, not a matched B8replication. No R2model/quality result yet; the healthy local F0B8fit continues unchanged.

## October21 Kaggle profile terminal update

The exactF1B8profile succeeded40finiteupdates, then intentionally skipped the sixepoch fit on measured cost:19,429.399projectedseconds exceeds16,000cap. ProviderERROR reflects missing unproduced fitoutputs; no qualityfit or negativequalityevidence. See [terminal interpretation](../research/campaign_2026-10-21/kaggle_h512_conditional_f1_r1/TERMINAL_INTERPRETATION_R1.json). Root is preparing explicitB16amendment, not admitted. LocalF0fit and automaticpostfit/nativewaiters continue unchanged. Older RUNNINGcloudsnapshot below is historical.

## October21 Kaggle launch update

One clean-source conditional F1 attempt is now admitted and provider-confirmed RUNNING: [execution packet](../research/campaign_2026-10-21/kaggle_h512_conditional_f1_r1/README.md). Private dataset v1/R2 bundle hashes verified; kernel v1 profiles first and fits only if measured cost fits refreshed quota and16,000second cap. No profile/fit progress or quality result yet. Local H512 F0 continues healthy; postfit and native endpoint waiters remain active. Earlier no-third-attempt-admitted statements below preserve prelaunch history.

## October21 sustained execution update

The same H512F0/seed1003 six-epoch fit remains active;89°Cthermal throttling reduced sustained throughput to~1.9sequences/s. Its [runtime-only amendment](../research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/THERMAL_BUDGET_AMENDMENT_R2.json) allows12h plus reserved25minpostfit within15hcumulative; no extra production fit. Postfit seven-step automation and fixed actual-TL3nativeendpoint evaluation are prepared, not executed. Engineering17.09MBZIP/resetchecks pass using weak fixture weights; never submit that fixture. KaggleR1/R2 both failed before training (R2missingstructuralcaches after raw hash verification);4.79935providerhours remain, no third attempt admitted. See the frontier and live unit; historical statements below are superseded.


## Active H512 fit — October21

[The H512 delivery packet](../research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/README.md) records one admitted cold label-only H512×1 F0seed1003 fit, running since23:18:35UTC under `wnn-h512-f0-fit-r1-20261021.service`. Six epochs, fixedCP5; SWA345 and fixedblends declared before outcomes. Core source/recipe static review passed and actual profile measured1.893s/batch. Full-data production remains conditional after actual scoring and budget reassessment. No new ZIP. Local15hbudget cumulative, parent5GBRAM/twoCPU/noSwap now enforced. Older no-running-job statements below are historical.

## October21 measured continuation update

Both matched WMSE/NLL fits and seven-endpoint scoring completed. The [actual results](../research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/RESULTS_F0_R1.md) favor retaining the fixed oldZ4/WMSE blend for further evaluation, without a demonstrated TL3/public improvement. NLL has no matched point advantage. The PR25-supported [H512 local profile](../research/campaign_2026-10-20/submission_delivery_r1/h512_replication_profile/README.md) completed safely at1.893s/batch; root is freezing exact fold/production commands before admitting a quality fit. Existing15hbudget remains cumulative; no GPU job is currently running. Historical running snapshots below are superseded.

## Active local delivery — October20

The latest user instruction chooses local small-GRU training after remote
Beleriand weights could not be retrieved. The [delivery packet](../research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/README.md)
records two matched H256×2 continuations, WMSE versus Gaussian NLL, using local
parent weights and cached teacher targets. Both bounded profiles passed; the
two-fit sequence started at20:33:35UTC and is running. No new quality result or
submission is claimed. Preserve the15localGPU-hour budget, oneGPUworker,
twoaggregateCPUcores,5GBtaskRAM/VRAM and zero swap. The original i1 comparison
remains deferred. Read the frontier and live service before admitting work.

## Latest amendment — teacher quality and scaling

The user now prioritizes progressively smarter teachers; distillation is optional,
not a teacher-promotion requirement. The [scaling programme](../research/unified/ideas/TEACHER_QUALITY_SCALING_20261019_R1.md)
proposes a six-fit three-size ladder, matched learning curves and ranked tuning.
This explicitly amends the earlier strategy; existing four-fit JSON recipes
remain historical proposals until a receiver freezes an amended runnable panel.
No fits/profiles or new compute allocation are claimed.

## Latest steering — October 19: fast screens and large B300 teachers

The user replaces mandatory30-per-arm experiment matrices for FUTURE drafts
with **4–6 total new fits per initial idea comparison**, normally2–3 matched
two-arm cells. Count cached controls and teacher/student stages explicitly;
retain historical frozen panels and expose small-screen uncertainty. The new
strategic priority is a large causal Transformer teacher on8 B300s, followed by
GRU distillation. See [ADR0043](adr/0043-fast-screens-and-transformer-teacher-scale.md)
and the [detailed design](../research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md).
The design proposes0.68B/1.65B capacity recipes, optional3.17B extension, strict
FP32 Muon, WSD, profiled DDP and causal dynamics objectives. It supplies no
new runnable DDP trainer, B300 reservation, fit or quality claim. Preserve the
local/Kaggle plan below and its start/allocation; do not restart stopped jobs.
The later user-supplied TradeFM paper motivates a fixed-scale event-likelihood
screen before the full ladder; its four fits and the capacity panel are separate.
[Draft PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31) contains the
source-bound design handoff and analytical tools; no runnable teacher fit exists.

## Active instruction — October 19, 2026

The user accepted the [plateau-review local/Kaggle plan](plans/2026-10-20-plateau-review-local-kaggle.md),
requested implementation, and asked that it survive context compaction. This is
the canonical plan; the [campaign frontier](../research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
records actual progress and the next action. Preparation is active. Model fits
start October 20 Kyiv; no scheduled or running job is claimed at this stage.

Allocation: **15 local GPU-hours**, **all verified available Kaggle quota in
parallel**, no paid compute, one local GPU worker, two aggregate CPU cores,
5,000,000,000 bytes task RAM, 5 GB VRAM and no swap. The local/Kaggle cycle has
**no mandatory 30-cell matrix**; full fitting remains eligible within budget.
The first comparison is cold, equal-width i0-only / standard / causal i1-history
GRUs on matched components. The imported review's larger grids are deferred.
All existing folds are exposed. Code/recipe preparation is not a scientific result.

The first-comparison implementation is integrated on main as `0cc795070`: causal
i1 histories, post-normalization semantic masks and checkpoint/prediction support.
Ruff and training-structure checks pass, with an independent static review. Exact
commands and fixed fitting membership are prepared; numerical checks, source seal,
actual-shape profiles and execution remain pending. See the campaign frontier.

The October 19 B300 PR status review found 12 verified but partial width/depth
pairs and 16/30 geometry pairs per arm; field-age and innovation have no scored
paired cells. The tabdata jobs were explicitly stopped. See the
[PR evidence review](../research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
before repeating these comparisons.

This scoped authority succeeds the October 18 stop for this cycle. Keep its
receipts and unrelated campaign restrictions intact. Read the new spec before
interpreting the dated snapshots below as active instructions.

## Historical October 18 state and results

**Latest instruction, 2026-10-18 08:27 UTC:** the user stopped research admission and prohibited new jobs. The local durable stop barrier `15e691414aa14bb4a3d888b841d85ae3` drained without terminating healthy work. No new fits, scores, packages, uploads or PRs are admitted. One Beleriand F1/seed-1001 run was last observed active at 08:20:52 UTC; let it finish and collect terminal artifacts only, with no successor. The user reports both geometry ZIPs already submitted, so they are not new candidates. See the [stop receipt](../research/campaign_2026-10-18/online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json), [operations note](../research/unified/operations/RESEARCH_STOP_20261018_R1.md), and [nightly research closeout](../research/campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md).

**Snapshot date:** 18 October 2026. **Latest user scope:** deliver one newly trained exploratory ZIP by **02:50 Kyiv (23:50 UTC, 17 October)** and publish the larger study as an 8×B300 draft PR to `research-team`. The multi-day local R4 matrix is stopped and drained. The [completed quick pair](../research/campaign_2026-10-18/quick_geometry_probe_r1/README.md) scored feature 0.6804165 / zero control 0.6804784 / original parent 0.6812265 on one exposed F0; this short continuation did not improve. The [adapter-only followup](../research/campaign_2026-10-18/geometry_adapter_probe_r2/README.md) also completed negatively: 0.6804499040 versus parent 0.6812265132, conditional component interval for the difference [−0.00135282, −0.00027913]. Bitwise checkpoint checks confirm all old weights stayed fixed; 60 new columns learned nonzero weights. Both results are one exposed F0/seed, not robust/public evidence. Local fit admission is drained; no further local fits are planned. The [ready directional q16 quota ZIP](../research/campaign_2026-10-18/geometry_adapter_probe_r2/RELEASE.json) is 5,323,200 bytes (SHA955bff715…): native served F0 WP0.68045197, exact-ZIP Python3.11 and full-F0 replay pass, local row-cost projection40–45minutes with target runtime unmeasured. The predeclared Torch/native FP32 tolerance failed and is explicitly recorded; the actual native endpoint is evaluated directly. It is released under the user's deadline exception, not promoted as a demonstrated gain. The 8×B300 study is published as [draft PR #15](https://github.com/VladKochetov007/WunderFunder/pull/15), base `research-team`.

## Latest remaining-hour continuation

The user's subsequent instruction explicitly authorizes a few more variations before the unchanged02:50Kyiv deadline. The [R3 packet](../research/campaign_2026-10-18/geometry_variations_r3/README.md) admits two matched moments-only/dictionary-only frozen-parent fits after a successful actual-shape mask profile, plus a separately predeclared inference-only feature-correction scale grid. The existing ready q16 ZIP remains frozen. This supersedes the earlier no-further-fit statement only for these bounded experiments; no broader local matrix or new cloud spending is admitted. One selected endpoint may get a complete native F0 replay, based on actual633-second prior measurement. Exposed-F0/adaptive selection limitations remain explicit.

## Latest completed variation screen

All two R3 fits and four inference-only alpha exports completed and drained by23:25:55UTC. Independent grading passed all source/hash/alignment and matched-fit gates. The top point endpoint is R2 geometry columns scaled by−0.5: WP **0.6814221100**, versus original parent0.6812265132 and full R2 0.6804499040. Conditional component difference versus parent is+0.00019560,95%CI[−0.00002480,+0.00042143]; versus R2+0.00097221,CI[+0.00027186,+0.00176521]. These are adaptive exposed-F0/one-fit comparisons, not independent-seed/public evidence. Moments-only0.68086802 and dictionary-only0.68061687 both remain below parent. The selected q16 native replay passed: served WP0.6814234038,468groups/9.36Mrows,595.41s wall,820168KiBpeakRSS; exactZIP offlinePython3.11/reset/ABIchecks passed. [Release receipt](../research/campaign_2026-10-18/geometry_variations_r3/RELEASE.json) binds the5,323,608byte SHA27f01f53…ZIP. Callback-only38.09minute projection is not measuredcompetitionruntime. No further fits belong to the closed deadline packet; new continuousresearch authority below is separate.

The user-reported [R2 result](../research/campaign_2026-10-18/geometry_adapter_probe_r2/PUBLIC_RESULT.json) is **0.66428** (30m47s/626MB, ID `14s6l2z7`); [R3](../research/campaign_2026-10-18/geometry_variations_r3/PUBLIC_RESULT.json) is **0.66456** (30m32s/627MB, ID `1wmshmaz`). Both are below the reported TL3 best of **0.66832**; R3 is higher than R2 by 0.00028 but still trails TL3 by 0.00376. Portal outcomes are transfer evidence for these exact packages and do not calibrate new local variants into public forecasts.

## Historical continuous research authorization — closed 18 October

The earlier user instruction authorized autonomous overnight and longer research **until explicit pause**, including causal online learning, custom/learned features, previous-competition methods, TL3 complementarity and larger promising runnable B300 draft PRs. The [continuous campaign](../research/campaign_2026-10-18/online_features_r1/README.md) preserves that authority and its 2-CPU/5-GB task-RAM/one-GPU limits. The explicit 18 October stop supersedes its admission scope; the completed data audit, experiments, frozen sources, ZIPs and results remain evidence.

### Latest online-learning evidence

The [causal proxy gate screen](../research/campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RESULTS_R1.md) found no useful gain from this one pseudo-label gate: best of three timescales improved fixed 75/25 by +0.00004795 WP (32-component conditional interval [−0.00003098,+0.00012853]) and remained 0.0023302 below TL3 on the same repeatedly exposed F0 components. Target maturity is undocumented and TL3 ancestry remains unknown. Prediction reconstruction passed before labels/masks were read; three CPU-only attempts had zero OOM deltas. This narrows the exact proxy-gate recipe, not online learning broadly. No separate B300 draft was warranted for this gate. The independent evidence review and execution receipt are linked from the [unified validation index](../research/unified/validation/README.md).

The [causal Kalman input screen](../research/campaign_2026-10-18/online_features_r1/kalman_filter_probe/RESULTS_R2.md) remains an inconclusive positive fixed-GRU diagnostic (+0.000338 WP for local trend versus raw; 12-component conditional interval [−0.000019,+0.000716]). A separate trained-feature/GRU-student proposal already exists as [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17); it has no B300 profile or fit, and its teacher registry remains pending. The gate result does not validate or invalidate that study.

The completed [Kalman output-filter screen](../research/campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md) compared the same raw parent with fixed Q/R=0.25 and first-99-prediction-estimated Q/R on 468 exposed F0 groups/141 components. Scores were 0.6812265, 0.6755207 and 0.6647620 WP respectively; both filters lost in all four descriptive partitions with wholly negative paired component-bootstrap intervals. This rejects these two output-smoothing recipes on this endpoint, not Kalman-derived input/hidden-state features or online learning. Independent review and resource receipt are linked in the packet; no additional B300 draft was warranted, while the distinct PR #17 teacher/student direction remains open.

The [TL3/R3 fixed-mixture F2 replay](../research/campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/README.md) completed on 443 previously inspected groups/144 components. TL3 scored 0.723887; R3, fixed 75/25 and fixed 50/50 scored 0.715413, 0.722678 and 0.720950. Their paired conditional component intervals versus TL3 were all negative. The first scorer attempt failed after accessing one target/mask; the correction, amended analysis recipe and successful grade are separately recorded. F2 is exposed and TL3 ancestry is unknown, so this is robustness evidence for those exact endpoints, not clean CV or transfer. It establishes no new large direction; existing B300 handoffs #17/#18 remain open and no new draft was justified by this comparison.

The [early-to-late sequence preference diagnostic](../research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md) completed on 468 exposed F0 groups/141 components. The early label-informed TL3/R3 selector scored 0.693871 late, −0.003798 versus TL3 (conditional 95% component interval [−0.005690,−0.001453]) and −0.000587 versus fixed 75/25 (interval [−0.002360,+0.001586]). Observed early/late expert-choice agreement (68.16%) is below independent-marginal expectations (70.16%/68.64%). Independent source and result reviews pass. Both CPU runs used 2 GB reservations and had zero OOM deltas; the profile materialized the full compressed target/mask arrays while scoring only eight groups. This narrows the frozen label-informed router, not broader online learning. No new B300 draft follows; existing PR #17 remains a separate Kalman feature teacher/student proposal.

The [groupwise fast/slow Kalman state screen](../research/campaign_2026-10-18/online_features_r1/group_fast_slow_r1/README.md) then completed an OOF linear proxy on 32 previously exposed components. Raw+39 group states scored 0.255766 WP versus 0.268859 raw (delta −0.013093; 95% component interval [−0.033693,+0.005956]); the scalar clock was effectively tied (delta −0.000108). The group arm lost in all four inner folds and position bins. Since train lacks the official score mask, this is a nonofficial all-post-warmup train-row proxy, not a validation result. It defers the simple additive ridge encoding only; recurrent Kalman use remains open. Existing [B300 draft #14](https://github.com/VladKochetov007/WunderFunder/pull/14) was updated with this caveat; no separate large direction emerged.

The subsequent [matched recurrent screen](../research/campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_R1.md) compared active 39-state inputs against an equal-width zero control on one exposed F0 fold. Scores were 0.67986413 and 0.67981250 WP (Δ +0.00005163; conditional 141-component 95% interval [−0.00013143,+0.00027018]); position bins are mixed. This is a one-seed incremental fine-tune over a common trained parent, not an independent replication or long-horizon online-learning test. The [independent screen review](../research/unified/reviews/GROUPWISE_FAST_SLOW_RECURRENT_SCREEN_REVIEW_20261018_R1.md) found no scoring mismatch and recommends no new B300 draft from this result. Keep the family open; existing PR #17 covers the broader Kalman teacher/student direction.

## Operating state

The latest research closeout is dated 15 October 2026 at 18:39 UTC. It records
that the admitted G fit, export and grade finished, all admitted work drained,
and the research admission barrier remained closed. The latest user explicitly resumed the sequence-moment/heuristic investigation locally on main with a5GB aggregate task RAM cap. This supersedes that pause for this exact scope; unrelated campaign admission and the other agent's Kaggle ownership remain unchanged. Older plans, quotas and stop snapshots are historical authority only.
See the [pause closeout](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md).

The separately recorded October 17 research-reset scope resumes research thinking, literature, EDA and bug repair while excluding model training for that packet. The geometry-specific local fit authority above is separate and does not expand that packet's scope.
The [research reset contract](../research/campaign_2026-10-17/research_reset_r1/CONTRACT.json)
records this boundary. No fit, rental or submission is authorized by this reset.

The [research reset packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
records deployed creative prediction/cache repairs, deterministic checks,
full-data target/event census, sampled full-feature EDA, source registers and
independent proposal/critique rounds. Its ideas are proposals, not new model scores.


## Current scoped geometry investigation

The [geometry record](../research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) contains 128-train/128-valid raw-input EDA, visual inspection of all 112 inputs, and 33 matched tiny-GRU endpoints across two capacities. Independently verified pooled moment gain is +0.0008833 with opposing exposed-fold effects; this is not a production or public gain. The discrete occupancy/dwell and capacity-interaction pilot completed from immutable source snapshots. R3 then trained 12 H128×2 endpoints for eight epochs: all-feature averaged 0.359690 versus 0.376941 for control; dictionary-only averaged 0.377442; a fixed half control/dictionary mixture was +0.007337 over control on the same tiny panel. These are three-seed, 43-fit/46-evaluation-sequence exposed screens, not significance or transfer evidence. The user directed a stronger from-scratch H512×1 feature/control comparison on broader component membership. Its R4 worker was subsequently stopped and drained under the deadline instruction; the previous in-progress status is historical. Earlier startup attempts are preserved and produced no fits. Cloud/team work and the other agent's Kaggle ownership are unchanged.

The [R4 plan](../research/campaign_2026-10-17/geometry_from_scratch_r4/plan.json) specifies 30 paired seed/fold comparisons (60 from-scratch H512×1 fits), fixed epoch 2, equal-width zero-feature controls, component-excluded fitting membership and separate seed/component uncertainty. All four exposed/potentially exposed folds have 443–505 evaluation sequences and 141–149 components each. Full-data training and prediction profiles passed under a 5,000,000,000-byte, one-core, no-swap cap: peak RAM was 2.71 GB, CUDA allocated/reserved peaks were 0.91/1.29 GB. [Admission R3](../research/campaign_2026-10-17/geometry_from_scratch_r4/admission_r3.json) binds verified inputs and current sources; attempt R4 was later stopped/drained for the deadline deliverable. The full matrix is estimated at 113.8 GPU-hours plus setup and exports; no R4 quality score exists.

## Public leaderboard snapshot

The official leaderboard was read on 17 October 2026. The team was at public
WP **0.66832**, rank **22**, with **97 team submissions**; the leader was at **0.67243**. This is a dated
external team result, linked to the [official leaderboard](https://wundernn.io/connectome/leaderboard).
The page does not establish which exact ZIP or submission ID produced the
current team-best score. The result is not private-leaderboard evidence and
does not replace local validation.

The user's later portal screenshot shows **99 total submissions** and **2/5**
used today. It reports R2 at 0.66428 and R3 at 0.66456; the team best remains
the reported TL3 score 0.66832 in the provided results. See their exact
receipts above. This is user-supplied portal evidence, not an independently
queried leaderboard snapshot.

## Latest measured local evidence

The October17 [no-fit packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
adds deployed legacy fixes and a four-group frozen-checkpoint replay (WP0.06289742
with wrong preprocessing,0.72994782restored; bitwise manual/config parity). This
is bug isolation, not a new model ranking. Its full four-column census refutes
H-001 with1,253need-eligible exceptions. The shared evidence and25proposal branches
are published on research-team; [publication status](../research/campaign_2026-10-17/research_reset_r1/publication/README.md)
records that new PRs/main promotion await API authentication.

The last completed comparison in the current pause record is the exposed-F0
G-versus-C grade: G scored 0.68125607, a delta of **+0.00002956** over C, with
a conditional component 95% interval of **[−0.00038018, +0.00037707]**. This
single-seed comparison resolves no quality advantage. See the
[pause receipt](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md)
and [campaign frontier](../research/campaign_2026-10-13/strategy_execution_r1/FRONTIER.md).

Other late Main results remain scoped to their recorded panels. For example,
the matched short architecture screen measured H512×1 at 0.67323776 versus
H256×2 at 0.66977764 on one exposed F0 seed. That is a conditional screen, not
a converged architecture ranking or public-score forecast; see the
[student frontier](../research/campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md).

## Code and draft status

The modular trainer is integrated in Main. [scripts/train_rnn.py](../scripts/train_rnn.py)
is the compatibility command-line entrypoint; reusable training components
live under [src/wnn/training/](../src/wnn/training/README.md). The refactor
receipt reports static structure/lint checks and CLI help validation. It does
not report unit or integration tests, numerical training parity, a new fit, or
a throughput benchmark. Read the
[refactor receipt](../research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md)
before interpreting the change as evidence about model quality or performance.

The subsequent [repository maintenance](../research/unified/operations/MAIN_MAINTENANCE_20261017.md)
adds synthetic CPU regression and integration checks, legacy class compatibility
fixes, and fewer intermediate feature concatenations. This is software evidence;
it establishes no new model score, GPU parity or measured training speedup.

The 17 October GitHub snapshot recorded 14 experiment-prefixed draft pull
requests. They contain implementation proposals and handoffs; the count is not
a count of completed fits. Historical results already present in Main must be
tied to their own receipts. See the
[repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
[pull-request list](https://github.com/VladKochetov007/WunderFunder/pulls), and
[experiment draft workflow](../research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md).

## Navigation

- [Research objective](research-objective.md) — goal and binding constraints.
- [Research library](../RESEARCH.md) — how to find ideas, results and source
  evidence.
- [Repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
  [model catalog](../research/unified/models/CURRENT_MODEL_CATALOG.md),
  [validation protocols](../research/unified/validation/VALIDATION_PROTOCOLS.md),
  and [storage map](../research/unified/operations/STORAGE_MAP.md) — current
  maintained indexes for the 17 October audit.
- [Decision index](adr/README.md) — current decisions and their successors.
- [Complete dated handoff](../research/RESEARCH_COMPLETE_HANDOFF_20261012.md)
  — October 12 archive, not a current state.
- [Preserved state history](current-state-history-20261017.md) — byte-exact copy
  of the previous 199,958-byte file, SHA-256
  37be5efacd617f8e712c99822c0399242949aef10fbeebc9abd20ee90f499902; its
  same-directory location preserves the original relative link bases.
