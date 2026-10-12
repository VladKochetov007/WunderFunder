# Active frontier

Current continuation, 2026-10-04: implementation and autonomous local/Kaggle
GPU/TPU research are user-authorized. Read
[the execution contract](campaign_2026-10-04/EXECUTION_CONTRACT.md) and
[implementation state](campaign_2026-10-04/implementation_state.json).
E-406 binds new user-reported public results: q16x D256 preserves 0.6641 in
32m17s; repeat reference 0.6641 in 37m37s; TE07 0.6636; chunk500 0.6633.
Earlier negatives are now controls, not automatic vetoes on bounded retests.
Historical paragraphs below retain their original time-bound meaning.

Current root Challenge2 campaign update,2026-10-03: user goal is long-term competition understanding and research through23:00 Kyiv today, with0.667 optional. Best reported public0.6641; no0.667 claim. The remainder of this file preserves the earlier September frontier and its historical targets/results.

Current evidence: two-seed/fold0+1 Muon/chunk1000/stopped-bootstrap combo +.001854 local pooledWP, CI[.000420,.003409], fold1/class3 weakness; fixedfold2+3 confirmation nowPASS +.00178777 CI[.00092491,.00274712], all preregistered gates and independent metadata/scorer-route review pass. Combined recipe supported, no isolated-mechanism or public-score claim. Width320 pair +.00007249 CI[-.001211,.001313], no demonstrated gain. Physical coordinate, objective real-state and causal history support gaps now have runnable falsifiers. Peer observable feature fails its gate, independently reproduced; retained GRU influence does not establish a deficit. Capturev1/v2 rejected, minimalv3 correction awaits independent review before actualCUDAreproduction. Read [current synthesis](campaign_2026-10-03/continuation/round2/SYNTHESIS.md), [round4 portfolio](campaign_2026-10-03/continuation/round4/LITERATURE_AND_OPTIONS.md), [current worker registry](campaign_2026-10-03/continuation/round5/agents.json) and research/experiments.jsonl E-359 onward. Optimization prototype remains unapplied; no new GPU candidate promoted from toy/diagnostic evidence.

Latest update at22:06 Kyiv: E-394–E-402 preserve v4 builder/review, genuine GPU measurement and independent narrow parity PASS, false timeout diagnosis, and R5/R6 instruments/reviews. Default CUDA normal/repeat/original/forced-skip comparisons on synthetic architecture and the real fixed warm panel were exact, with observed peak3.71GB VRAM. Real trajectory had four completed warm updates and no skips. This unlocks reviewed local mechanism probes; it does not establish mature-fit recovery or public gain. State-drift original entrypoint was blocked for finite/runtime/memory guards and is being corrected separately; descriptive AMP component-sum diagnostic was blocked for tensor-valued metadata JSON identity and is being corrected separately. Actual scientific results remain pending. Current workers: [round6 registry](campaign_2026-10-03/continuation/round6/agents.json); newest verified restore: [recovery checkpoint](campaign_2026-10-03/continuation/CURRENT_RECOVERY.md). Monitor is healthy, GPU idle, no new OOM.

## Historical September frontier

## Target and baseline

Goal: validation WP >= 0.80 (clipped weighted Pearson, METRIC.md). Organizer
baseline 0.5896 (full). Best own model: GRU 384x2 + event features, 3000 train
seqs, epoch 0: 0.6096 on dev600 (E-028); CPU-budget GRU 256x2: 0.6036 (E-023).
Single seed; second epochs on 3000 seqs overfit.
Best devB764: mask-weighted devA fine-tune 0.6339 (E-036); train-only propensity^2-weighted GRU 256x2 0.6332 (E-039), 192x2 0.6317, 128x2 0.6310 (E-042, deployable candidate); plain 0.5950.

## Submission candidate

`submissions/ab9_pow2_h128/submission.zip` (1.2 MB): GRU 128x2 + causal event
features, trained on train.parquet only with scoring-propensity^2 weights
(propensity net fit on devA masks). Native Rust predictor, abi3 wheel built in
python:3.11-slim-bookworm. Holdout (official scorer) 0.6333 vs baseline 0.5899;
devB 0.6310/0.6314 (two seeds); 56 us/row incl scorer (~35 min for a
validation-sized test). Independent audit E-049: no look-ahead or leakage; crash and NaN issues fixed (E-050); open rules question on mask-derived training weights (user decision).

## Decisions pending (campaign paused 2026-09-15)

1. Rules: are training weights derived from validation is_scored allowed?
   Candidate (holdout 0.6333) uses them; fallback `submissions/ab12_h128_plain_fallback`
   (devB 0.5978) does not. Ask organizers if unclear.
2. Validation data in final training (stacking measured 0.6389 devB, E-045).
3. Hardware check (btrfs scrub, memtest) and restore train.parquet (E-032).
Goal 0.80 is not supported by any tested information source (H-014, H-016, E-021).

## Incident (E-032)

train.parquet sha256 changed during the session without a write (2 runs failed
with ZSTD corruption). Treat train-reading runs after ~02:30 as invalid until the
file is restored from the archive and its hash verified. valid.parquet is intact.

## Supported claims

- Targets move only on joint i0 bid+ask price events (H-001).
- Four market classes by target quantum; class 3 (q=0.65687) is large-tick with
  ~8 book states, ~35-38% of sequences, and the worst class (0.35-0.44) (H-002).
- Class-3 WP is bounded by next-break predictability; the next state alone would
  give 0.77 (H-009).
- Training is GPU-bound; inference is FLOP-bound at 12-15 GFLOPS (profiling note).

## Debunked

- H-004 more data/epochs alone: flat.
- H-005 dense native rewrite as big speedup: FLOP-bound, 112 vs 119 us.
- H-008 event-lag state features fix class 3: event-row evaluation artifact.
- H-010 snapshot queue features predict break direction early: AUC 0.55-0.59 beyond 20 rows.
- H-011 in-context Markov transition statistics: no early break information.
- H-012 class-3 scored rows near breaks: they are farther away.
- H-013 i1 sign-cumsum lead-lag: AUC <= 0.53 in class 3.

## Live mechanisms

| hypothesis | mechanism | evidence | next test |
|---|---|---|---|
| H-015 | exact target definition on event clock | r 0.5-0.6 with crude levels | better level recovery; online label reconstruction |
| H-007 | scored-row propensity weighting | AUC 0.82 | weighted training A/B |
| H-006 | sparse input compute | microbench 3x | sparse projection in native model |

## Score arithmetic

WP is an equal mean over sequences. With classes 0-2 at ~0.73 average and class 3
(~38% of sequences) at 0.38, overall ~0.60. Reaching 0.80 needs class 3 near
0.70 or every class far above current levels; the class-3 break-horizon result
(H-010) makes this the binding constraint.

## Bound (H-014)

Perfect next-break knowledge within 20 rows gives class-3 WP 0.42; within 100
rows 0.615. Measured direction AUC is 0.83 at 6-20 rows and ~0.55-0.60 beyond.
With class shares ~25/17/19/38%, 0.80 overall requires classes 0-2 near 1.0 or
class-3 break direction ~500 rows ahead. Current evidence says 0.80 is not
reachable causally unless a new information source for class 3 appears.

## TODO

- [x] H-011 in-context Markov break features (refuted)
- [ ] H-015 exact target definition; if exact, online per-sequence recalibration from reconstructed past labels
- [ ] Visual analysis toolkit (fast renderer) of books, events, targets, mask
- [ ] arXiv scout: queue-reactive, microprice, OFI, propagator, cross-impact
- [ ] H-007 propensity-weighted loss A/B
- [ ] Auxiliary targets at multiple horizons (Jane Street 8th-place pattern)
- [ ] Online test-time adaptation: labels are future-derived only; check if past targets are reconstructible from later features
- [ ] Perf pass: no per-row allocations in PyO3 step, 64-byte aligned weight arena in access order, fused gate loop, sparse input projection
- [ ] Portable wheel build in python:3.11-slim-bookworm container; submission timing on 1 vCPU


### 2026-10-12T10:39:53.824662+00:00 — R20 joint endpoint optimization and public C confirmation

User reports exact frozen C scored0.6671 in36m36s/642MB (portal1ya21vpd); public best in our user-reported records improves0.6668 by0.0003, runtime headroom1404s. Not independently browser-verified; no component causal decomposition/public forecast. A/B not submitted. Joint membership/precision/weights plan saved in research/campaign_2026-10-12/joint_endpoint_optimization_r1/PLAN_R1.md, four null-prior interaction hypotheses registered. Linear solve/SOCP limited to inactive clipping; bounded SLSQP follows float64 surrogate and final candidates are ranked with unchanged official float32 scorer. R1 numerical exactness wording retracted; R2 ULP test demonstrates proxy1 vs official0, eight synthetic tests pass on MAIN. No real-data optimized blend or new CV gain yet. Keep equal Q8 control; compare extra seed versus family by conditional same-CV gain and actual whole-package cost, mixed precision only if whole blend wins.

Complete clean F0 target cache materialized/finite/hash validated; all three teacher FIT_TRAIN exports retained. ProfileR1 failed before CUDA with KeyError(epoch_convention), exit1, zero batches, no scientific negative. Identified metadata-only cacheR2 preserves immutable NPY hardlinks/hashes and matches trainer field contract; independent critic reproduced R1 failure and R2 PASS using actual frozen trainer AST. ProfileR2 root release pins cae424 contract/02481 manifest/284c92 review; fresh lock/current+5.5GB<=18GB and bounded <=3GB filecache reclaim required, caps unchanged. Only resource profile authorized, no full fit. Earlier root max_batches/LR-horizon claim retracted: actual source retains44760 total scheduler steps. User-manager campaign limits authoritative; page-rounded cgroup bytes are not a cap change.

Local profile owner clean_t3_fit_cache_delivery retains end-to-end execution/terminal responsibility. Kaggle idle pending corrected cache/source asset bindings and actual-device profile plus refreshed quota; no duplicate teacher exports. VastB latest owned inventory absent, collection/balance unresolved; no new paid spending. Numerical actor prepares bounded real-data comparison; separate reviewer identifies next feasible C addition. Preserve all unrelated working-tree changes and disabled grade-bank/bulk-pull timers.


### 2026-10-12T10:51:28.770257+00:00 — R21 completed profile, next fit and bounded blend analysis

R2 cleanT3/F0 H256student resourceprofile terminalexit0 in15.16s:4completebatches40optimizerupdates0AMPskips; peakworker4.3307GB,parent15.9572GB,observedVRAM2.1789GB; worker/parentmemoryeventdeltas0. Same44760step sixepoch horizon and clean10535FT/1405FV/468evaluationexclusion retained. One warmup+3measuredcompletebatch median1.12639s/14.2seq/s is shortprofile, not sustainedfit guarantee. Actor freezesminimal controlp1(epochs0–1, stop1,fullstate) withsame6epoch scheduler; independentcritic alreadyaudited actualresume implementation, awaits exactfitargv/contract. Profile output cannotbequalityresult. Kaggle refreshed10:44:18Z hascampaign21712.352GPU-s/account35884.515s/reserved0, alltrackedkernelsterminalcollected. Paircohort needsactualR2cache/sourceprivateassetversion+deviceprofile. VastB10:42:26Z noinstance rows, cost/balance/finalcollectionunknown; no newpaidspend.

Nextendpoint audit: literalC+availableH388s5 exceeds20MB(projected20,665,891B beforeheaders). MatchingH390s5 absent. FixedH388replacement/compactpair eligibleforboundedcomparison butno directCconditionalCV. Q4rounding's.00180125 panel loss/notnativeINT4 speedproof still prevents defaultpromotion. CurrentC userreported.6671/36m36s,1404sheadroom; no newpublic result thischeck or forecast. Nativeanalysis owner'stwoNPYheadercommandsOOM in650MiBscopes: actualfirstparserdidnotverify/skipmagic, bogusheaderlength causedlarge read. Preservedjournal/diagnosis; no capincrease/blindretry, boundedactualformatfixture and completecomponentpanel instead of899MB3arraystack. This is supportexecutionfailure, notscience negative ortrainingOOM.

Latestuserrequests plotvalidation/publicdependence: standalonePNG/SVG+CSV withprotocolstrata andmodellabels commissioned fromLuna,11existingpairedrows, no imputedCvalidation orpooledregressionforecast. Keepquality/inference/cleanfold branches active. No newCVscore(n0newscientificseeds), ZIPs unchanged; root ownsactualnextfitstart andplotreview. See monitorR21 receipts.
