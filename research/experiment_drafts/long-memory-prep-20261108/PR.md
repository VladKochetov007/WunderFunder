# TL3 role replay and long-memory successor audit

1. **Status, lineage and owner.** ROOT completed a CPU score-only replay of
   three existing TL3 role caches and their fixed blend. No model inference or
   new fitting occurred. The proposed long-memory successor is preparation
   only; ROOT owns any later resource admission. The executed R4 scorer and
   result hashes are in RESULTS_R4.md. Native B300 optimizer/horizon fitting
   belongs to separate PR72, not this packet.

2. **Why investigate this.** Determine whether the large historical reference
   advantage comes only from ensembling, then avoid repeating existing compact
   architectures. The best standalone TL3 role scores0.6914165; the fixed blend
   scores0.6923774. Ensemble gain alone therefore does not explain the gap to
   the earlier Transformer. The six historical12E curves show late degradation
   in GRU, GRU-local and Mamba2. A future memory/local-attention interaction
   could add complementary context; extra capacity or optimization differences
   are alternative explanations and need matched fresh controls.

3. **Compute advantage.** This replay needs CPU only and reuses pinned cached
   predictions. No new L4/B300 allocation is consumed here. A prospective
   Mamba2-memory/local-attention/direct-head model targets a useful compact
   regressor, but its resource cost and serving compatibility are unmeasured.

4. **Comparison and recipes.** The completed comparison is three fixed role
   arrays plus the fixed FP32 left-to-right0.34/0.33/0.33 blend. No validation
   fitting or weight search occurs. Four JSONs are conditional6E GRU/GRU-local
   retest recipes only, not new architecture discoveries. They differ from old
   12E recipes in schedule, B32/chunk1000 and local FFN width. The distinct
   Mamba2 hybrid has no adapter or frozen runnable recipe yet.

5. **Bounded panel.** Actual new scientific fits:0. Reused roles:3, one fixed
   blend. Future conditional hybrid panel:2 paired seeds × (fresh Mamba2
   control, hybrid),4 new fits total,6E each. Seeds, admission and profiles are
   not approved; there is no automatic dispatch from this draft.

6. **Validation and uncertainty.** Official global_wp_clip2_v1 on all1,873
   valid sequences /3,528,273 is_scored rows after warmup. Valid is already
   exposed. Cached regrading measures fixed-artifact performance, not fitting
   variance; two historical seeds supply descriptive contrasts only. No public
   improvement or significance is claimed.

7. **Leakage and coverage.** Exact official manifest SHA and unique fit-member
   sets are checked. Qualified train contains10,325 noncontiguous groups in a
   10,607-group raw source; never infer membership from a range. Provenance
   list permutations are allowed only for the exact unique member set; cache
   row order remains strictly tied to the valid source index. The historical
   consumed training-cache ancestry is not reconstructed. That limitation is
   retained, not repaired by this replay. Primary validation remains ADR0044.

8. **Decision-changing result.** Replay reproduces the baseline and identifies
   a strong standalone reference. Historical GRU-local has mixed E6 deltas and
   worse E12 scores; a blind repeat is low priority. A fresh hybrid would need
   a predeclared useful paired effect/cost criterion before admission. Missing
   criteria and implementation are explicitly pending; no threshold is chosen
   after seeing future hybrid results.

9. **How to run.** R4's exact CPU-only native command is in README.md. It reads
   pinned role files, sidecars, provenance, split and valid data, and requires
   a new output path. R1–R3 are preserved failed graders before scoring, not
   launch alternatives. No hybrid trainer/profile command exists yet.

10. **Evidence to return.** RESULTS_R4.md lists role/checkpoint/prediction
    hashes; evidence/tl3-roles-r4.json is the exact native result record.
    SCREEN_REVIEW_R1.md retains all six12E validation trajectories, train-eval
    diagnostics, recipe differences, static counts and focused test results.
    Existing native role weights remain at their recorded paths; they are not
    embedded in this Git commit. Future fits must retain all epoch curves,
    objective/LR/gradient/resource metrics, weights and graded predictions.

11. **Results and trace.** Three roles score0.6914165040,0.6914152791,
    0.6909489195; fixed blend0.6923774248. R4 succeeds; R1/R2/R3 are API,
    contiguous-membership and provenance-order adapter failures, respectively.
    This draft records evidence and conditional successors; it does not
    promote a new model, submit an archive or restart any historical fit.
