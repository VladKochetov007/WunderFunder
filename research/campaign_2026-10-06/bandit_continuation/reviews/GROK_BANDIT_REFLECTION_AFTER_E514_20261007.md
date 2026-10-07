# Independent Grok 4.7 bandit reflection after E-514

Started: 2026-10-07T06:11:46+03:00

Error: No such device or address (os error 6)


Grok CLI exit status: 1
Completed: 2026-10-07T06:11:46+03:00

## Retry after terminal fix

Started: 2026-10-07T06:12:07+03:00

The positional-prompt invocation entered terminal job-control stop (`grok` process state `T`, background process group, no response). It was terminated without a model answer. This was an execution/terminal failure; no analysis was received.

## Retry in bounded headless single mode

Starting after a bounded `READY` smoke check confirmed native Grok works in this non-interactive resource scope. This retry uses `--single`, a 32-turn cap, and the same one-core/1.5-GB/10-minute limits.

Started: 2026-10-07T06:21:17+03:00

I'll treat this as a read-only critic pass: first the research contract, current campaign state, and the evidence-review rules, then E-511, E-513, and E-514 on their exact scopes. No files, allocation, experiments, or submissions will change.The campaign tree is dirty with E-514 artifacts and later state updates. I'll read the frozen receipts and the metric next, and keep earlier reflection notes out of the evidence.## E-514 critic memo

Read-only pass under the evidence-review and competition-metric rules. No files, allocation, fits, or submissions were changed. Claims below are tied to the receipts. `specs/current-state.md`, `FRONTIER.md`, and `research/bandit_tree/generated/INDEX.md` still describe E-511 seed 4 as running and this arm as untested. The later records are journal rows `TREE-20261007-009` and `TREE-20261007-017`, `research/experiments.jsonl` E-511/E-514 rows, and the E-514 result.

### 1. E-514 evidence and limits

The completed v3 run is an exposed fold-0 chain cross-fit of one fixed pool. The anchor is the equal raw mean of `bd26_c1v2_s3_f0` and `bd26_c1v2_s4_f0` at epoch 1. The pool is `ab12_h128_plain_fallback`, `ab48_basehead_f0`, and the organizer baseline at ×0.25. Scoring is `global_wp_clip2_v1`: the notebook mask is cache `is_scored` and step ≥ 99 (`weak_portfolio_eda.py`), then `GlobalAccumulator` in `src/wnn/gmetric.py` clips targets and predictions to ±2, weights by the absolute clipped target, and averages the two target weighted Pearsons.

Every selected lambda is 0. That is all 15 strategy-by-outer-fold endpoints in `E514_RESULT.json` (`selected_endpoint_by_outer_fold`). `_blend` returns the anchor whenever lambda is 0 or the member set is empty (`weak_portfolio_eda.py`). The five once-held-out strategies are therefore the same prediction. Concatenated OOF WP is 0.6832264634983685 for each, and every delta is 0. The same identity holds on outer folds of 150, 130, and 152 groups (41 chains each; 432 groups, 123 chains). The paired 2,000-replicate intervals `[0, 0]` are the bootstrap of identical endpoints. They are not a precise estimate that some other blend has zero effect.

The conditional-signal rule kept a candidate only when the mean of its two target partial correlations was positive. On all three calibration partitions that mean is negative for all three candidates, so the member set is empty and lambda is forced to 0. The closest case is fold 2 `ab12`, mean about −0.000415. Per target, `ab12` has a small positive partial on t0 and a negative partial on t1 in every fold. The other three strategies did search the grid `{0, 0.05, 0.10, 0.15, 0.20, 0.25}` with official calibration WP. `_choose_lambda` keeps the smaller lambda when scores are within 1e-8. Stored winners are still lambda 0, including best-single, which selected `ab12` and then lambda 0. The calibration margins for the rejected positive steps are not in the result file.

Descriptive pure-fold-0 standalone WPs are C1 pair 0.683226, `ab48` 0.662881, baseline×0.25 0.659408, and `ab12` 0.604966. Mean official-weighted prediction correlations with the C1 pair are about 0.887, 0.972, and 0.967. Those correlations did not choose the reported endpoint.

This slice is not the E-511 comparison. E-511’s equal pair is 0.682176 versus C0 0.681153 on 468 sequences (`local/E511_C1V2_C0_EVALUATION_v1/REPORT.md`). E-514 drops train-bridged chains and does not score C0.

The result digest `8de4dc6c…` is the file hash of `SOURCE_MANIFEST_E514_V3.json`. The result’s `source_manifest` name still says `SOURCE_MANIFEST_E514.json`, whose file hash is the earlier v1 digest `95fa6c08…`. The current notebook hash `0f03c686…` and partition hash `e9e0cb93…` match v3, and `src/wnn/gmetric.py` matches the digest v3 records. I read the ancestry asserts. I did not re-execute them or rehash all 56 manifest inputs. The recorded gate says C1 uses fixed epoch 1, no evaluation during fit, and `c20260929_prop_f0` at fixed epoch 3 with fold-0 fit/eval membership separated; `ab12` is train-only with empty propensity; `ab48` has empty propensity and fold-0 groups disjoint from its fit groups; the baseline is the organizer ONNX. Fold 0 was already exposed (`PRE_REGISTRATION.md`).

Earlier failures are execution evidence, not scores. The graph-name collision and the JSON partition-key mismatch started no comparison. The v2 attempt raised `NameError: _merge_moments` after first-fold calibration and some anchor moments (`E514_EXECUTION_FAILURE_20261007.json`). That receipt says no held-out aggregate, no result file, zero OOM kills, and that partial numbers were not inspected. The v3 rerun is the scientific result. `REPORT.md` prints +0.000000 without stating that the endpoints are the anchor.

E-513 is not usable as out-of-fold evidence. `weak_cohort_portfolio/INDEPENDENT_REVIEW_MEMO_20261007.md` records that `prop_v1` fit or selected checkpoints on evaluated folds, and `prop_v2_f{k}` selected checkpoints on the evaluated-fold `is_scored` masks. The preregistered bootstrap seed was incremented for one direction; the memo’s exact-seed intervals still cross zero. E-514’s preregistration says those scores were not used to choose this pool or its weights. H-201 stays insufficient evidence. Its point scores stay descriptive.

### 2. Direction decision

**Exact E-514 arm: STOP.** Status for this pool, this lambda grid, this conditional-mean rule, and these fold-0 chains: `screen_failed`. The preregistered continuation rule required the conditional-signal portfolio to beat both the anchor and the best single control, with uncertainty compatible with a positive change. The served endpoint is the anchor, so that bar fails. Do not widen members or lambdas on this cohort.

**Broader weak-model complementary use: REFRAME, and do not continue it as “merge less-correlated weak scores.”** H-202’s predicted observable was a cohort that first had positive conditional target signal and then improved held-out `global_wp_clip2_v1`. No candidate produced that signal, so that half was not tested. The lowest-correlation arm also never left the anchor, so this run also does not show that adding the less-correlated member hurts out of fold. It shows the registered selector refused every positive grid weight.

Low prediction correlation is agreement between models. The official score is pooled clipped weighted Pearson with the labels. A less-correlated weak prediction can be residual noise. Mixing `(1−λ)A + λC` raises prediction variance under the Pearson denominator whenever the added piece does not raise covariance with the clipped, weighted target enough to pay for it. The quantities that can create a gain are target-aligned residual association after the anchor, target-specific scale and bias, the clip and `|clip(y)|` weights, which connected chains carry the weight, and the fact that member and lambda were chosen on already exposed chains. Here the partials change sign across t0 and t1, the served scale is far from a fitted coefficient, and the smallest positive grid step is 0.05. The selector’s answer on official calibration WP was to keep the anchor.

No local delta from this file is a public-score forecast. Own public best remains 0.6660 in `specs/current-state.md`. `new_public_gain_verified` is false.

### 3. Bandit consequence and relative priorities

Assign no numeric reward, prior, or expected gain. Close launch admission for `arm:conditional-signal-weak-portfolio` on this recipe. Keep the artifacts. Preserve teacher-target, residual-representation, and distillation uses as separate arms.

Priority after this result, without using the tree’s integer priority field as a reward:

1. **Cloud baseline and raw-teacher path.** `FRONTIER.md` still has E508-v3 and E463 waiting on provider reconciliation, and E512 waiting on E510. That is blocked execution, not a negative model result. The resumed allocation’s leading work is independently trained teachers and raw-versus-clipped family targets. New GPU/TPU experiment compute is Kaggle-only after 10:00 Europe/Kyiv (`runtime/USER_STEERING_20261007_0105Z.json`). This memo was written at 06:21 EEST, so the local window is still open, but this arm does not justify spending it.

2. **E-511, uncertain.** Equal-pair delta about +0.001023, component-bootstrap 95% interval [−0.001473, +0.004619], `p(delta ≤ 0) = 0.3015`, 141 components, exposed fold 0. The interval includes zero. The journal leaves the numerical prior null. A second ancestry-clean fold would be a new experiment, on Kaggle after the cutoff, and is not promoted by adding that local delta to 0.6660. It still ranks above remixing predictions E-514 already served as the anchor, because C1 at least changed the prediction relative to C0.

3. **Heterogeneous teacher compression** (`arm:heterogeneous-teacher-compression`) remains untested and not launch-authorized. It is a student-training question, not this blend. Teammate weights stay forbidden. It stays ahead of another weak-score screen and behind unblocking the cloud teacher inputs.

4. **This exact weak pool** stops. E-513 stays closed.

`generated/INDEX.md` is staler than `config.json` and the journal. Do not treat its `not_tested` labels for these two arms as current.

### 4. Next test

**Defer.** Do not preregister another grid on this fold-0 cohort.

A later card is admissible only when a new member already has clean fit, propensity, checkpoint-selection, and row-order ancestry, and a calibration set disjoint from evaluation shows positive target-conditional association after the anchor. The served endpoint must be frozen `global_wp_clip2_v1` on connected chains that did not choose the members or weights. The falsifier is: selected lambda stays 0, the held-out blend does not beat both the anchor and the best single control, or one chain block carries the delta. Stop that new pool if calibration again refuses every positive predeclared weight. Per-target scale, bias, and sub-0.05 weights were not tested here. Searching them on these same chains would be the forbidden grid extension.

### 5. Decisive evidence gaps

- Rejected calibration WPs are not stored, so the size of the loss at lambda 0.05 is unknown. Only the winner and the 1e-8 tie rule are known.
- The conditional gate averages the two targets. Opposite-sign partials, including `ab12`’s small positive t0 partial, were discarded. A per-target coefficient was not scored.
- The partial correlation is scale-free and computed on series clipped before mixing. The served blend is clipped after mixing. It is not the derivative of pooled WP.
- Because every served prediction is the anchor, chain composition and bootstrap uncertainty for a nonzero mix were not observed.
- Fold 0, these families, and this selection are exposed. The bootstrap conditions on the fitted predictions and the chosen lambdas. It omits training seeds, pool choice, and public transfer.
- E-513 cannot fill those gaps.
- Ancestry for E-514 is a recorded gate that I inspected in source and did not replay.

This memo does not rewrite the grader and is not self-approval of the E-514 actor.


Grok CLI exit status: 0
Completed: 2026-10-07T06:26:12+03:00
