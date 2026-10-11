# Strong sources, conditional diversity, and serving headroom

Latest user instructions: pursue a real gain and a new submission with an honestly supported expectation of public 0.67+, use available inference time for quality, take time to train the best models, and prioritize mixing them well. The broader top-1/>0.700 objective remains. This changes allocation priority; it does not establish a numeric public forecast. Root retains the existing compute boundaries and parallel branches.

## What quality and diversity actually mean

For one target with a fixed official mask and label-derived weights, consider centered, unit-variance predictions p and q. Before clipping changes their values, let a=corr(y,p), b=corr(y,q), and c=corr(p,q), all using the same label weights. The correlation of their standardized mixture is

    r(t) = ((1-t)a + t b) / sqrt((1-t)^2 + t^2 + 2t(1-t)c).

Its derivative at t=0 is b-a*c. A weaker member can help if it predicts the incumbent's residual sufficiently well. Low prediction correlation alone can be independent noise, which does not help. With unstandardized outputs the derivative includes the positive factor sigma_q/sigma_p; equal raw weights are not equal normalized influence. The official score averages two target correlations, whose conditional contributions can differ.

For our actual raw-output mixture, clipping is part of evaluation. Away from values exactly at the clipping thresholds, put P=clip(p), Y=clip(y), and h=1{|p|<2}(q-p). The directional derivative of the weighted correlation is

    Cov_w(Y,h)/(sigma_Y sigma_P)
      - corr_w(Y,P)*Cov_w(P,h)/sigma_P^2.

These identities explain the interaction, not a replacement evaluator or a weight-search recipe. Execute frozen raw mixtures and then official clipping with `global_wp_clip2_v1`. There is no universal member-quality cutoff, and a favorable derivative does not prove that a 50/50 mixture or public transfer improves.

## Measured source-bank comparisons

Six mixtures were fixed before this grading pass. Prediction byte counts, shape/dtype, complete payload SHA, row ordering and fold split were checked; direct fitting membership excludes all 457 evaluated valid groups. The antecedent teacher/initializer audit remains a separate provenance obligation. Paired comparisons use 149 components, 2,000 resamples, seed17 and official centered moments. All estimates condition on fitted models and repeatedly exposed fold1; they are not seed or public uncertainty.

| Endpoint | Official WP | Difference and paired component 95% interval |
|---|---:|---|
| Original strong H384 teachers s3/s4/s5, equal | .672541621 | Reference; original set was selected during research |
| Same recipe, add s8/s9/s10, six equal | .671833719 | -.000707903 vs original three; [-.001907506,+.000291462] |
| Replace original s5 with H384 prop0 s3, three equal | .671848737 | -.000692885 vs original three; [-.002827986,+.000936884] |
| Half current student, half original teacher ensemble | .672640093 | +.001608620 vs student; [+.000613667,+.002561849]; +.000098471 vs teacher ensemble, [-.000659363,+.000890530] |
| Current student + stronger-parent student, equal | .671580100 | +.000548627 vs current student; [-.000212449,+.001278840] |
| Current student + alpha.95 sibling, equal | .670997634 | -.000033839 vs current student; [-.000234275,+.000145482] |

Interpretation: source quality presently dominates simply increasing the member count in these tested sets. This comparison confounds source replacement with seed quality; it does not establish a universal diversity ranking. The student retains complementary information, but mixing it with its teachers mainly recovers teacher signal lost during compression. An alpha variant adds almost none; parent variation adds a modest conditional contribution. These findings favor stronger sources and better absorption over another alpha/epoch sweep. The intervals do not automatically close uncertain descendants.

Other landed single endpoints: H384 teachers s8 .66870 / s9 .66723, older recurrent-only B64-linear .66715, sum/difference-weighted KD .67040, alpha.9/e10 .67094, stronger-parent/e6 .67033, alpha.95/e6 .67094. Actual exported epochs are respectively2/2/2/1/9/5/5. Same-seed machine reruns are not distinct seeds. The ten-epoch student has no observed improvement over the six-epoch .671031473 endpoint; this is one-seed evidence, not equivalence.

Raw records: [grade freeze](ops_continuation_20261011/QUALITY_DIVERSITY_GRADE_FREEZE_R1.json), [paired results](ops_continuation_20261011/QUALITY_DIVERSITY_PAIRED_R1.jsonl), [singles](ops_continuation_20261011/QUALITY_DIVERSITY_score_prediction_remote_R1.txt), [fixed mixtures](ops_continuation_20261011/QUALITY_DIVERSITY_blend_score_remote_R1.txt).

## Use the serving budget for stronger endpoints

User-reported portal durations are about15minutes for the latest h256 singles and25–27minutes for pairs, with roughly625MB memory. Under the existing60minute/16GB envelope this is substantial headroom. Platform differences and nonlinear runtime mean these timings do not establish that an arbitrary larger mixture fits.

An isolated packaging role now measures exact causal native Q8VNNI callback cost for three h256 members and h256+h384, optionally two h384, on the same complete sequences. Repeated-weight fixtures measure capacity only and are never scientific ensembles. Bind the measured ZIP/native library/weights identities, verify deterministic reset, and report per-row cost rather than inventing a portal time. Training larger teachers remains independently useful even when their direct ensemble cannot be served. Q8VNNI remains the primary serving precision because existing parity evidence is strong; use extra time first for more useful models, and assess higher precision only through its own measured endpoint contribution.

## Allocation order

1. Deliver new all-data strong students and fixed student mixtures, with declared independent own parents/seeds and complete teacher-target coverage. Recover the strong teacher signal before increasing member count. All-data scores cannot support held-out claims.
2. Test a wider student against the same strong target and compatible own parent. The pending h384 two-epoch/default-alpha arm is useful but is not the same treatment as the successful h256 alpha.9/e6 recipe. A larger student and an all-data H384 teacher successor need explicit recipes and actual device admission.
3. Test a few mechanism-diverse additions by conditional value to a fixed strong endpoint. Retain admitted Pearson/run-level KD and OOD/path branches; score complete landed mixtures, not just component quality. Avoid exhaustive pair/weight searches on fold1.
4. Use Kaggle's next bounded cross-fold contrast to diagnose scratch robustness, preserving already admitted seed work. Do not spend the next cohort solely on more fold1 scratch seeds.
5. Keep the two reviewed same-prefix tails queued locally as a controlled test of movement and auxiliary conflict. H377 profile/initial controlled batch contrast remains useful; the remaining quartet treatments are lower priority than strong final models. H379's unimplemented probe is deferred rather than called queued.

Root owns artifact collection, official grading, recipe/source reviews, resource admission, and packaging through human-upload handoff. Exact live queue admissions and dependencies are in receipts; this plan does not turn prepared work into executed work or promise public0.67 from a local delta.
