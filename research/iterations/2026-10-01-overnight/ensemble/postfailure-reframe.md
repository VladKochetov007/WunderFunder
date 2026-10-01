# ENSHEAD post-failure reframe — analysis only, no new allocation

**Recommendation: stop the ensemble-adaptation family for this campaign.** The existing four-corner comparisons show a substantial negative interaction when independently adapted members are combined. This makes stale-partner overcompensation a discriminable hypothesis, but it does not establish that hypothesis or supply evidence of a useful price improvement waiting to be recovered. Even removing the entire observed interaction only returns the sum of the nearly-null single-slot effects, still below baseline and well short of the allocation threshold. A properly controlled joint-head experiment is specified below as one unallocated reserve, not a rescue that should run automatically.

This analysis reads only the frozen loss/source/specification and already registered result JSONs. No predictions, target arrays, new sufficient statistics, forecasting fit, metric query, GPU job or implementation change was made. The arithmetic below is a descriptive recombination of existing readouts, not another evaluation or a new uncertainty estimate.

## What the already registered comparisons establish

All numbers are candidate-minus-original four-member H2 blend unless otherwise stated.

| Existing intervention | Pooled delta | Fold0 | Fold1 |
|---|---:|---:|---:|
| Full head, seed3 slot alone | −0.000009403 | +0.000107456 | +0.000002122 |
| Full head, seed4 slot alone | −0.000096562 | +0.000103868 | −0.000347555 |
| Both full heads together | **−0.000596892** | −0.000185676 | −0.000893625 |
| Scale, seed3 slot alone | +0.000161311 | +0.000198280 | +0.000076754 |
| Scale, seed4 slot alone | +0.000084875 | +0.000177748 | +0.000021912 |
| Both scales together | +0.000121319 | +0.000260040 | −0.000042112 |

The point interaction is the mixed finite difference

`I = S(both replacements) − S(slot3 only) − S(slot4 only) + S(original)`.

Computed directly from those stored deltas:

| Interaction | Pooled | Fold0 | Fold1 | Target0 | Target1 |
|---|---:|---:|---:|---:|---:|
| Full head | **−0.000490928** | −0.000397000 | −0.000548192 | −0.000412264 | −0.000569591 |
| Scale only | **−0.000124868** | −0.000115988 | −0.000140778 | −0.000105513 | −0.000144223 |

Both targets and both folds show the same point interaction sign. This is stronger descriptive evidence than merely observing a bad combined result. However, the separate result files do not contain a jointly resampled four-corner interaction distribution. Their marginal confidence intervals cannot be added as if independent, so **no interaction CI or significance claim is available here**.

Full-head standalone WP changes are −0.0285964 for seed3 and −0.0147957 for seed4, while their single-slot ensemble changes are approximately zero. This is consistent with members specializing to fixed partners; standalone degradation is not itself a failure for an ensemble objective. It also does not establish useful specialization: neither pooled single-slot full-head effect is positive, and neither reaches the +0.0003 materiality gate. The combined full-head comparison has conditional95% interval [−0.0014942,+0.0002225]. Full-head minus scale is −0.0007182, with conditional interval [−0.0014194,−0.0000578]. These are conditional comparisons on exposed chains, not evidence that all ensemble training is futile.

The scale arm does not isolate post-hoc scaling of H2: it starts from the original parent and omits ordinary nonlinear H2 adaptation. Its ~−0.001 standalone difference against H2 therefore cannot be attributed to scaling alone. Within each fold, positive scaling would preserve unclipped standalone Pearson; cross-fold pooling and clipping can still change the reported pooled value. Likewise, seed3's single-slot full-head effect is positive within both folds but slightly negative when pooled: the global statistic includes between-fold location/scale relationships and is not a weighted mean of fold scores. Do not choose the favorable representation of that result.

## Why stale partners could cause this, and what remains unidentified

Let B be the original preclip ensemble, and let d3,d4 be quarter-weighted changes of the two ordinary slots. Then the separately evaluated ensembles are E3=B+d3 and E4=B+d4, while the served pair is

`E34 = B+d3+d4 = E3+E4−B`.

If both independent objectives learn a similar correction d intended to repair the same residual left by the same frozen partners, deployment applies approximately2d. Each original training objective saw the **old H2 version** of the other ordinary member, not its eventual adapted version. The current recipe therefore permits double compensation. The observed negative interaction is compatible with this story.

But Pearson is nonlinear even without clipping. Its curvature alone can make two individually useful movements combine subadditively; clipping adds another nonlinearity. Negative interaction does not identify positive correlation of corrections, an optimal step length, or overfitting. The scale arm also has negative interaction, so a complex nonlinear cancellation mechanism is not required to produce it. Both full-head single-slot effects being near zero remains consistent with in-sample-partner overfit, surrogate/global-score mismatch, optimizer behavior, or simply no transferable incremental signal. We have no registered train-versus-held error decomposition here that separates those accounts.

An arithmetic ceiling on this particular rescue story is useful: deleting the full-head interaction while leaving the observed single-slot effects unchanged yields **−0.000105965**, their sum. That is still **0.000405965 short of +0.0003**. Joint training could change the individual movements and do better, so this is not a formal bound on joint methods. It does show that “remove the observed double compensation” alone is insufficient to justify promotion.

## One concrete joint test, if a later controller explicitly reallocates budget

The scientifically cleaner practical question is whether a shared ensemble objective improves two heads trained together compared with the same two heads trained on separate standalone objectives **under an identical common minibatch order**. Existing on_ctl128s3/on_ctl128s4 cannot be this matched control: their random orders differ, so reusing them would change objective and coupling of optimizer samples together. Old ENS results remain an exploratory reference only.

Use the same two parent epoch3 checkpoints per fold (ref seed3 and refs4 seed4), both frozen inp/gru backbones, and the unchanged fixed distilled256H2 and chunk500H2 partners. Fit the two nonlinear ordinary heads together in each process. Both arms see the same complete sequences and masks, same batch16/chunk2000, two epochs, AdamW lr5e−4/wd.01, original OneCycle, auxiliary coefficients .3/.3, exact fixed final epoch1, no held-out evaluation or checkpoint selection. No scale-only arm, SAM, regularizer, clipping change, schedule sweep, model blending or single-slot rescue is added.

Let A3,A4 be the original auxiliary losses including their fixed coefficients, q the sum of the two fixed partner predictions, and rhoLoss the existing **unclipped minibatch×chunk** gcorr surrogate.

- Matched separate-loss control: `L_sep = rhoLoss(p3,y,m) + rhoLoss(p4,y,m) + A3 + A4`.
- Joint candidate: `L_joint = 4*rhoLoss((p3+p4+q)/4,y,m) + A3 + A4`.

There is **one** 4× ensemble main term, not one duplicated nondetached term per head. When all four predictions coincide, each head's main output gradient matches the standalone control. Both heads receive gradients evaluated at the same current ensemble before either optimizer steps. Two per-head AdamW optimizers and separate per-head clipping at norm1 preserve the existing update convention; clipping one joint norm would introduce an additional coupling. Keep loss scaling/overflow handling matched and log skipped steps. The genuine joint gradient may equivalently be implemented with two losses using the other head's **current detached output**, provided both forwards and both backward passes precede either update; verify equality against the one-common-loss reference on synthetic tensors.

Shuffle replication0 uses one explicitly seeded Torch generator with seed3, independent of model-construction RNG, supplying the same epoch permutations to both heads in both arms. Shuffle replication1 uses seed4, again paired across arms. Log exact sequence-order hashes. Parent checkpoints and their original seed identities stay unchanged in the second replication: it measures optimizer-order variability conditional on two existing parent backbones, not robustness to independently trained new parents. The two fixed partner caches must use the same own-fold models and eligible training row groups already validated; no alternative-fold fill and no held target access.

Before any fit, freeze exact source, legal cache manifests, common shuffle/order hashes, per-head optimizer/scaler handling, endpoint, serving blend and the following gates. Allocate at most two primary metric comparisons for the first shuffle and two more only after its gate passes; no standalone or favorable-slot readout is needed for this reserve. Grade only the simultaneous two-head replacement at quarter weights alongside the unchanged fixed two partners.

Proposed first-screen spending gate: joint minus matched separate-loss control ≥+0.0003 pooled with neither fold negative **and** joint minus original H2 serving blend ≥+0.0003 pooled with neither fold negative. A positive joint-versus-stale historical difference alone is insufficient: that could merely repair a known bad recipe. If either comparison fails, stop this joint recipe without another shuffle, loss variant, slot selection or epoch search. If both pass, repeat the entire paired comparison with common shuffle4 and require the same directional/fold gates and materiality. Final recipe would use predeclared shuffle3, not whichever shuffle scored best.

This tests useful joint optimization. It would not uniquely prove that ENSHEAD-01 failed because of stale-partner double compensation: the new comparator is a matched separate standalone loss and the old stale-trained models had different data order. The mechanism claim must remain narrower than the practical result. A valid nonpositive joint-versus-matched result would falsify the useful-joint-update prediction for this specific recipe, not every ensemble architecture.

## Why this reserve should not consume the current remaining budget

A screen is2folds×2arms×2updated heads = **8 forecasting fits**, although it uses only four training processes. A paired second shuffle costs another8. Using root's budget snapshot,28completed +8SAM running/allocated +8conditional SAM replication +2final heads =46reserved, leaving18. The joint screen and repeat would consume16 of those18, competing with a genuinely different feature/representation direction and leaving only2 spare fits.

More decisively, current state.json already lists **all8 allowed scientific configurations**: plainH2, AUXoff, F, T, ENShead, ENSscale, SAMrho0, SAMrho005. The matched-common-shuffle control and joint treatment are **two additional configurations**; existing control names or a changed shuffle cannot conceal this. Remaining fit capacity does not override the configuration cap. No new experiment is allocated by this document; a controller would first need an explicit recorded contract amendment/reallocation, with the full selection history retained.

Given the absence of a material positive single-slot effect and the rescue arithmetic above, my recommendation is **do not amend that cap for ENS tonight**. Preserve the negative results and the identified interaction, complete the already allocated distinct training work, and spend any newly approved configuration on a different mechanism with stronger evidence. This is a resource decision, not a claim that the data contain no further signal.

## Evidence identities

All paths below are existing immutable outputs under `../evaluations/`:

- ENSHEAD-01-primary.json: `0a6a91f7f81ad0ebd83b2e4ec7f8ff1eb295e1a57a0e2e6ee7eb492b3d7d12a0`.
- ENSHEAD-01-ens-seed3-slot.json: `ff40fbd9f02e8b251e5e1bb58f3170e1dd31d377851ecd3d3f5e45813a1bd477`.
- ENSHEAD-01-ens-seed4-slot.json: `3197db672fd92297b07f9246abf25ee61a7f46645cf18e39ecc34bb07a356703`.
- ENSHEAD-01-scale-primary.json: `d1414951829209d59c04100c1749347f34d6f29f363f489c9e4234a12ed8efbf`.
- ENSHEAD-01-scale-seed3-slot.json: `4c7d68711f6b6733468ff67269ca646147dde74ef929152b2097390fdadf0d97`.
- ENSHEAD-01-scale-seed4-slot.json: `2430f8f2d10ccea6494370cc65a15229112594b4d64313fe77d58aa82fd84c15`.

The frozen ENS specification and actual-fit/scale-export checks were also read. No source or existing artifact was edited.
