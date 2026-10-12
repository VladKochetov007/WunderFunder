# Submission quality, diversity and inference at the research stop

This is a synthesis of recorded evidence, not a public-score forecast or authority to launch more work. The user requested completion of the admitted runs, one detailed linked handoff, useful Git cleanup, then no further research. [Stop/drain scope](STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json) governs execution.

## What the actual uploads establish

| Endpoint | User-reported public | Competition elapsed | Interpretation |
|---|---:|---:|---|
| H390 s3/s4 equal Q8 (A) | .6656 | 24m31s | Seed averaging alone did not outperform the mixed family. |
| H390 s3/Z4 equal Q8 (B) | .6668 | 25m56s | Better point score than A at similar native Q8 cost. This does not isolate diversity from source quality. |
| H390 s3/s4/Z4 half-family Q8 (C) | .6671 | 36m36s | Highest user-reported result; only +.0003 versus B for another10m40s. |

These overlapping single uploads are not independent replications or a reliable runtime-to-quality regression. Public feedback is now exposed development/transfer information. The earlier .6668 old all-data pair and the latest .6671 are different mixtures; there is no justified numerical forecast for D/E. [Exact public receipts](../../submissions/10-12-best-r1/PUBLIC_RESULTS_20261012_R1.json), [observed runtime plot](public_runtime_quality_r1/PUBLIC_RUNTIME_VS_SCORE_R1.png), [protocol-separated validation/public plot](public_validation_plot_r2/PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.png).

The earlier fold1 scratch model's .67195 local/.6616 public miss was not a serving failure: the archived ZIP/PyTorch replay was nearly identical. Fold selection, family transfer and the nearby longer-training cliff remain more plausible than quantization failure. The distilled student's local improvement transferred partly. Do not choose a supposedly public-like fold or regress a forecast from a few adaptively selected uploads. [Public-gap review](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md).

## Source quality and conditional diversity

Teacher seed averaging is useful within the measured family: exposed F0 three own H384 teachers mean .67901819 versus equal trio .68090199; exposed F1 mean .67068046 versus trio .67254162. These are three fitted teacher seeds, not three independent tests of the ensemble. Adding unrelated-looking weak models can still hurt; disagreement alone is not useful signal.

The newly completed clean T3 F0 control s3 CP5 scored .6811553603 (468 exposed groups, n1 student seed). The old own Z4 F0 student scored .6831418134, but it differs in recipe, ancestry, checkpoint and FP16 inference; their gap does not isolate teacher quality. The preregistered fixed equal blend scored .6832549498: +.0001131364 versus the old student, component-bootstrap95% interval [-.0012870,+.0015166]. Thus the weaker single has a small positive conditional point, but this is uncertain fixed-prediction evidence, not a recipe winner or public gain. Its t0 .6769247 is below old-only .6771695; its t1 .6895852 is above old-only .6891141. Target-specific mixing is an existing hypothesis, not a new tested endpoint.

[Exact fixed screen and resource limitations](clean_t3_f0_student_contribution_screen_r1/RESULT_HANDOFF_R1.json), [independent CP5/ancestry review](clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md). The screen completed without OOM, but its1.436GB RSS was too close to a1.5GB cap; future same-task support needs measured working-array margin and the stronger17GB aggregate admission target, not a blind retry.

Earlier teacher/student half blending added only .00002562 on F0. Wider H512 fixed blending was weaker on the inspected F0 panel. Learned teacher weights gained .000032 in calibration but lost .000120 on separate exposed components versus equal averaging. Current evidence therefore favors a few strong sources and simple fixed mixtures over seed proliferation or high-flexibility stacking. Conditional uncertainty is not an automatic mechanism kill rule; allocation ended at the user's instruction.

## Quantization and CPU headroom

Q8 is the default supported route. Exact Python3.11 archives have native ABI, finite-output, warm-up, causal state-reset and repeatability receipts. The competition CPU is AMD EPYC9224, one vCPU,60minutes,16GB RAM,20MB ZIP. Local Intel measurements alone cannot establish deadline compliance.

C provides a real36m36s calibration point. A paired C/E/C local experiment measured E/C callback ratio1.3911, giving a50.92minute capacity heuristic. That leaves about9minutes under the competition deadline, but it is not a full EPYC guarantee. D has three-model technical callback evidence without a matched paired runtime forecast. [Paired runtime receipt](z5_blend_runtime_pair_r1/PAIRED_RUNTIME_REVIEW_R1.json).

The available Q4 rounding screen lost about .00180125 on its small gate; it used a Q8 backend and does not demonstrate native INT4 speed. More members are justified only when their measured conditional gain survives quantization and cost. Mixed precision,5–7bit storage, quantization-aware fine-tuning and native packing remain recorded ideas rather than established gains. Exact shared-backbone inference approximately halved local callback cost with parity, but its head mixture had mixed F0/F1 quality and lacks final EPYC validation. Reducing allocations, sharing exactly identical preprocessing/trunks and optimizing gate matvecs are valuable when they preserve predictions; they do not supply a new predictive signal by themselves.

## Fixed trials available at the stop

- E: .75*C+.25*own Z5, four Q8 members,16,805,643bytes; SHA2aed2137c347344146e3481c83388d64bd5060b1c13bb66554b0bf9cc0027024.
- D: .25H390s3+.50Z4+.25Z5, three Q8 members,13,178,260bytes; SHAbf9f48dbb05668181038d28386a076774d41146efb7c4f1680fcd724a303df09.

Both exact archives passed independent technical review. Their clean-CV/public quality is unmeasured; original Z5 i16 public .6664 cannot be assigned to its Q8 version or either blend. ZIPs remain local ignored weight-bearing artifacts; manifests, profiles and reviews stay in Git. Human uploads only. [Root trial release](../../submissions/10-12-best-r2/ROOT_UPLOAD_RELEASE_R1.json), [independent package review](independent_reviews/submissions_10-12-best-r2/D_E_TECHNICAL_REVIEW_R1.json).

## What would distinguish an actually better solution

The bottleneck is obtaining useful complementary signal with reliable evaluation, not filling60minutes. Recipe/family differences, initializer ancestry, teacher target construction and fitting membership must be matched before attributing a gain. Reused fold scores assess robustness but do not become fresh confirmation. Complete all-data successors must implement the evaluated recipe faithfully, including actual fixed checkpoint, preprocessing and teacher ancestry; all-data training scores cannot validate the final model.

For future reference, model/precision selection is a discrete cost-constrained problem and blend weights are continuous. Optimize the exact mean of the two official clipped, weighted correlations with nonnegative weights and a measured package/runtime budget; shared-trunk costs are not generally additive. Covariance analytical weights apply only to an unclipped, fixed-weight single-channel correlation surrogate. They are not an exact solution to the clipped two-target competition metric. Fit combining rules on separate calibration groups and assess them on other components/folds; regularize correlated sources. The current frozen equal-weight comparisons avoid claiming an optimum from an exposed weight search.

Primary literature supports evaluating models by their contribution to the ensemble rather than averaging everything: [Caruana et al., ensemble selection](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf). [Beyer et al., patient and consistent distillation](https://arxiv.org/abs/2106.05237) motivates longer distillation with consistent teacher/student views in vision; it does not establish that more epochs improve these market streams. Our longer scratch schedules failed and cross-fold epoch effects differed, so the local evidence takes precedence over a transferred headline.

The admitted seed4 confirmation and Kaggle matched-parent treatment are being drained. New event/residual objectives, causal representations and other previously registered directions retain their evidence and unfinished status in the ledger. No new experiment or submission policy search is authorized after the final archive.
