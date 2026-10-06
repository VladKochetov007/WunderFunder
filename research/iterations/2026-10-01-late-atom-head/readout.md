# Late categorical residual: no material gain

The new head learned the categorical task but did not add a useful forecast improvement to the fixed strong ensemble. Four head-only fits and two paired inference passes completed in 313.22 seconds. The original frozen member's OOF predictions were exactly equal to the archived baseline on every evaluated row; its full ensemble WP was reproduced as 0.7023796676142182.

| Fixed ensemble | WP | Difference from incumbent |
|---|---:|---:|
| Incumbent | 0.702379667614 | 0 |
| Categorical residual, CE coefficient 0.5 | 0.702381378288 | +0.000001710674 |
| Identical residual, CE coefficient 0 | 0.702380944235 | +0.000001276621 |

The primary candidate difference has a conditional paired-chain 95% interval of [-0.000004164109, +0.000007843874]. Fold differences are -0.000002271331 and +0.000005824555. Candidate minus the equal-capacity CE0 control is +0.000000434053, interval [-0.000004756875, +0.000005924384]. The extra categorical supervision has no useful measured advantage here. Quiet-class-3 candidate minus incumbent is +0.000002718228, interval [-0.000030351609, +0.000036464735], again with opposite fold signs.

These are pooled weighted Pearson results on the same 1,614,484 scored rows, 860 sequences and 250 pure chains. Both folds have extensive adaptive exposure. The intervals condition on the one fitted seed; they do not estimate seed population variability, selection uncertainty, or competition transfer. This is a completed exploratory screen, not proof that all categorical representations are ineffective.

## What the experiment actually changed

The entire current shallow256 H2 model remained frozen. Both arms added the same 512-to-64-to-64 MLP and an unconstrained residual gain initialized at zero. The decoder removes the amplitude tilt induced by weighted CE and uses clipped target moments. Candidate and control differ only in the categorical-loss coefficient. Baseline forecast/state identity, paired initialization and epoch sampling order, frozen original tensors, target-free prediction, and exact original OOF output were verified. Moment ratios use FP32 in both arms; backbone and MLP use FP16 autocast.

The common dictionary was built from 4,314 original-train class-3 sequences whose chains share no validation member. Its 64 atoms cover 99.8918% of the training metric mass; the two actual fitting pools have CE support coverage above 99.99%. Candidate epoch-1 mean CE was 1.8008 / 1.7930 versus control 4.0865 / 4.3486. These training objectives have different optimization and are not held-out predictive scores. Each candidate had one AMP overflow skip (1,739 updates out of 1,740 attempts); each control completed 1,740 updates. No run was retried or selected by epoch.

Learned candidate gains were -0.01003 and -0.02094. A post-readout descriptive calculation from saved predictions confirms the residual was active: class-3 weighted member-change RMS ranged from 0.00183 to 0.00378 across folds/targets, with near-zero exact-unchanged fractions. Thus the score result is not simply an identically disconnected branch. That diagnostic is not another score query, a configuration choice, or evidence that larger gain would help.

## Decision and preservation

Close this exact recipe with `insufficient_evidence` for a useful quality improvement. Do not spend on its final-data training, native implementation, gain/width/CE sweeps, or seed rescue. The measured effect is negligible relative to the extra implementation and serving cost; this decision does not depend on an arbitrary universal significance threshold. The qualified student ZIP remains unchanged, and no new public score is claimed.

Peak observed owned GPU memory was 1,498 MiB. All six jobs and the CPU evaluator completed successfully; owned scopes were released. The vocabulary took 54.48 seconds separately. `preservation-complete.json` verifies 72 local/SSH files totaling 43,153,812 bytes before these closure documents, including all eight epoch checkpoints, predictions, source, logs and receipts. The sole pre-execution code correction canonicalized artifact paths. The anticipated batch-shape parity concern did not materialize: original predictions matched exactly.

Primary artifacts: `evaluation-01/report.json`, `evaluation-01/sufficient.npz`, `execution-01/result.json`, `finish-result.json`, `review.json`, `prediction-change.json`, `contract.md`, and `preservation-complete.json`. No GitHub push, public submission, or modification of the closed overnight archive occurred. A new recurrent representation remains a separate research question; this head screen does not authorize relabeling another small head change as a new architecture.
