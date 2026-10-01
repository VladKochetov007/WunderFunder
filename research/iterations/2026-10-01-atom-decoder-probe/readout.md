# Fixed-weight atom decoder probe: negative

The predeclared corrected decoder did not improve the saved atom model. Two inference passes, no fits, evaluated four readouts of the same hidden states and probabilities. Each replaced the shallow member of the same four-member ensemble; the other three members and quarter weights stayed fixed.

| Readout | Pooled WP | Difference from original atom readout |
|---|---:|---:|
| Original | 0.701612632358 | 0 |
| Clip atoms only | 0.701556810617 | -0.000055821740 |
| Remove CE amplitude tilt only | 0.701456794314 | -0.000155838043 |
| Both corrections (primary) | 0.701387212843 | -0.000225419514 |
| Incumbent ensemble | 0.702379667614 | +0.000767035257 |

For the primary contrast, the paired whole-chain conditional 95% interval is [-0.000440214332, -0.000011012315]; fold deltas are -0.000210160703 and -0.000231899700. All four atom readouts underperform the incumbent on both folds. The quiet-class-3 contrast also has a negative pooled point estimate, with a wide interval spanning zero.

This closes the proposed post-training formula replacement. The isolated weighted-CE derivation remains valid, but a jointly trained model need not emit its isolated-CE optimum: its regression branch, mixture, and probabilities were optimized together. This result is compatible with compensation, not proof of that mechanism. It does not test training a new head with the corrected formula from initialization.

The population is 1,614,484 scored rows, 860 sequences, and 250 pure chains. Both folds have extensive prior exposure. Bootstrap intervals condition on these fixed single-seed models; they do not include training-seed variation, adaptive search, or competition distribution shift. No new submission candidate is justified.

Execution completed without retries: 29.66 seconds for the two GPU passes and 8.23 seconds inside the CPU evaluator. Generated checks verified original-forward/state parity and the isolated-CE correction oracle before execution. Exact incumbent statistics and row/target sufficient statistics were reproduced. Source and input hashes, compact predictions, copied source checkpoints, logs, and results are preserved locally and on beleriand: 35 files, 38,245,152 bytes in `preservation-complete.json` before this closure note.

Primary evidence: `evaluation-01/report.json`, `evaluation-01/sufficient.npz`, `execution-01/result.json`, `finish-result.json`, `contract.md`, and `preservation-complete.json`. The qualified `student_fixed4.zip` is unchanged. No GitHub publication or public submission was made.
