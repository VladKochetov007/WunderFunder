# Student fit and grading results

These artifacts belong to the matched F0 seed3 screen. All folds remain exposed; completed fitting alone does not establish prediction quality.

- [First control terminal fit](control_fit_terminal_r2/README.md): successful14920-update H256×2 F0 seed3 artifact and exact trainer/controller evidence; validator-field diagnosis precedes export/grade, with no quality claim yet.

- [First control prediction and grade](control_prediction_grade_r3/README.md): complete fixed-checkpoint export and official468-group F0 WP0.6697776418601, with source/metric/ancestry gates and resource receipts. This one-epoch control is below the stronger historical different-recipe student and is not promoted.
## H256×2 control fit and post-fit gate

This folder preserves the actual completed F0 seed-3 H256×2 one-epoch fit receipt and the first post-fit-validator failure. The first failure did not invalidate or modify the fitted checkpoint: the validator incorrectly expected a serialized beta1 CLI argument that the frozen trainer does not define.

- [R3 failed validator attempt](student-frontier-f0s3-verify-h256l2-r3-attempt.json), SHA256 49783b739488bfb5c9b35ebb0530b967cc306239a5addd5d26b0f4c0c23ed5cc, and [its worker log](student-frontier-f0s3-verify-h256l2-r3.log), SHA256 3e3c5c6ab175d57577f54bb1d54e2f06c904527da47f5253eaa30ef2e292f823. The first error was trainer argument beta1 differs: expected 0.9, got None; the training attempt itself had already finished successfully.
- [R4 post-fit validation receipt](student_frontier_f0s3_h256_l2_r2_fit_validation.json), SHA256 1042e31a73456edd88f40a8bd03b11c91ff346907ace16c50abc10b58fecdf43. It confirms epoch 0, 11,936 exposed sequences, 1,492 batches, 14,920 attempted and completed updates, zero AMP skips, 14,920 recorded gradient-stat steps with zero nonfinite steps, and 16 finite floating-point model tensors. It preserves the original checkpoint SHA256 1baa583573089da3eec02bee9256c32e3caeae59e15f77c7475eea0717f006e6. No quality evaluation was run by this gate.

The repaired gate does not weaken source, split, cache, propensity, sampler or normalization checks. It checks that the contract specifies AdamW constructor beta1 .9, that the actual trainer has no beta1 CLI override, and that the already-pinned trainer dynamics source is unchanged. It matches the recorded PyTorch version, checks OneCycleLR default momentum extrema, and reconstructs the 14,920-step momentum path .95 → .85 → .95. This is a deterministic reconstruction from the fitted recipe's pinned code and runtime because the trainer does not serialize a per-update optimizer trace. The fit's per-step loss finiteness is also not serialized and is not claimed.

- [H256×1 first cache-handoff failure](h256l1_handoff_failure_r5/README.md): R5 stopped before reclaim on missing user-bus environment, after successful fit/profile; terminal worker and sequence receipts are collected. R6 is a command-only successor, with no model retraining.

- [H256×1 complete result](h256l1_terminal_r6/README.md):14,920 updates, full export and officialF0 WP0.6693463182; paired versusH256×2 delta−0.0004313, component95%interval[−0.0023043,+0.0013212]. One exposedfold/seed, with intact model/teacher ancestry and terminal resources; compact candidate retained, no winner.

- [H384×2 complete first-screen grade](student_frontier_f0s3_h384_l2_r2_fold0.json) and [grade validation](student_frontier_f0s3_h384_l2_r2_fold0_validation.json) record WP0.6716880442, paired gain+.0019104024 over matchedH256×2, conditional141-component interval[.0003020132,.0034776104]. [Full sequence](student_frontier_f0s3_h384_l2_r6_sequence.json) finished all10steps. One exposedF0/seed3/short14920-update recipe; independent result review/native grade and broader fits remain pending, no promoted submission.
