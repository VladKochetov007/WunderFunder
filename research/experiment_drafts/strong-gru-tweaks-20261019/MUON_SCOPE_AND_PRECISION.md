# Current-source Muon scope and precision

This is a source-qualified optimizer adaptation for the clean Main T3 trainer. It does not claim to recreate the historical Muon `all` implementation exactly. The full-GRU recipe must record the flags used and keep this distinction in its ancestry.

The legacy default remains `--optim adamw --muon-scope rnn --muon-ortho-dtype auto`, with `--strict-muon-fp32` off. `auto` preserves the existing Newton–Schulz arithmetic: BF16 on CUDA and FP32 on CPU. Thus existing commands retain their prior optimizer behavior.

The explicit `--muon-scope all` applies Muon to GRU gate matrices as separate gate blocks and every other trainable 2-D interior matrix in the standard `inp.weight` / `head.2.weight` layout. The first input projection, terminal output readout, vectors and biases stay on AdamW. The scope rejects models without that standard input/output layout. This definition is deliberately local and reviewable; it is not evidence about an unsupported historical flag.

The opt-in `--muon-ortho-dtype fp32` performs every Newton–Schulz iteration in FP32 and fails if the optimizer receives a non-FP32 gradient. For the strict recipe, pass `--strict-muon-fp32`, which requires active Muon (directly or as an optimizer-switch target), `--precision fp32`, and explicit FP32 orthogonalization. On resume the canonical restore path rejects optimizer checkpoints that would downgrade the Muon group policy. The policy is stored as a boolean group field so optimizer state remains comparable and serializable without changing legacy defaults.

These flags do not configure global CUDA TF32 or matmul settings. A strict training launch must independently disable TF32/autocast and record those runtime settings under the project FP32 contract.

Validation here is source-only synthetic CPU work: focused grouping, dtype, update, switch and resume-policy checks; existing optimizer/dynamics/state tests; Ruff and the training structure checker. No dataset, checkpoint, GPU or quality fit was read or run.
