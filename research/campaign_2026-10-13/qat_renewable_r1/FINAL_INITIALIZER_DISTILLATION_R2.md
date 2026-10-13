# Final-initializer target distillation R2 — implementation receipt

This code-only follow-up adds the opt-in `--final-distill-initializer` target source to the final-model-adaptation path. It addresses a target-source hypothesis after the fixed QAT blend comparisons; this receipt does not claim that the hypothesis improves quality.

## Frozen semantics

The flag is accepted only with a fixed all-data final adaptation, inherited initializer preprocessing, a dense FP32 plain GRU, a validated `--distill-cache-root`, WMSE/chunk normalization, and `--distill-alpha 0.9`. It rejects legacy teacher predictions, non-default update/detach or recurrent-credit modes, non-chunk denominators, origin replay, cache scaling, alpha schedules, bootstrap/reanchor, family targets, EMA/SWA/TD targets, and training-state continuation. The guard runs after initializer architecture/preprocessing inheritance so it checks the resolved model.

After loading the initializer checkpoint and preprocessing, the trainer clones an independent FP32 model in evaluation mode with gradients disabled. The clone is made before configuring QAT on the student and is never added to the optimizer. A per-sequence-batch hidden state starts at `None`; each ordered input chunk advances this teacher state under `no_grad`, then detaches it for the next chunk. The cache remains integrity-validated and supplies the existing dataset layout, but each collated main-target batch is copied and its teacher-target slices are replaced before the objective reads them. No mapped cache storage is written.

The main target is `0.1 * clamp(label, -2, 2) + 0.9 * clamp(initializer_prediction[..., :2], -2, 2)`. Existing main row weights remain `abs(clamp(label, -2, 2)) * main_valid`; any configured auxiliary losses are unchanged. Student QAT policy and learned-scale selection remain separate recipe options. The new provenance block records the initializer checkpoint SHA, the cloned FP32 state fingerprint, relevant source hashes, checked cache hashes, target formula, unused cached main values, teacher reset/carry semantics, and student QAT settings.

## Checks and limits

The focused regression uses a tiny CPU GRU to compare a full teacher pass with three causally carried chunks, verify the per-batch target copy leaves source cache values unchanged, perform one QAT student backward/update, and confirm the frozen teacher fingerprint/predictions do not change. Guard tests cover the accepted narrow mode and unsupported target/update/averaging/continuation variants. The trainer CLI help loads with the new flag.

The final focused run passed **13 tests in 1.79 seconds** under one CPU thread and a 1 GB memory cap. An earlier run had 12 passes and one test-only failure because a `1e-7` tolerance was below observed FP32 multiply rounding (`1.19e-7`); the assertion was corrected to `2e-7` and the suite then passed. No training loop, dataset, GPU, provider, or submission was run for this implementation receipt. Root owns the measured profile and matched FP32/QAT fits.

The implementation is based on `e1ab711ccb1fafc7dae6c0bdce246f92a1f50052` and changes only `scripts/train_rnn.py`, `tests/test_final_initializer_distillation.py`, this receipt, and the two navigation README files. The source and test SHA-256 values are listed below after final edits.

- Trainer source SHA-256: `d80aa6b1fabf491907a6c88904496c90abfc1d07d23532931d1edb674ab44315`
- Focused test SHA-256: `92562b5ee91471c3558240c899736d6d0c08f97d1b3a50f968a50250c415077a`

This is an opt-in experimental training treatment. It does not supersede the R1 learned-scale specification, change frozen fit recipes, or establish a public or held-out result.
