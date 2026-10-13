# Independent LSQ-inspired learned-row-scale QAT review R1

**Decision: PASS for a root-admitted bounded actual-data GPU profile and the matched 10% uniform-W4 fit.** This clears implementation semantics for the stated contrast; it does not approve a quality claim, promotion, cross-seed conclusion, or public-transfer claim. Root retains fit admission and grade ownership.

The proposed contrast is one parent and one fixed recipe in both arms: H256 plain GRU, FP32, batch 16 with 1,000-row chunks, AdamW (`lr=1e-4`, `weight_decay=0.01`), OneCycle, no EMA/SWA, the same Q4 preset and data/cache/seed/update budget. The treatment is the opt-in `--qat-learn-scales` flag. The control should be named **nonlearned per-update absmax row scales**: its detached row steps are recalculated from each new latent-weight version, not held at their initial numeric values. Root confirmed that it is preparing a same-source control for this comparison.

## Scope and source identity

This review checked the LSQ addition authored at `05f543d38b6e4d05b0a85c913c42af6ae497f8a6` (parent `0321802cb32cabba8b3bc1c1f7f15fdac3bc58ed`), integrated as `6dde7d9d268475102849fda43b31b8e42162b3c8`, against campaign HEAD `fbf2632b0bbc28fa28f3425434be618a9477506e`. The reviewed implementation and test files are unchanged between the integrated LSQ commit and campaign HEAD.

| File | SHA-256 |
|---|---|
| `src/wnn/training_qat.py` | `d4ecfddb9027e7402b91d980b79ed8ebf67cef0b825006cdfd09571715c016e6` |
| `scripts/train_rnn.py` | `ea12e6b1016906c68580894a6b8b3362c4b1893b6e22b3818ba4dce0e41f4778` |
| `scripts/evaluate_qat_deployment.py` | `a8bb0f15d49f0a6a36ea9f37995ea5a1906814e56bed566a829fb4ff396a80ce` |
| `tests/test_training_qat.py` | `c6f162bbf8da46186a6440a65be207f749d9e6aa36bd049271dd108720fb8ce6` |
| `tests/test_trainer_resume_integration.py` | `c722d8656c329cddc710a23b7b1ffa858c45715032c13ac9b12ec327b745cec5` |
| `tests/test_evaluate_qat_deployment.py` | `44ae2666a3143f067079c481a614cd7f2c365982da438ab4a052e8cba5765fe8` |
| `LSQ_ROW_SCALE_SPEC_R1.md` | `59ff146f9a4832e0b43815cac895f2d5168d4e79b6408128ce9c664783704bbf` |
| `LSQ_ROW_SCALE_PROTOTYPE_R1.md` | `883f082533ca146b12bbec6d3f5a7641c7b06120c9c4ff0975749192feb64f56` |

I reviewed source, recipe semantics and tests statically; I did not run Torch tests, a GPU profile or a fit. The prototype reports 23 focused checks passing in 7.62 seconds, including numerical rules, exact export, resume and FastEMA integration. That is an actor-reported test result, not an independent rerun. The campaign's exact frozen command, parent artifact and source-overlay admission remain the controller's responsibility.

## Findings

1. **The scale derivative is a valid LSQ-inspired component for this transport.** `_LearnedScaleWeight.backward` uses the LSQ piecewise step derivative and `1/sqrt(input_columns * QP)` normalization for each row. The inclusive saturation boundaries agree with Equation 3. It intentionally uses the project's symmetric code domains `[-7,7]` and `[-127,127]`, rather than LSQ's signed two's-complement lower bound, and retains the existing identity STE for latent weights instead of LSQ's clipped weight STE. The method is therefore not a full LSQ reproduction, as the metadata and prototype say. The paper's ImageNet outcomes are motivation only; see [Esser et al., LSQ](https://arxiv.org/abs/1902.08153).

2. **The flag learns steps for every selected matrix/block in its preset.** `recurrent_w4` includes recurrent Q4 matrices and both Q8 blocks of `head.0.weight` (hidden and projection inputs). `input_w4_recurrent_w8` includes Q4 `gru.weight_ih_*`, Q8 `gru.weight_hh_*`, and both Q8 head blocks. Each selected output row/block gets a separate FP32 parameter. Thus the treatment should be described as learning all selected preset row steps, not “W4-only scale learning.” Root accepted this scope for the first uniform-W4 arm.

3. **The two arms share the initial represented model, while later scale rules differ.** Learned scales initialize through the same existing Q4/Q8 absmax quantizers, and export replaces all corresponding codes/scales with the learned representation. The fixed QAT path recomputes detached absmax/QP row scales after each latent update via the ordinary QAT cache invalidation. The phrase “fixed-scale control” in the prototype/spec is shorthand and can be read incorrectly; use “nonlearned per-update absmax row scales” in the fit recipe, table and grade. The metadata's `initialization` field correctly describes matching initial codes/representation; it does not claim that control scales stay numerically fixed. Learned steps are nonnegative, with negative values reset to the current absmax/QP rule, all-zero rows allowed to retain zero, and nonfinite values rejected.

4. **Optimizer, freezing, schedule and clipping are explicit.** Learned parameters are registered before strict resume loading, inherit the associated matrix's `requires_grad`, and enter the ordinary trainable optimizer set. Under the proposed AdamW arm they use the same base LR/OneCycle schedule and configured `weight_decay=0.01` (decoupled). Global gradient clipping covers the learned step gradients too, so a candidate scale gradient can change the shared clip factor applied to model weights; that is part of the learned-scale treatment and should not be described as a weight-update-identical ablation. SGD uses coupled decay. Under Muon grouping, scales are vectors kept in the AdamW group; optimizer-switch state handling also leaves them in the non-Muon group. No switch or Muon is in the proposed recipe, and those combinations lack a learned-QAT end-to-end test.

5. **Continuation and export bind the learned semantics.** The CLI defaults remain unchanged; the new flag is opt-in and rejects unsupported presets and non-FP32 QAT. `resume_contract` includes the flag and source hashes; scale parameters are registered before strict state load and are included in the raw continuation model/optimizer. FastEMA updates all model parameters and the actual trainer integration test covers FastEMA resume/export. `AveragedModel` cloning is covered by a focused unit test; no learned-scale SWA continuation integration test was identified, so keep SWA out of this recipe and verify that path before using it. Inference export filters the training-only scale parameters while carrying the exact learned codes/scales. The deployment consumer requires matching flag, scale-learning metadata, training rule, optimizer and decay provenance, shapes and represented weights.

6. **Quantization remains weight-only.** The trainer keeps input projection, activations and recurrent state in FP32. Native deployment retains its separately documented I16 input projection and dynamic A8 activation behavior. Learned row steps do not close that training/deployment gap, and no activation-QAT claim follows.

## Admission limits and follow-up

There is no implementation blocker to the specified bounded profile and matched 10% fit. Execute only after the root freezes the paired commands to the same checkpoint/parent, subset and split, teacher/cache lineage, seed, normalization, sample order, update count and source. Compare native-endpoint scores through the existing frozen grader after both exact-code exports complete. Label the 10% panel as exposed adaptation evidence and this first learned-scale result as one seed; do not infer pristine CV or public transfer.

Before any later SWA learned-scale recipe, add an actual trainer continuation/export integration check for SWA. If a later question specifically asks for W4-only scale learning, add a separate mode that leaves Q8 `head.0` (and mixed-mode W8 recurrence) nonlearned; that is not what the current flag implements. No measured scale-learning benefit or inference-runtime benefit is established by this review.
