---
kind: method
status: proposed
created_utc: 2026-10-14T02:33:52.201048+00:00
updated_utc: 2026-10-14T02:33:52.201048+00:00
hypothesis_ids: [H-MINGRU-TIMESCALES-20261014, H-STATE-INIT-20261014]
experiment_ids: [E-INITIALIZER-GRADIENTS-20261014-R1]
evidence_paths: [RECURRENT_TIMESCALE_ALGEBRA_R1.json, INITIALIZER_GRADIENT_TESTS_R1.xml]
---

# Memory timescales and complete-sequence initializer gradients

## Question or claim

A fast recurrent cell can begin with too little memory for this task. Separately, ordinary state detachment can prevent late losses from training a once-per-sequence initializer. Neither issue should be mistaken for evidence that cheap recurrence or learned initialization cannot help.

## Mechanism and method

[Tallec and Ollivier](https://arxiv.org/pdf/1804.11188), section2, propose initializing write-gate biases using a prior over temporal scales; their chrono initialization is an initialization choice, not a permanent gate constraint. [Feng et al.](https://arxiv.org/html/2410.01201v3), appendixD.4, report a benefit from retention-biased initialization in minLSTM. That experiment is not a minGRU or Connectome result. These primary sources motivate a separate minGRU initialization arm.

Our derivation uses the implemented minGRU convention: `h[t] = (1-z[t])*h[t-1] + z[t]*candidate[t]`, where `z` is the write fraction. At zero gate input, exact e-folding time `tau` rows requires retention `r=exp(-1/tau)` and bias `b=log(expm1(1/tau))`. The sign differs from PyTorch GRU's retention update gate. With bias0, the nominal timescale is only1/log(2)=1.443rows; an initial perturbation is multiplied by2^-100 after100rows. This is a constant-gate derivation, not an estimate for the fitted GRU or a stacked network with changing inputs.

Proposed comparison: preserve default minGRU initialization as its own control, then initialize write-gate biases over a declared log-spaced range from2 to10,000rows. Keep candidate/gate weights, architecture, seed, targets and fit budget otherwise fixed. This exact discrete-time/log-spaced construction is our variant, not a replication of the paper's uniform chrono distribution. First measure actual gate products on fitting-only sequences: the nonzero GELU encoder and lower-layer states can move gates away from their bias-only values. Additional persistent-conditioning interactions stay separate.

## Evidence and interpretation

The [scalar algebra receipt](RECURRENT_TIMESCALE_ALGEBRA_R1.json), produced by [verify_recurrent_timescales.py](../../../scripts/verify_recurrent_timescales.py), checks six timescales and full20k recurrence gradients against finite differences. For tau10,000, a2,000-row truncated gradient contains only20.05% of the complete positions99–19,999 initializer gradient for the declared linear toy loss. Tau5,000 retains32.23%. These are illustrations, not estimates of prediction loss gradients.

For an initializer fitted against a frozen backbone, the current trainer's normal hidden-state detachment removes every later-chunk gradient path to the initializer. Later predictions still depend on its earlier value, which makes a silent early-only objective easy to miss. [ARTBP](https://arxiv.org/abs/1705.08209) studies truncation bias generally; we are not adopting its stochastic estimator here.

[initializer_training.py](../../../src/wnn/initializer_training.py) implements a separate complete-sequence weighted prediction/distillation loss. It carries hidden-state gradients across chunks and uses [non-reentrant activation checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) to recompute activations. Targets and original-label/propensity weights enter only the loss; causal inputs alone enter the initializer. Rows0–98 are processed with zero loss. The first chunk contains the row99 prefix injection. A complete sequence contributes one optimizer update, unlike the ordinary ten-update TBPTT recipe. Its denominator is total sequence-batch weight; the ordinary trainer normalizes each chunk separately. Both objective normalization and update budget are explicit differences in this arm. The caller validates finite targets and finite nonnegative weights, recording actual weight mass.

[Five tests](INITIALIZER_GRADIENT_TESTS_R1.xml) passed in3.33seconds under1CPU/900MB: constant, first-row and prefix initializer losses/gradients match full unchunked reference; late-only loss reaches the initializer across multiple checkpoint boundaries; joint backbone updates are rejected by this initializer-only helper. These tests use tiny synthetic GRUs. The default backbone/trainer remains unchanged. A follow-up [six-test run](INITIALIZER_GRADIENT_TESTS_R2.xml) passed in3.44seconds after aligning the denominator floor with the trainer at1e-8 and adding a below-unit-mass weight-scale invariance check. The [independent source review](../../unified/reviews/INITIALIZER_COMPLETE_SEQUENCE_TRAINING_REVIEW_20261014_R1.md) predates that floor correction and permits a bounded actual-data profile, with no fitting or quality conclusion.

## Limits and validity

No gate-bias model or initializer has been fitted, graded or promoted. Actual-data CUDA profiling, end-to-end data/teacher/initializer ancestry checks, checkpoint reload and native export remain required before a fit. The complete-sequence helper is not wired into the trainer. Joint training remains a separate experiment with its own declared gradient horizon. Checkpoint recomputation must remain deterministic; dropout is disabled by the wrapper. The existing native initializer already transports the required FP32 parameters.

## Decision and next step

Root will profile the complete-sequence initializer-only path on a source-qualified fitting batch, then compare it with zero-state evaluation and separately admitted joint training. The main eight-shape screen keeps its frozen recipe. Root owns subsequent fitted checkpoints, predictions, position-bucket effects and native grades.

## Bounded profile packet staged

The [complete-sequence initializer profile packet](profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R1_README.md) binds the completed F0 seed-3 H256×2 terminal checkpoint, exact clean teacher/propensity/norm ancestry, and source-qualified B8 sampler member `(valid,37,44346154)`. Its [R3 standard-library preflight](profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R3_PREFLIGHT.json) passed under a 400 MB/one-CPU limit; the execution entry point repeats these checks and the unchanged strict target validator runs again inside the separately admitted worker. The declared resource envelope is 5 GB host, 3.5 GB PyTorch allocator, and 5 GB observed VRAM. Root retains GPU admission and launch; no initializer update, fit, or quality result exists yet.

## Source records

[Initial plan](PLAN.md), [initializer implementation](INITIALIZER_IMPLEMENTATION_R1.md), [minGRU reference](MINGRU_REFERENCE_R1.md), [one-sequence state decay](STATE_DECAY_DIAGNOSTIC_R1.md), and [first fixed-budget control grade](results/control_prediction_grade_r3/README.md) supply the local context. All existing folds remain exposed.

The separate [seed-3 complete-component initializer screen](initializer_subset_r1/README.md) specifies 258 whole F0 fitting sequences from 82 components, the shared order and update budget for all three learned arms, and the zero H256×2 CP0 control. It still requires root's actual R2 B1 resource profile and the concrete C++ native-consumer gate; the teacher and frozen backbone retain full fitting ancestry, and no subset fit/grade is claimed.
