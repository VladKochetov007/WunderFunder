# CPU student theory and accounting audit R1

This independent, read-only review checks the derivations and transfer claims in [the CPU student theory note](../../campaign_2026-10-14/cpu_student_frontier_r1/THEORY_AND_DISCRIMINATING_TESTS_R1.md) and [shape table](../../campaign_2026-10-14/cpu_student_frontier_r1/SHAPE_ACCOUNTING_R1.csv). It is tied to `H-CPU-POLICY-20261014`, `H-STUDENT-LADDER-20261014`, and `H-STATE-INIT-20261014`. It performs no fit, profile, benchmark, score, or quality evaluation.

## Findings

The matrix coefficient formula is correct for the current `RNNModel`: the input projection contributes `211H`; each GRU layer has two `3H×H` matrices, or `6H²`; `head.0` maps the concatenated `2H` recurrent/skip vector to `H`, adding `2H²`; and the final two-output matrix adds `2H`. Excluding biases gives `(6L+2)H²+213H`. All eight CSV entries match this formula.

The unpadded native-Q8 logical matrix byte formula also checks out under the stated convention. Recurrent weights contribute `6LH²` bytes, Q8 `head.0` contributes `2H²`, I16 input projection contributes `422H`, and FP32 `head.2` contributes `8H`, totaling `(6L+2)H²+430H`. The H256×2 result, `1,027,584` bytes, matches the historical actual-shape accounting. Biases, scales, panel padding, scratch and retained copies remain outside this total; it is not a traffic or complete-working-set measurement.

For the proposed prefix MLP at H256/L2, `633×64 + 64×512 + 64 + 512 = 73,856` parameters, or `295,424` FP32 parameter bytes. Its two dense layers require `73,280` multiply-accumulates per sequence, which is `3.664` per row when amortized over 20,000 rows. This excludes prefix statistics, normalization and preprocessing, as the theory note states.

The recurrent-state claim is correct: an initializer perturbation propagates through products of the full state Jacobians. The update gate alone omits the reset-gate and candidate-state dependencies on the old state. The trainer detaches carried state at chunk boundaries (`scripts/train_rnn.py`), so later chunks cannot backpropagate through those detach points into the initializer. The theory note's 1,000-row example is illustrative; the PLAN/CONTRACT reviewed here do not freeze that as the screen's exact chunk size.

The overflow bound `255×127×1280 = 41,452,800` is below signed 32-bit range for the stated unsigned-byte/signed-byte dot product. It supports exact integer accumulation only; float epilogues and nonlinearities still need their own parity checks. The rank-`r` factorization break-even for a single `3H×H` matrix is `4Hr < 3H²`, hence `r < 0.75H` before extra execution costs. For a GRU layer with input width `I` and hidden width `H`, the minGRU paper removes hidden-state dependence from gate/candidate calculations, leaving about `2IH` matrix coefficients instead of `3H(I+H)` and a vector affine recurrence that can be scanned in parallel during training. In this repository's current `RNNModel`, the encoder maps the 211 raw features to width `H`, so the GRU layer input width is also `I=H`. minGRU's inference recurrence remains sequential, and its simplified transition is a distinct learned student.

The source transfer limits are appropriately stated: AMD's 4th-generation EPYC document supports the 1 MB private L2 and 256-bit internal data-path facts, but those facts only motivate testing; T-MAC supplies a lookup-table design reference for low-bit LLM kernels, not a recurrent-runtime prediction; contextual-RNN work motivates an input-conditioned causal initial state, not a task-specific gain; and the cited compact-RNN work is speech-model precedent, not competition evidence. None of these sources supports a new quality or target-host runtime claim.

Two wording improvements: specify “input projection and final two-output layer” in the sentence saying those terms scale only 2× from H256 to H512. The hidden readout `head.0` is `2H×H`, so it scales 4× with the other quadratic matrices. Define `I` as the GRU layer's input width when quoting minGRU's matrix count; it is `H` after the current input encoder. Also keep the 1,000-row TBPTT example distinct from whichever chunk/detach values are eventually frozen in the runnable recipe.

## Sources checked

- Local implementation: [`src/wnn/nn.py`](../../../src/wnn/nn.py), [`scripts/export_native_weights.py`](../../../scripts/export_native_weights.py), [`src/wnn/quantization.py`](../../../src/wnn/quantization.py), and [`native/src/rnn_fast.rs`](../../../native/src/rnn_fast.rs).
- Historical accounting: [`LOGICAL_Q8_WEIGHT_TRAFFIC_R1.json`](../../campaign_2026-10-13/team_models_r1/LOGICAL_Q8_WEIGHT_TRAFFIC_R1.json).
- Primary sources: [AMD 4th Gen EPYC architecture white paper](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-epyc-processor-architecture-white-paper.pdf), [Lu et al. 2017](https://arxiv.org/abs/1709.02540), [T-MAC v2, 25 March 2025](https://arxiv.org/html/2407.00088v2), [Contextual Recurrent Neural Networks](https://arxiv.org/html/1902.03455v1), [minGRU](https://arxiv.org/html/2410.01201v3), and [Learning Compact Recurrent Neural Networks](https://arxiv.org/abs/1604.02594).

