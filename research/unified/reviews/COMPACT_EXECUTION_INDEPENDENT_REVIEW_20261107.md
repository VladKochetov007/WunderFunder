# Independent review of the fresh compact execution cohort

Reviewer: PR67 receiving execution agent, 7 November 2026. This reviewer did not
implement these compact adapters. Review was read-only in the architecture
execution worktree; all outputs here live in the reviewer's separate worktree.
No SSH, source download, competition data, GPU, profile or scientific fit was used.

Disposition: **no source blocker found for bounded receiver acceptance/profiles**.
This is not native GPU admission, measured serving performance or scientific
approval. The builder's R2 inventory SHA256 is
`f5906c9346703d28afe957cd54c990e5413a44d61606eab89c114a5b0fc3832f`.
All 11 inventoried source/document/test files and eight config hashes matched
the inspected working-tree bytes over base `b080d3f0874d45a6112f409603a0a10f6ec48420`.

The reviewer independently ran the compact, canonical sequence model and contract
suites with existing `galam-test-env-20261027/Scripts/python.exe` (Torch 2.14.1 CPU,
OMP/MKL two threads): **28 passed, one required real-Mamba CUDA test skipped in
6.21 seconds**. These tiny correctness checks establish neither GPU parity nor
training quality. They cover future perturbation, full/chunk/one-row parity,
reset, normalization, attached/detached gradients, EMA recurrence, and recipes.

- GRU/local attention normalizes once through the fitting-only outer statistics;
  its branch buffers remain identity. GRU and exact bounded local attention carry
  separate causal state. Their gated fusion and current-row direct head introduce
  no future labels, time-axis normalization or cross-sequence state.
- The EMA scan implements the associative current-inclusive recurrence with
  incoming state and no division by tiny powers. First differences use only the
  preceding row. Its four-member rank-one ensemble and scalar embeddings are
  honestly labelled a custom TabM-style mean-trained adaptation.
- The Mamba wrapper uses exact causal convolution history and explicit initial/
  final SSD states; state/device/FP32 checks and architecture-owned detach are
  present. The genuine pinned backend remains mandatory. The reviewer did not
  execute or independently fetch the upstream CUDA implementation.
- The shared recipes preserve raw112, fitting-only all-fit statistics, original
  label weights/mask, fresh state, zero KD/hints and fixed final epoch 12. The
  report cohorts use the canonical scorer. No early-stopping policy is declared
  in this cohort, and validation remains previously exposed.

Receiver gates still required before each corresponding fit:

1. Exact official split SHA/membership: 10,325 train, zero valid fit, all 1,873
   valid, and 3,528,273 selected rows after warmup. General monitor membership
   checks do not replace this concrete source-qualified receiving gate.
2. Real L4 identity, actual package/environment manifest, finite forward/backward,
   normalization and fresh-state identity, all-rank/source admission checks.
3. For Mamba: installation of the exact pinned backend and its real CUDA test
   against every upstream full-forward block, full/chunk/reset/future behavior,
   FP32 carried state and finite backward. Failure leaves that arm unadmitted;
   it must not substitute a CPU or Mamba-1 stand-in.
4. Real-shape headroom/timing profiles for every family. EMA's four-member first
   expanded activation is approximately 180 MB at B8/C2000 in FP32, before
   autograd retention. Canonical math attention limits each query tile to 256
   rows and trims keys, so GRU/local does not allocate one full 2000-square score
   matrix. Neither observation is a measured peak or proof of L4/CPU performance.

CPU serving cost, native backend compatibility, throughput, 12-epoch wall time,
full-valid curves and final scores remain unmeasured by this reviewer. Primary
chronology comes from the official source boundary, never shuffled sequence IDs.
