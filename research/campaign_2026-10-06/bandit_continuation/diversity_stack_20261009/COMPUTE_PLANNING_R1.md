# Compute planning: examples, not a rental request

User question October9: how many GPU hours, or RTX5090s for a three-hour batch, to train diverse parameter combinations and choose a blend? The user explicitly says not to rent. No provider mutation or spending is authorized by this note.

## What is measured

- This laptop has an RTX3060 Laptop GPU,6GB physical VRAM; research observed-VRAM cap remains5 decimalGB.
- One own full seed99 strong-parent fit completed in6,031.27seconds (1.675GPU-worker hours),62,320 attempted updates. This is that exact method/device/local allocation, not a5090 estimate. [Receipt](../continuous_execution_E529/H230_SEED99_LOCAL_FIXED_EP3_ROOT_TERMINAL_REVIEW_20261008_R1.json).
- Actual T4 profiles measured1.84501s/B8 outer batch for Muon GRU512 and2.86157s for GRU768. Four-epoch projections including1.2safety were3.67 and5.69T4-hours respectively; these were projections from230-update profiles, not completed fits. Their recipe/batch differs from the new GRU256 loss screen. [Receipt](../continuous_execution_E529/GRU_CAPACITY_V4_ROOT_POSTHOC_PROFILE_REVIEW_20261008_R1.json).
- GPU R2 and R3 failed before fitting. Corrected R4 completed both matched GRU256 FP32 loss fits on two T4s:220updates each,57.7s epoch0 plus23.1s epoch1 per arm (80.8s of recorded epoch training). This is a350-sequence screen; it is not a full-data fit or5090 estimate. Prediction export failed afterward on an unsupported CLI argument, so completed fit-plus-evaluation time remains unknown. [Independent tensor/update review](ROOT_GPU_R4_COMPLETED_FITS_REVIEW_R1.json). The7,200GPU-second hold is a ceiling, not measured consumption.
- No representative RTX5090 run and no matched actual-device JAX TPU comparison have completed. A TFLOPS or memory-bandwidth ratio is not an end-to-end training speed estimate for recurrent models with input/CPU overhead.

## How the count grows

Four optimizer/loss cells ×2folds =8fits. A broader illustrative grid with2optimizers ×3losses ×3architectures ×3schedules ×2activations ×2sample treatments ×2folds =432fits, before extra seeds, LR tuning, teacher stages or students. This example is not the admitted queue.

Training time scales with the number of trained members, not the number of blends tried from their saved predictions. Blend selection still consumes CPU/I/O and needs valid selection controls; it is not free or necessarily instantaneous. Distillation or neural boosting creates additional GPU fits.

The following is a **capacity scenario only**, assuming30minutes per fit including its prediction export on a5090, and30minutes shared setup/collection reserve inside a3hour deadline. The30minutes is not measured.

| Batch | Fits | Total GPU work | GPUs needed with2.5usable hours/card |
|---|---:|---:|---:|
| First2×2cells on2folds |8|4GPU-hours|2|
|24selected configuration/fold jobs |24|12GPU-hours|5|
| Illustrative exhaustive grid |432|216GPU-hours|87|

Discrete scheduling matters: each card fits five30minute jobs into150minutes. For24fits, if actual fit+export is20/40/60minutes, total work is8/16/24GPU-hours and the card requirement is4/8/12 respectively. A single job longer than the usable deadline cannot be rescued by adding independent workers unless its algorithm itself is parallelized.

For equal jobs: `GPU-hours = fits * minutes_per_fit / 60`; `cards = ceil(fits / floor((180 - shared_reserve_minutes) / minutes_per_fit))`. For heterogeneous jobs use actual per-recipe timing and a scheduling simulation; add cold staging/compilation, checkpoint collection and failures explicitly. Separate average expected completion from a hard deadline. Multiple workers can share a staged dataset only if host storage/RAM/I/O headroom and ownership permit it.

Practical allocation proposal: first benchmark one representative completed pipeline, then size a small sampled batch (e.g.24jobs), and confirm the useful members rather than exhaustively multiplying every dimension. No count above establishes currently available5090 inventory or a quoted rental cost. The [official Vast5090 page](https://vast.ai/pricing/gpu/RTX-5090) showed no current offer in the fetched public page onOctober9; this is not an authenticated inventory query or proof that the entire marketplace lacks cards. [Vast pricing](https://vast.ai/pricing) varies by marketplace supply. Any future cost estimate must use a fresh concrete offer including storage/transfer/setup; no rental now.

## Native GPU/TPU fairness

The [native comparison contract](tpu_gpu_benchmark/native_device_comparison_r1/CONTRACT_R1.md) holds mathematical logic and data fixed while allowing cuDNN/CUDA GPU and JAX/XLA TPU execution. Report synchronized warm steps/second and sequences/second, cold load/compile, complete fit+export wall time, validation quality and skipped updates. A separate native-precision comparison labels T4FP16 versus TPUBF16 explicitly and retains a matched FP32 correctness anchor.

Compare both one-fit latency and completed experiments per whole Kaggle allocation: twoT4 independent fits may run concurrently, whereas the eight-device TPU worker (configured `TpuV5E8`, actual device kind unverified) may use all8cores for one fit. Do not call that a chip-to-chip ratio. TPU compilation can make a short screen slower while a warm longer cohort is faster; the crossover must be measured. Report actual provider quota consumption independently of device-count arithmetic.
