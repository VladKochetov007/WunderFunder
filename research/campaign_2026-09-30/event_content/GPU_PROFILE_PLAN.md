# Conditional D1 profile: unlaunched

No GPU execution or competition targets are authorized by this implementation.
Root must first receive independent CPU grade, complete four clean control folds,
and amend conditional card timing/budget before any fit. Retain original 3 arms,
4 epochs, final epoch3, seed3, <=1080 GPU seconds/arm (0.9 GPUh total) unless root
freezes an equal revised cap before fits. No truncated-update arm comparison.

Future synthetic profile must instantiate each arm with input211/output23, h96,
GRU1, block64; batch32, full20k, TBPTT2000; actual student and FastEMA, AdamW,
outer FP16 autocast/GradScaler, inner FP32 memory. Use identical seed, features,
gates/counts workload, optimizer recipe, ten successful updates and EMA steps.
Log peak allocated/reserved/observed device memory, wall time and update/skip
counts. Root's loader/feature-building and total host/device monitors remain
necessary; synthetic core cost alone cannot approve real fit or deployment.
Budget <=5GB observed VRAM and root current resource contract; abort on ceiling,
nonfinite tensors/gradients, or inability to finish identical prescribed updates.
Do not silently lower batch/change accumulation for one arm.

Before fit, freeze feature-only TRAIN query/written-key Walsh exposure, aggregate
uncentered rank96 and eigenvalue tolerance, percomponent/perclass write counts,
empty histories and modulo128 alias multiplicities. This actor synthetic test
proves fullcycle rank and active gradient exposure only. Fresh-component endpoint
reproduction must preserve knownzero/nonzero/unknown denominators; no oracle
class/labels in state. No decoder required by this implementation.

Actual deployment requires exact Rust endpoint extraction before normalization,
raw22 equality with previous-known semantics, six state fields and sequence resets,
FP32 serialized decay/logit bounds/ELU/tanh arithmetic, full/chunk/stream parity,
and observed one-vCPU package/runtime. No native port is provided here.
