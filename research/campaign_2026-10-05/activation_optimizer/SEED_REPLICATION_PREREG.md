# E472: two additional fixed activation × optimizer seeds

Preregistered before any new fit: repeat **all sixteen E470 recipes** at seeds
271829 and 271830. Retain every activation and both optimizer recipes; do not
prune based on the original seed. This adds 32 fits to the original sixteen.
The same exposed 24-fit/16-test component cohort, fit-only normalization,
42 causal inputs, width32, one epoch, B4/chunk256, 474 updates, output anchor,
loss, LR0.001, decay0.01, clipping and official metric remain fixed.

The wrapper imports the immutable E470 trainer and sets only its explicit
seed. Source/cohort/data identities are bound by the guarded runner. Each
seed resets every arm to the same initialization and shuffled minibatch order
for that seed. The model layout and optimizer partition are unchanged.

Question: does the activation-by-optimizer pattern persist across training
initializations, especially sine versus plain GELU and Muon versus AdamW?
Report all scores and within-seed paired component contrasts. Cross-seed
means/SDs are descriptive; do not pool conditional component intervals or
treat repeated test components as new data. This is a replication of an
exposed-panel mechanism screen, not fresh generalization or public evidence.

Two sequential GPU workers, each 2CPU/6GiB RAM/2GiB allocator/3decimalGB
observed GPU stop and 1,200s cap. Parent limits remain 12CPU/30decimalGB RAM,
27GB soft, zero swap, local5GB observed VRAM. The successful E470 full-matrix
shape/resource probe supplies admission evidence because shape/code are
identical. Inspect explicit-seed initialization and update parity on CPU
before launch. A failed job stops this fixed queue, with no automatic retry.

The active goal is research until **2026-10-05 10:00 Europe/Kyiv (07:00UTC)**.
No new scientific fit will be admitted after that boundary; admitted jobs
are allowed to finish. The existing80-device-GPU-hour ceiling and local24h
block remain unchanged. Keep local/GPU costs and provider quota separate.
