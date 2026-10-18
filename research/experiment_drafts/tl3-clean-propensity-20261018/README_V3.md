# Strict propensity progress and recovery (V3)

V3 extends the opt-in `strict_fp32_v1` propensity path with update accounting and epoch-boundary recovery. It is a local engineering slice in `experiment/tl3-clean-propensity-20261018`, based on `0f17c2b6daab1df70186273738fb0fbadbe4193a`. V1 and V2 records remain preserved. The V3 validation and independent review bind this version separately.

The strict fit records optimizer attempts, calls, committed updates, non-finite rejection stages, learning-rate range, and pre-clip gradient norms. It rejects non-finite logits, losses, gradients, aggregate gradient norms, model weights, and AdamW state. Its checkpoint captures model, optimizer, loader/sampler, RNG, progress, normalization, source/data/split identity, and the runtime determinism contract, including CUBLAS and cuDNN settings. Resume verifies that the loader's declared row-group order exactly matches ordered `split.valid_groups`.

Recovery rolls back to the last completed epoch and replays an incomplete epoch. A terminal checkpoint can resume only to finalize status after checking epoch counters and the selected checkpoint files. Each caught failure is written to create-only attempt-specific JSON evidence, and a later successful resume exposes `fit_status: completed` while retaining prior failures. When a hard interruption leaves an update start without a terminal journal event, recovery reports the optimizer-step-call status as unknown instead of inferring that a step completed.

The progress journal is append-only JSONL, synced every 64 journal events and at recovery boundaries. A missing, truncated, or invalid journal fails closed; it is not repaired. An interruption can roll back model/optimizer changes to the preceding epoch checkpoint, but this is not mid-batch resume. The cost of event writes and checkpoint hashing has not been measured on campaign data.

Validation used tiny synthetic CPU fixtures only: 33 focused tests, Ruff, the repository training-structure check, CLI help, and `git diff --check`. No campaign data, large arrays, GPU, cache generation, or real fit was used. The initial macOS `uv` dependency-resolution failure and existing-environment fallback are recorded in [VALIDATION_V3.json](VALIDATION_V3.json).

This capability remains **unadmitted and not campaign-ready**. It has no real-data runtime or memory profile, GPU numerical check, strict cache-generation run, or scientific quality comparison. The modified recipe, canonical tied-lead integration, and complete 30-pair study remain separate work. This packet grants no training or execution authorization.

V2's initial setup limitation is retained: the repository-pinned CUDA Torch wheel was unavailable for macOS, so validation used `/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python`; no dependency install or download completed. The small ignored stub `.venv` remains unused.
