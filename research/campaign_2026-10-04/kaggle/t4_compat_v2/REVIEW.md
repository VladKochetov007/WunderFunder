# E-408 accelerator compatibility evidence

The private two-T4 kernel completed both fixed compatibility workers. Each ran
the exact root `campaign_compatibility.py` source bound in its sealed spec: tiny
and representative GRU forward, backward, AdamW update, and full versus split
float32 inference. Both reported `passed`; representative split maximum error
was 7.450580596923828e-08. This is implementation compatibility, not a scientific
fit, cross-stack training equivalence, or full 211-input pipeline validation.

Observed GPU peaks were 600,834,048 bytes each, including context. Host process
tree RSS peaks were 1,486,196,736 and 1,422,721,024 bytes, within the two 2-GiB
reservations. Worker wall times were 35.43 and 7.71 seconds; starts were staggered.
Inventory and complete output logs are preserved in `output/`.

Kaggle torch was 2.11.0+cu128, CUDA 12.8, cuDNN 91900; both were Tesla T4.
The provider reported GPU quota 29.98 hours remaining (0.02 used), versus 29.99
before these probes; TPU remained 20.00 hours. Rounded quota differences cannot
identify a precise dual-device charging multiplier. The TPU probe remains queued.

The first attempt, preserved separately as `t4_compat_v1`, failed before any
worker because Kaggle automatically expanded a `.tar.gz` dataset asset. The v2
transport uses an opaque `.wnn` archive with a bound SHA-256; provider file listing
verified it remained intact. No raw data, target caches or credentials were sent.

Controller review is still required before scientific launches. The sampled
watchdog is not a provider-enforced GPU/RSS hard limit.
