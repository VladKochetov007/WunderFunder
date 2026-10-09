# H230 seed54354 Vast fallback packet

This is a frozen, review-ready transport for one independently initialized
all-supplied-data H230 strong-parent run on an owned RTX 3090. It reuses the
reviewed H230 V4 trainer/source and exact profile-then-fit gate. The only
scientific argv change from the seed9 V4 contract is the fresh seed `54354`;
run names and output paths are attempt identities. The profile gate separately
requires an RTX 3090-class CUDA 8.6 device with native BF16 orthogonalization
support before the conditional fit begins.

V1–V3 packets and their failures are preserved. V4 fixes prediction-size
validation by including the `.npy` data offset and payload size, then streams
the file and compares its SHA-256 with the prediction sidecar. Its tests use
real `open_memmap` files and reject truncated payloads, trailing bytes, and a
wrong sidecar digest. The startup script records the pinned interpreter,
Torch/CUDA versions, CUDA initialization time, and actual device before it
installs worker dependencies or starts the profile. Image-pull duration must
come from the provider lifecycle receipt; the worker cannot observe that phase.

V4 keeps all ten frozen worker inputs. It stages `datasets/train.parquet` and
`datasets/valid.parquet` through the repository's existing signed prefetch
path and sends the remaining eight inputs plus the packet over SSH. Root must
provide the private URL plan and its public admission at launch. The private
URLs must stay in the protected runtime directory and out of committed files
and logs. The supervisor still verifies the complete ten-input manifest.

The packet is not authorized to create a Vast instance or launch a fit. Root
must refresh the offer and cost quote, verify the 120 GB disk plan and current
account scope, then seal a fresh intent, prefetch plan/admission, campaign
admission, and dispatcher receipt.
This packet requests at most `$2.80` all-in for the single owned attempt; it
does not authorize a credit top-up or touching any pre-existing instance.

The worker stages six supplied data/index files, three main-source cache
files, and the project's own propensity checkpoint. It preserves the clean-
fold training scope of the source cache in its receipts. A successful run
exports fixed zero-based epoch 3 FP32 predictions for the complete train and
valid target-index orders. Both splits include previously exposed groups and
provide no held-out score or public-score claim.

Build and run focused checks from the repository root:

```sh
.venv/bin/python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/h230_seed54354_v1/build_packet.py
.venv/bin/python -m pytest -q research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/h230_seed54354_v1/test_worker.py
```

The builder verifies the 57-file V4 source manifest and all small code/recipe
pins without rehashing the large local data or cache inputs. The Vast
supervisor binds and remotely verifies staged inputs; the unchanged V4
trainer then performs its mandatory raw/cache validation before fitting.
