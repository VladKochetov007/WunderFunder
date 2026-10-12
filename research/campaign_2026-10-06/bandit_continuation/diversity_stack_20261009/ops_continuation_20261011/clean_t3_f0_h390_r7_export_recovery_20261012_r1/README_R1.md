# Clean F0 H384 export recovery R7

R7 reuses the frozen R5 source bundle and its nine immutable export specs, plus the
three fit-complete R6 version-2 checkpoints. It launches six exports only:
`fit_train` and `fit_valid` for seeds 3, 4, and 5, each at the already selected
`model_ep2.pt`. It does not fit, profile, or re-export any checkpoint.

The three R5 `full_valid` specs remain in the sealed catalog. Their logical output
counterparts were already exported locally and are bound by the s3 and s4/s5
terminal receipts, source/checkpoint/sidecar identities, FP32 precision, finite
value checks, and canonical row-group proof. Those local arrays are not claimed to
be byte-identical to outputs from a Kaggle device, and R7 does not upload duplicate
`full_valid` jobs.

The six Kaggle arrays total 5,731,200,768 bytes. R7 runs one unchanged R5 worker at
a time so every worker invocation obtains a fresh host/device inventory. Each
worker retains the R5 2-CPU, 8-GB RAM, 5-GB GPU-memory, 6-GB GPU-stop, and 1,800-s
limits. The bootstrap keeps a 20% host-memory reserve, a 4-GiB disk reserve, the
8-GiB collector cap, and a 300-s finalization margin. It refuses to start when a
resource check fails. The per-copy temporary disk bound is 2 GB; after the runner
has copied and hashed an array, R7 verifies its NPY header, ordered fit-membership
hash, finite values, and digest before removing only that verified transient
original.

R6 ended with Kaggle status `KernelWorkerStatus.ERROR` after all three fits had
completed, before any prediction worker started. Root’s collection receipts show
complete model/provenance outputs and the error was an insufficient-host-headroom
admission failure. It is not a training-quality result. R7 requires the source
kernel’s exact version-2 output mount, matching CP/provenance hashes, and a new
root-authored release receipt. Kaggle kernel metadata addresses the source by its
slug; it cannot pin the source version, so root must verify the read-back version
and mounted artifact hashes before dispatch.

The local `full_valid` arrays cover all 1,873 fold-0 valid feature sequences and
are feature-only. Fold 0 is exposed. This packet makes no held-out score,
competition-score, or public-score claim. It is an export recovery packet, not a
submission ZIP, and the embedded root release file is deliberately pending.
