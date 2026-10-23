# Dictionary-only child recipe binding

This folder freezes the local recipe for the next fixed F0/seed1001 arm:
`fg19_child_dictionary_f0_s1001_fit_v1`, unit
`computer-vision-homework-fg19-child-dictionary-fit-v1.service`. Its exact
source-qualified mode is `--representation-mode sparse` with `--loss wmse` and
`--mean-variance-objective wmse`. The trainer argument vector is recorded in
[`DICTIONARY_RECIPE_BINDING_V1.json`](DICTIONARY_RECIPE_BINDING_V1.json).

In this source, `sparse` preserves the signed values of the stable top four
absolute-magnitude coefficients from the fixed 16-wide learned dictionary,
then retains the 112-wide normalized-raw residual. Those 128 columns occupy
the active suffix slot; the remaining 128 appended columns are exact zeros.
The unchanged legacy 211-column prefix gives a common 467-wide input. This is
the sparse dictionary arm, not `ema_sparse`, causal EMA history, or parameter
EMA. The shared topology remains GRU 256×2 with a linear encoder, head layer
1, and the 25-output mean/logvariance layout; WMSE leaves the trailing two
logvariance outputs unused.

The recipe keeps the exact parent checkpoint at epoch 3 and its preprocessing,
the accepted shared dictionary and T3 cache, F0/seed1001 split and seeded order,
batch 16/chunk 2000, six fixed epochs/checkpoint epoch 5, and the unchanged
AdamW/OneCycle/WMSE target and weights. The 44,760 updates are an arithmetic
expectation only, not a result for this child. The registered dictionary
artifact remains the exact 22,557-byte file at
`data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt`, SHA
`9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659`. It must
pass its accepted-receipt and exact-stop gates and a fresh path/size/SHA check
before any future child start. The canonical source can fit and create a new
dictionary if that cache path is absent, so a missing or mismatched artifact
must fail closed before invoking the trainer.

The source binding is commit `68ff682977b822f38351aac0f621146f5f13b440`,
167 files, manifest SHA
`a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`; the
canonical source checkout was clean at that commit. Mode-critical component
byte counts and hashes are pinned in the recipe JSON. Local source inspection
confirmed the parser enum and feature construction; it did not run model,
source, profile, or unchanged test suites, and did not read or hash model,
cache, dictionary, or other payloads.

This packet is **local preparation only**. The prior EMA native run has exited,
but its actual terminal integrity acceptance and exact-stop receipts are still
unbound at the paths recorded in the JSON. The captured native exit is not
treated as acceptance. No executable `REMOTE_SUBMISSION.py` or wrapper is
present, and no staging, admission, worker action, fit, evaluation, or quality
claim is made. Root must bind both actual EMA receipts, perform its fresh
admission checks, then independently review any later rendered body before a
launch. The existing accepted shape profile is reused without replay.

The frozen resource contract remains one owned GPU worker on
`GPU-ec74220d-78d7-cc47-69fc-0de1fb580944`, one CPU per worker and two aggregate
CPU cores, worker memory high/max 7.5/8 GB with no swap, allocator 4 GiB plus a
1 GB reserve, zero loader workers, and the existing 24-hour runtime ceiling.
