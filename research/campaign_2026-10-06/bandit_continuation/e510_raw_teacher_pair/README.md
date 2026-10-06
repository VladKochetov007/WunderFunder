# E510 raw teacher pair packet

This directory contains the packet builder and worker source for the independent
same-fold raw-only baseline pair. No Kaggle packet is materialized yet because
the actual E508 profile receipt and its controller-reviewed SHA-256 have not been
supplied. The builder requires both values and checks the receipt bytes, the
no-fit profile status, exact fold-0 counts, finite timing, projected pair runtime,
allocator peak, and T4 identity before it creates a packet directory.

After the controller reviews the real E508 result and supplies its path and hash,
build the private packet with:

```sh
./.venv/bin/python research/campaign_2026-10-06/bandit_continuation/e510_raw_teacher_pair/prepare_packet.py \
  --e508-profile-receipt <reviewed-E508-profile-result.json> \
  --e508-profile-receipt-sha256 <actual-SHA256> \
  --controller-reviewed-profile-sha256 <same-actual-SHA256>
```

The generated packet remains private and is not pushed by the builder. Its
prepared Kaggle command uses a 7,200-second provider watch. The embedded worker
rechecks the receipt hash before input discovery, profiling, or fit work, then
runs one internal resource profile and the serial paired fits only if the actual
profile is within the sealed limits.

The teacher recipe is scratch raw-only GRU(128, two layers), seeds 3 and 4, four
epochs with fixed zero-based epoch 3, wMSE, AdamW/OneCycle at lr 0.001, batch 16,
chunk 2,000, and no EMA, distillation, auxiliary, or propensity losses. It uses
the exact fold-0 fit groups (10,535 train and 1,405 valid), excluding 468
evaluation groups and corrupt group 528. Prediction exports are feature-only
float32 arrays in the complete existing target-cache row order. The pair receipt
will include checkpoint/provenance identities and a realized `SOURCE_FAMILIES`
snapshot for controller pinning before any C1-C4 cache or student fit.

No experiment or cloud push is performed by local packet tests.
