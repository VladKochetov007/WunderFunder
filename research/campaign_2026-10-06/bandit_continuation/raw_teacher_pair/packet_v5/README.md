# E508 v5 raw-only baseline teacher pair

This is a private Kaggle profile-gate packet prepared locally from the frozen
production trainer at `69b2331825e193a070015ca4182872724595209a`. It is reviewable but has **not** been
pushed or run. This kernel runs the short profile and then stops for controller
review; it cannot launch either scientific teacher fit.

It trains fresh raw-only GRU(128, 2-layer) teachers for seeds 3 and 4. Both use
the exact fold-0 fitting sets: 10,535 train groups and 1,405 valid groups. The
468 evaluation valid groups and corrupt train group 528 remain excluded from
training. The legacy `--train-seqs 10607` value is only a ceiling; the campaign
manifest supplies the actual members.

Each seed uses four full epochs and the predeclared fixed zero-based epoch 3
checkpoint, with wMSE, original clipped-label-magnitude weights, AdamW/OneCycle
at lr 0.001, batch 16, chunk 2,000, and `--fast`. Initialization, distillation,
propensity weighting, EMA, SWA, mask weighting, and auxiliary heads are disabled.
The trainer never reads evaluation labels during fitting.

At startup the kernel recursively inventories `/kaggle/input`; it resolves the
fold-0 asset manifest, raw parquet tables, and audit-only E463 source manifest by
content identity rather than assuming a mount directory. The raw files are
symlinked, not copied. The Parquet validator reads the final eight-byte
length/PAR1 trailer, then hashes exactly the metadata footer plus that trailer.
This matches the portable registry builder's recorded range. It also computes
each full raw-file SHA-256 against the frozen portable registry, so matching
footer metadata alone cannot pass a changed data file. An atomic
`input_identity.json` records expected and observed hashes, sizes, mount paths,
and the Kaggle dataset version captured at preregistration. Kaggle's mounted
filesystem does not expose an attachment version identifier; full-file hashes
bind the observed bytes to the version-1 source registry. The E463 checkpoint is
checked against its manifest and is not used to fit this scratch baseline.

The kernel runs one short seed-3 profile only. It requires a finite multi-step
timing, the declared T4 device, enough host/device/disk headroom, and a projected
pair/export runtime below 6,300 seconds. It emits a measured profile result and
exits with `profile_ready_awaiting_controller_review`; no scientific fit is
admitted by this packet. On a dual-T4 allocation the profile job has a 4,200-second
provider-watch budget (at most 2.34 allocated GPU-hours). A later, separate fit
packet can use at most 6,300 seconds; its 7,200-second provider watch bounds the
worst-case dual-T4 charge at 4 GPU-hours. The profile wrapper enforces a
3,600-second process-group timeout, records durable phase files and logs, and
never retries an OOM automatically. Its parent uses nonblocking byte reads, so
the wall-clock deadline does not depend on output newlines.

The frozen follow-up recipe exports train and valid predictions in the existing
full target-cache row order as float32 `(rows, 20000, 2)` arrays. Its exporter
projects feature columns only and records the fixed checkpoint hash, prediction
hash, split hash, row groups, and label/mask-read flags in sidecars. Build that
fit packet only after the controller reviews the actual E508 profile receipt.

Prepared command (not executed):

```sh
./.venv/bin/kaggle kernels push -p research/campaign_2026-10-06/bandit_continuation/raw_teacher_pair/packet_v5 -t 4200
```

Packet kernel SHA-256: `8d5bc4a9f9c29173ef014e4b01a34cb7c6afdbdcc6125a88436f11ef23209f86`<br>
Trainer source SHA-256: `8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf`<br>
Fold-0 manifest SHA-256: `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`
