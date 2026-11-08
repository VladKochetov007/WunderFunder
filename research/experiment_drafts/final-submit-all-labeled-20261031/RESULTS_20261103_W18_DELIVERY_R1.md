# Final W18 submission delivered

The remaining W18 model completed its frozen full-corpus fit and the actual ZIP
passed the unchanged Linux/Python3.11 CPU package checker. The release and all
seven primary assets were downloaded again on the node and hash-verified.
The exact published-original ZIP was also copied through the existing SSH
connection to Windows: SHA256, CRC and all eight member hashes passed locally.
Direct Windows GitHub downloads encountered TLS timeouts; this did not affect
the successful server-side GitHub round trip or the byte-identical local copy.

- [Upload-ready ZIP](https://github.com/VladKochetov007/WunderFunder/releases/download/tabdata-pr54-final-w18-20261103-r1/tabdata-w18-final-20261103-r1.zip)
- [Release, model/fit/build/check receipts and 13-role full checkpoints](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr54-final-w18-20261103-r1)
- [Training loss, LR and throughput](https://github.com/VladKochetov007/WunderFunder/releases/download/tabdata-pr54-final-w18-20261103-r1/training_loss_lr_throughput.png)
- [Shared resource telemetry](https://github.com/VladKochetov007/WunderFunder/releases/download/tabdata-pr54-final-w18-20261103-r1/resources_shared.png)
- [Monitoring and engineering-recovery proof archive](https://github.com/VladKochetov007/WunderFunder/releases/download/tabdata-pr54-final-w18-20261103-r1/tabdata-w18-final-monitoring-and-recovery-20261103-r1.zip)

The submission ZIP is5454663bytes. SHA256:
932700e4e10721829641870f18198fca06ba3f9c5f06be5541d70bf38bf880f9.
Local exact copy:
C:/Users/admin/Documents/Codex/reviews/tabdata-final-submit-artifacts-20261031/w18-ssh-verified-20261103-r1/tabdata-w18-final-20261103-r1.zip.

## Fit result

This is one served W18 GRU256x2 student, not the complete four-member mixture.
Its six frozen teachers and initialization ancestry cover13 exact trained roles.
Together with the preserved17-model archive, all18 planned scientific fits are
complete. None of the other17 was retrained during this recovery.

| Endpoint | Training seconds | Completed updates | Mean train objective | Nonfinite / skipped |
|---|---:|---:|---:|---:|
| Epoch0 |2240.3|7800|0.6356465971|0 /0|
| Epoch1, fixed delivery |2209.9|7800|0.6230432442|0 /0|

The owned W18 process exited0, without timeout, after4489.25seconds including
startup/checkpoint work. The fixed model_ep1.pt is4043687bytes, SHA256:
70cfefcfeb14b5a5e1f2a181fc7ddf15330a24aaf32e7f324b122fb54327daf8.
All12480 official train+valid sequences fit. Independent evaluation is zero.
Neither training objective is a validation/generalization score. No new public
score, expected improvement, or competition-platform acceptance is claimed.

## Actual serving admission

All10 gates passed: ZIP size, memory, CPU-time projection, frozen new-weight
numeric ceilings, old-reference fidelity and R11/R12 equality, old-reference
reset/prefix checks, and delivered-package reset/prefix checks.

On one pinned CPU under Python3.11.18, the20k-row measurement was14.609microseconds
per row; projection for1970 sequences is575.60seconds, against3600seconds.
This is a measurement-based projection on this host, not platform Duration.
Checker peak RSS was2570469376bytes, below16GiB. Serving uses q8vnni with the
unchanged fast activations and three verified native ISA libraries.

New-weight RMS errors versus CPU Torch were0.00632210 /0.00376464; p99 absolute
errors0.01407504 /0.00999868; maxima0.03188994 /0.02552474. The frozen numeric
ceilings passed. The historical pointwise allclose-at0.02 diagnostic remains
false and is preserved; no FP32 equivalence is claimed or threshold relaxed.

## Diagnosed engineering attempts

1. The final-only metadata reader was bounded at2MiB after measuring the actual
   694052-byte cache metadata. The generic512000-byte bound and all lineage,
   payload, membership and numerical checks remain. Seven regression tests pass;
   actual validation verified all four existing payloads. One60-update profile
   passed independently before the single scientific fit.
2. R1 packaging failed before ZIP creation because generic ancestry walking did
   not include cache-teacher branches. The qualified package builder independently
   re-audits all six roots and matches their exact recorded closure/checkpoint
   hashes, then retains the exact13-role gate. Actual Linux re-audit passed.
3. R2 built a ZIP but its callback compared package.run_reference with the
   nonexistent receipt.run_reference. R3 maps that field to receipt.run and keeps
   all three SHA checks. Twenty focused tests and independent review pass; actual
   Python3.11/native preflight passed before the final full checker.

Both failed attempts, their states, artifacts, stderr, source bindings and exit
receipts are preserved. R3 did not repeat fitting or modify weights, native
libraries, independent checker, numerical ceilings or quality-selection rules.
Every publication wrapper removed its own inode-bound0600 credential. The final
wrapper exited0. The GPU receipt confirms no training GPU PID from this W18
recovery remains; foreign processes were untouched. The original four-hour
deadline was preserved throughout.

Canonical terminal evidence is in
[recovery-20261103/terminal-receiving](recovery-20261103/terminal-receiving).
Human upload of the ready ZIP to the competition platform is the next step.
