# B300 throughput engineering R2

ROOT admitted these five new 50-update profiles after the first R1 shape matrix.
They create zero scientific fits. Stage this folder at
`/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/throughput-engineering-r2`.
The unchanged S3 scientific source is commit
`6ab2f679e70928f913b97d5a29f0d1e09ceacaa3`; all commands and config SHA256 values
are in [manifest.json](manifest.json), SHA256
`a29bbe001d7fd0b768113ebd6732e5e59e6076c07be7d78d5e99c6e01f9c66ab`.
All remote argv/environment paths use POSIX separators and all config hashes
were checked after generation. R1 and the live receiver are preserved.

| Recipe | GPUs | Global chunk mass | Updates/epoch | Updates/30E |
|---|---|---:|---:|---:|
| labels W2/B256/A4 | 0,1 | 2,048 | 420 | 12,600 |
| complement W2/B128/A4 | 2,3 | 1,024 | 820 | 24,600 |
| labels W1/B256/A4 | 4 | 1,024 | 820 | 24,600 |
| complement W1/B256/A4 | 5 | 1,024 | 820 | 24,600 |
| labels W2/B128/A4 peak | 6,7 | 1,024 | 820 | 24,600 |

Every profile has a 900-second external owned-command bound. Discard the first
20 successful updates and measure the last30. Stop a profile if any rank's actual
reserved memory exceeds 90% of that GPU's actual capacity; B256's memory is an
unmeasured expectation until receiving evidence exists. Record all-rank memory,
optimizer time/fraction, preclip norm/clipping and complete native wall time.
The peak recipe explicitly sets warmup_fraction=0; the other four retain 5%.

Use each manifest command directly with the canonical
`scripts/train_sequence.py --profile-updates 50` and the listed environment,
through ROOT's existing owned-command helper. Make the short TMPDIR first;
it avoids the existing long-path AF_UNIX failure. Use fresh output directories.
Complement commands include the immutable fit reference cache; labels omit it.
Do not use the frozen four-config wrapper to admit these new config identities.

All profiles retain T768/L24/context1024/chunk256/A4, BF16/BF16x9, original
optimizer and LR, fitting membership and chronological validation. The optional
S4 Muon batching source is not included. Larger batches change optimizer update
counts and schedule exposure explicitly; a later scientific fit needs a new
recipe admission. W1/B256 and W2/B128 have equal global batch and A4 temporal
grouping, with expected numerical reduction/batch-shape differences.

Rank0 globally reduced row counts are counted once. For global batch256, first20
forward 5,120,000 real rows, last30 forward 7,741,440, and all50 forward
12,861,440 (12,785,408 selected). W2/B256 doubles those counts. Full four-chunk
windows contain global_batch*1024 real rows; the tail contains global_batch*544.
Compare measured real rows divided by summed window time, and preserve native
loading/compile overhead separately. Prepared counts do not establish speed or
quality. No actual R2 GPU outcome is claimed here.
