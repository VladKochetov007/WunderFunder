# Independent admission review of two measured L4 GRU profiles

Reviewer: PR67 receiving execution agent, 7 November 2026. No SSH, GPU, training,
profile or new data access by this reviewer. ROOT retains the full native
receipts and is the approving operator. The review reads ROOT's captured native
profile summaries and terminal ownership evidence, not a scientific endpoint.

Disposition: **no measured resource or finite-work blocker for admitting the two
GRU cells independently**. ROOT must verify each complete native source/config/
seed/data/normalization/runtime/device identity and preserve its approved copy.
Mamba installation/oracle and profiles for other families do not block these GRUs.
This does not approve the receiving controller before its separate review.

| Evidence | gru_s101001 | gru_s101002 |
|---|---:|---:|
| Native updates | 20 | 20 |
| Native measured seconds | 5.0464985445 | 5.0450750776 |
| Real and forwarded rows | 640,000 | 640,000 |
| Objective-selected rows | 636,832 | 636,832 |
| Parameters, all trainable | 950,274 | 950,274 |
| Peak allocated bytes | 1,200,361,472 | 1,200,361,472 |
| Peak reserved bytes | 1,306,525,696 | 1,306,525,696 |
| Device total bytes | 23,670,685,696 | 23,670,685,696 |
| Peak host RSS bytes | 4,423,458,816 | 4,406,255,616 |
| Errors | 0 | 0 |
| Initial label objective | 2.0007147138 | 1.8315152772 |

The work counts match two complete B16 passes through 20,000-row sequences:
`2 * 16 * 20000 = 640000`, and `32 * (20000 - 99) = 636832` training rows after
warmup. Reserved VRAM is about 5.52% of device capacity. The ownership records
report native exit zero, no timeout and approximately 21.22 seconds total owned
command time, distinct from the trainer's five-second measured phase. Initial
losses are finite warmup diagnostics, not quality or seed-ranking evidence.
The reported nonzero unused distillation numerator is multiplied by KD weight
zero; ROOT's config/teacher check must retain that label-only identity.

The summary includes `config: null`; it cannot substitute for the complete native
identity. ROOT must check the actual frozen runtime-r3 B16/chunk2000/accumulate1
config, fixed 12 epochs, fitting seed, exact source inventory, physical GPU UUID,
strict FP32 environment and 10,325 train/zero valid fitting membership. ROOT
reported an actual separate CPU audit of all 1,873 valid sequences and 3,528,273
selected rows. This review has not fetched that complete remote audit receipt.

The revised runner must preserve the initial profiles' runtime environment.
ROOT independently caught an attempted forced `CUBLAS_WORKSPACE_CONFIG=:4096:8`
that would disagree with the profile's absent value; the builder is removing it
before fitting. Canonical profile admission still compares the full identity.

Local captured evidence under
`C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/l4-rpc/results`:

- `010-diagnostic.json`, SHA256
  `d1d4efca107b5e7fcc7856a58d63609eb1809ef3b54e22d94af343d4d1ced3fb`:
  two completed receipt-owned native commands, exit zero, exact cell/config/output
  paths and eight-thread worker environments.
- `011-profiles.json`, SHA256
  `c11e936a7bf8a4510ec826d70e3685374a8112443565801c832e3320e766e5e4`:
  captured profile files, receipt excerpts and successful-update observation.
- `012-profile-summary.json`, SHA256
  `68ef4a186a689c60493cfcc9d4e382cedef9cf8fc8dc093b950d405390269aa9`:
  two actual `SUMMARY` JSON records from its `output` field, reviewed above.

The two fixed 12-epoch fits must remain within the root-owned 24-hour campaign,
starting at actual first-profile epoch 1794071266.717658, with the final hour
reserved for collection. A short warmup profile does not establish later-step
stability or full-validation peak memory. Keep reporting validation each epoch;
the primary endpoint is complete only after the declared final epoch.
