# E476 source and CPU preflight receipt

This packet is source-only. No full fit or GPU command was launched.

The CPU design tests passed:

```text
ruff format --check research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py research/campaign_2026-10-05/activation_schedule/tests.py
2 files already formatted
ruff check research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py research/campaign_2026-10-05/activation_schedule/tests.py
All checks passed!

python research/campaign_2026-10-05/activation_schedule/tests.py
E476 bounded CPU invariants passed

python research/campaign_2026-10-05/activation_placement/tests.py
E474 activation placement CPU invariants passed
```

The inherited E474 tests check exact base parity, state layout, causal prefix
invariance under future-input perturbation, chunk/state carry, reset behavior,
and activation finiteness. E476's tests check the 61,058-parameter layout,
AdamW/MuonAdamW grouping (exactly four GRU matrices, three gate blocks),
E470's unchanged `need_prediction`-weighted loss, and LR trace boundaries and
matched arithmetic sum. A seeded-order invariant checks that full-fit row-group
receipts use the exact traversal used for training and rejects any non-fit
group.

The real-data, fit-only CPU one-update matrix passed for all 12 factorial
cells:

```bash
python research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py \
  --device cpu --preflight-only \
  --output research/campaign_2026-10-05/activation_schedule/preflight_cpu_v5
```

Receipt: `preflight_cpu_v5/fit_manifest.json` reports `preflight_passed`, 12
cells, one optimizer update per cell, all finite losses, at least one changed
parameter tensor in every cell, 61,058 parameters per cell, and
`exposed_test_groups_loaded=false`. It records exactly 24 fit groups
and hashes for all included code and data. All recorded source hashes were
recomputed against the current files after the preflight and matched; this
includes the finalized E474 trainer SHA-256
`40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068`.
`preflight_cpu_v5/lr_traces.json`
stores all 474 values for each protocol and their SHA-256 digests. Means are
.001 for constant peak and .00055 for both warm-cosine and constant-mean;
warm-cosine and constant-mean sums match within floating arithmetic.

CPU cells report CUDA peak fields as null, as expected. The source resets CUDA
peak statistics before each actual-data CUDA preflight cell and each full-fit
cell, then records per-cell maximum allocated and reserved bytes. No CUDA
device was used in this receipt, so no GPU-memory bound has yet been measured.

The actual-data CUDA probe is callable but intentionally unrun:

```bash
python research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py \
  --device cuda --preflight-only --output <new-empty-output-directory>
```

That command makes one fit-only update per cell, proves each update changes
parameters, and records CUDA allocated/reserved peaks per cell; it is not the
12-fit command.
Independent source approval, root-owned grading, resource admission, and the
CUDA probe remain root's responsibility before any full-fit launch.
