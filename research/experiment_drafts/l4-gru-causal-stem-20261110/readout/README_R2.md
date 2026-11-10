# Causal-stem fixed-endpoint readout R2

`readout_stem_r2.py` re-exports the already-completed final checkpoints and
scores them with the canonical scorer. It does not train, retry a fit, or alter
any fit output. R1 remains frozen. R1's native attempt stopped during
`validate_admission`, before creating its owner receipt or output directories:
the packet compares raw JSON against the normalized configuration hash recorded
by the trainer. The source loader removes explicit `model.input_stem_kernel: 0`
from a plain recipe, so those two valid identities differ.

R2 keeps both identities: the raw config bytes must match
`raw_config_sha256`; after confirming the source capsule, R2 imports
`sequence_config.load_config` only from that pinned source's `src` tree and
compares its normalized configuration hash with `config_sha256`. The effective
plain kernel check uses `model.get("input_stem_kernel", 0)`. The fixture
`config_normalization_fixture_r2.json` binds the real `plain_s203301` raw hash
`0353a460…d03d467` to normalized hash
`322c24cb…fc905e` for commit
`336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2`.

Before importing the source config loader, R2 runs the source capsule verifier
against commit `336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2`, source SHA
`b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188`, and
manifest SHA
`482d13529e8c65386e6f74e7e81f607eeb3b0ed0795beff44902bf703e7d342e`. The
source capsule, configs, fit receipts, split, and completed checkpoints remain
unchanged. Admission still binds the ROOT-approved six-cell profiles, exact
GPU UUIDs, raw config hashes, normalized hashes, and receipt bytes.

## ROOT launch

After review, ROOT deploys this new receiver as
`/data/lc-epanov_s/tabdata-stem-20261110-r1/readout-receiver-r2/readout_stem_r2.py`.
R2 reads the six on-time results from
`control/root-fits-r1/{cell}-result.json` and uses the original clock and
admission. It writes only to `control/readout-r2/` and
`runs/{cell}/final-readout-r2/`; no R1 failure log or path is reused.

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  /data/lc-epanov_s/tabdata-stem-20261110-r1/readout-receiver-r2/readout_stem_r2.py \
  --root /data/lc-epanov_s/tabdata-stem-20261110-r1 \
  --source /data/lc-epanov_s/tabdata-stem-20261110-r1/source-r1 \
  --python /tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src \
  --r2-helper /data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py \
  --admission-sha256 <ROOT-provided-64-hex-SHA256>
```

The receiver enforces the existing first-profile cutoffs: fitting ended at
`first_profile_epoch + 10,800s`; collection ends at `+14,400s`. Each export is
limited to 1,200s, each CPU scoring stage to 600s, aggregate readout output to
2.5GB, and free disk must remain at least 4GB. Six readout workers use their
preassigned L4 identities. No new L4 fit is started.

## Outputs and interpretation

- Cell output: `runs/{cell}/final-readout-r2/` (`predictions/`, `score.json`,
  `readout-receipt.json`).
- Cell status: `control/readout-r2/{cell}-result.json`.
- Controller summary and owner: `control/readout-r2/complete.json` and
  `owner.json`; stage receipts are under `control/readout-r2/owned/`.
- Formal receipt shape: `readout_schema_r2.json`.

Every completed cell must contain 1,873 sequences of shape `[20000, 2]` in
float32 and 3,528,273 scored rows. The independent scorer must agree with the
producer's fixed-E6 full-valid monitor within `1e-5`. Paired contrasts are
withheld unless all six cells pass. This is a re-export of repeatedly exposed
validation, not a fresh confirmation or public-score result.

The focused local regression verifies the real raw-versus-normalized K0
identity, rejects a wrong raw byte hash and rejects a loader/config mismatch:

```powershell
C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest research/experiment_drafts/l4-gru-causal-stem-20261110/readout/test_readout_stem_r2.py -q
```
