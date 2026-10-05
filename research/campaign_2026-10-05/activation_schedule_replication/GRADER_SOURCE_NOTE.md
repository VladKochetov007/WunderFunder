# E480 independent scorer source note

The standalone scorer is ready for root review at
`score_activation_schedule_replication.py`. It reconstructs the two H64
two-layer RNNModel arms independently (exact GELU baseline and GELU-plus-sine
embedding variant), validates the eventual sealed run and attempt receipts, and
exports raw paired selected predictions, labels, masks, IDs, and shared
component-bootstrap draws without overwriting an existing output directory.

The registered design wording “exact six registered outputs per seed” is a
count typo: the full design has 12 cells per seed (2 activations × 2
optimizers × 3 schedules), 24 cells across the two seeds. The scorer computes
20 registered pair contrasts plus 2 schedule-shape interactions per seed,
for 44 total, including 4 primary warm-cosine-minus-constant-mean contrasts
per seed (8 total). Seed-level results stay separate; there is no pooled-seed
estimate. Bootstrap draws are common across all arms and effects within each
seed and resample cohort components.

The scorer binds the fixed attempt/spec identities
`e480-activation-schedule-replication-twentyfour-fit-v1` and
`e480_activation_schedule_replication_twentyfour_fit_v1`, with run directory
`runs/twentyfour_v1`. It validates the runner runtime and OOM snapshots, exact
artifact hashes, the immutable cohort and data, all 24 fit/test role records,
fit-only normalization and identity model buffers, checkpoint/source
identities, and the frozen E476 learning-rate receipt
`046f38f67792af73450bbbb958403834d948495e4ba64b263a2a9a2ff935e4ff` before
scoring. It rejects nonfinite raw labels before metric clipping and nonfinite
full-sequence predictions before scoring.

The actor manifest does not embed normalization statistics. The scorer
reconstructs the E470 normalization directly from the 24 raw fit row groups
using the pinned causal-feature and float64 population-statistic arithmetic,
then requires exact equality with the frozen E470 fit-only loader's
normalizer. The frozen helper source hash is part of the registered fit
identity, and the derived mean/scale hashes are saved in the score receipt.

Validation was source-only and synthetic; no actual test rows were read or
scored, and no GPU or fit was run. Commands:

```text
ruff check research/campaign_2026-10-05/activation_schedule_replication/score_activation_schedule_replication.py research/campaign_2026-10-05/activation_schedule_replication/test_score_activation_schedule_replication.py
python -m py_compile research/campaign_2026-10-05/activation_schedule_replication/score_activation_schedule_replication.py research/campaign_2026-10-05/activation_schedule_replication/test_score_activation_schedule_replication.py
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited python research/campaign_2026-10-05/activation_schedule_replication/test_score_activation_schedule_replication.py
```

All six bounded synthetic tests passed, including fail-closed synthetic
runner receipts, frozen schedule identity, formula/normalizer checks,
nonfinite-label rejection, and exact contrast/bootstrap checks.

Final source SHA-256: `9a83cf5e10c0019bc49106eeded55f79d2f6c49107f3aeb0ec4c9e1374d06862`.
Final test SHA-256: `6bce83c0aea32ee8cc0991bfcd74fe1e44cf94097f243cc58d7c7299f58df652`.
