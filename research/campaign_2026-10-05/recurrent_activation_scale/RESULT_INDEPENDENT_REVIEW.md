# E478 independent score-export review

**Result:** the six exported official scores and all three within-seed contrasts reproduce from `RESULT_paired_components.npz`. The review used no checkpoint or fit source and no GPU. E478 remains an exposed 16-component descriptive comparison, not evidence of unseen-component or public-score improvement.

The independent checker verified that the NPZ SHA-256 is `235790580187c69532338a7134d23de4e3c3d6b7d4f0f23152857805b59f3be9`, matching the hash recorded in `RESULT.json`. It failed closed on the expected 11-array inventory, 16 distinct component IDs and row groups, 16 full-length masks, selected-row counts, exact sequence/within-sequence index order reconstructed from those masks, the six registered `(seed, alpha)` keys, and expected array shapes/dtypes. The 26,990 selected raw targets and all six raw prediction arrays are finite before clipping. Each component's score-mask count matches the exported target and prediction row mapping. I did not reopen the validation Parquet, so this confirms mask/export alignment and result metadata, not a second derivation of the official row mask from source data.

I recomputed each pooled score with the official `global_wp_clip2_v1` implementation, then independently formed the centered per-component weighted moments and pooled them using the official `_score_component_moments` routine. The two official paths agreed within `2e-7`; all six pooled and per-target scores match `RESULT.json` within `2e-12`.

| Seed | alpha 0.0 global WP | alpha +0.1 global WP | paired delta | paired component 95% CI |
|---:|---:|---:|---:|---:|
| 271828 | 0.48467710 | 0.48737957 | +0.00270247 | [+0.00050912, +0.00592503] |
| 271829 | 0.48235345 | 0.49302830 | +0.01067485 | [+0.00673042, +0.01506593] |
| 271830 | 0.51974230 | 0.52422107 | +0.00447876 | [−0.00080305, +0.00846145] |

For each contrast, the review reran 2,000 paired component bootstrap samples with the registered fit seed and compared the interval, standard deviation, point difference, and nonpositive fraction to both the official bootstrap helper and `RESULT.json`. The independent bootstrap reproduced the official output exactly within floating-point tolerance. Matching required preserving the official helper's sorted `np.unique(component_ids)` resampling order; sampling the export's original component order with the same RNG seed would give a different interval.

The point contrast is positive at all three fixed fit seeds, with two of three exposed-panel intervals excluding zero. The third includes zero. Because these 16 test components were exposed, this is a conditional mechanism screen only. It does not support a public score claim or seed-generalization interval.

The reproducible source is [reproduce_paired_components_independent.py](./reproduce_paired_components_independent.py), SHA-256 `51bb22441d88cb68c61791ef5b5e207dd435090f405dcea3165c3b1179dc4fd1`. Command and checks:

```bash
.venv/bin/ruff format --check research/campaign_2026-10-05/recurrent_activation_scale/reproduce_paired_components_independent.py
.venv/bin/ruff check research/campaign_2026-10-05/recurrent_activation_scale/reproduce_paired_components_independent.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/recurrent_activation_scale/reproduce_paired_components_independent.py
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_SLICE=wnn-campaign.slice \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/recurrent_activation_scale/reproduce_paired_components_independent.py
```

Ruff format/lint and `py_compile` passed. The bounded one-CPU/2-GiB run returned `independent_npz_reproduction_passed`, reproduced all six scores and all three 2,000-draw intervals, and reported `GPU_used=false`, `checkpoints_or_fit_source_used=false`.
