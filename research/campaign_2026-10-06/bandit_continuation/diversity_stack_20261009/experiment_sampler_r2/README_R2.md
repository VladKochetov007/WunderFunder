# Diversity recipe sampler R2

R2 preserves the R1 recipe ideas and fixes their source-policy binding. R1
claimed FP32 Muon while hashing the live `src/wnn/optim.py`, whose CUDA path
casts gradients to BF16. R2 binds all candidates to the immutable 63-file D02
R2 source closure (`df81ef0de614de954249f1d37c9a55402a2e2c83`, source payload
SHA-256 `3d86386dcde2ea099610f949004ceb64dc9200fcf511333caf79539f2d62e8c0`).
The exact source implements FP32 Newton–Schulz matrix math for Muon regardless
of model autocast, while its optimizer builder supports both AdamW and Muon.

The audit also caught a scheduler mismatch: R1 described `pct_start=0.3` and
`final_div_factor=10000`, but the D02 R2 scheduler uses `warmup_frac=0.05` and
`final_div_factor=100` (with PyTorch's `div_factor=25` default). The R2 pool
records the actual OneCycleLR, WSD, and SGDR settings. Its matched quartet
remains AdamW/Muon crossed with WMSE/weighted Huber(δ=1); all four recipes share
the same source closure, initialization, sample membership, schedule, and
remaining controls.

The R2 source binding does not create an executable queue. The sealed D02 R2
GPU worker is still a Muon-only WMSE/Huber pair worker, and no recipe-aware
runner has been validated for the sampler's AdamW/Muon cells and broader
variants. Every recipe therefore records
`implementation_dependency.state=pending_recipe_aware_runner`, null launch
argv, `admission=blocked_pending_recipe_aware_runner`, and disabled queue
activation. `assert_recipe_runner_ready()` rejects a fit until a validated
runner provides exact argv. No fit, provider mutation, or device profile was
run for this code-only delivery.

The deterministic pool has at most 24 recipes. It preserves the four-cell
matched block, reserves requested mechanism coverage, canonicalizes aliases,
hashes source/data/ancestry/recipe identity, and writes JSONL plus a readable
table. The four-recipe manifest and 24-recipe preview are review artifacts;
both have `fit_allowed=false` and `queue_activation_allowed=false`.

Rebuild the pool from the preserved R1 ideas with:

```sh
python research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r2/freeze_candidate_pool_r2.py
```

Reproduce the four-cell manifest with:

```sh
python research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r2/recipe_sampler.py \
  --root . \
  --spec research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r2/sampler_spec_r2.json \
  --output-dir research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r2/manifest_example_r2 \
  --count 4 --seed 20261009
```

Use `--count 24` with the same seed to reproduce the bounded review preview in
`manifest_max24_review_only_r2/`. The tests load and exercise the actual D02
optimizer and scheduler modules: they verify low-precision gradients enter the
FP32 orthogonalizer, the builder constructs AdamW and FP32 Muon, OneCycleLR
arguments match the recipe, the source path is not the live main optimizer,
and the runner gate rejects pending recipes.
