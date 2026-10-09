# Deterministic diversity recipe sampler R1

This is a review-only recipe manifest generator. The frozen `candidate_pool_r1.jsonl` holds complete recipes, hashes, and coverage tags; the sampler reads the repository's actual trainer CLI choices from Python ASTs, rechecks code and data-manifest hashes, canonicalizes aliases, validates compatible combinations, and samples without replacement. It never launches training, edits a queue, contacts a provider, or grants admission. If a source or membership hash changes, it fails closed until a new candidate pool is reviewed and frozen.

The first four rows preserve the matched factorial: AdamW × WMSE/weighted Huber(δ=1), and recurrent-gate Muon × WMSE/weighted Huber(δ=1). The initial fit membership is the frozen 350-sequence train subset; its 28 eligible valid groups remain unused. With batch size 32, the final batch contains 30 real sequences and two zero-weight padding sequences, giving 220 scheduled updates over two epochs. Additional candidates are a bounded pool of architecture/width, activation, schedule, sample/fold, precision, loss-shape, optimizer, and interaction variants. The sampler reserves one candidate per requested mechanism tag before filling remaining slots by a SHA-256 rank derived from the sampling seed. It rejects counts below four or above 24.

Each recipe hash includes its trainer/model/optimizer source pins, dataset and split membership, fold, sample seed and order, parent/teacher ancestry, model dimensions, activation, optimizer, loss and Huber delta, schedule/update horizon, and precision. Names expose the optimizer/loss, architecture/width, activation, schedule, membership digest, fold, seed, precision, and a recipe-hash suffix. Historical folds are labeled exposed or unverified; the manifest makes no fresh-holdout claim.

The generated four-row `manifest_example_r1/` is the starting factorial. `manifest_max24_review_only_r1/` demonstrates the bounded sampler and its reserved mechanism coverage; it is a review preview, not an active queue. Each directory contains `sampled_recipes_r1.jsonl`, `raw_candidate_pool_r1.jsonl`, `sampled_recipes_r1.md`, and `sampling_receipt_r1.json`. Every row states `device_profile=pending`, `admission=not_admitted`, and `queue_activation=disabled`. The runner argv remains null until an operator binds the exact recipe to a reviewed runner and root admits it.

After integration into the repository, reproduce the four-row example with:

```sh
python research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r1/recipe_sampler.py \
  --root . \
  --spec research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r1/sampler_spec_r1.json \
  --output-dir research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/experiment_sampler_r1/manifest_example_r1 \
  --count 4 --seed 20261009
```

Set `--count` to any integer from 4 through 24 for a larger manifest. Changing the seed changes only the deterministic selection order among compatible non-mandatory candidates; the four matched cells stay together. The same command and frozen inputs produce the same JSONL bytes when the bound source/data files do not change.
