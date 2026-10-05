# Independent review: E471/E475 three-seed recurrent activation result

The saved result contains all 18 planned fit scores and 21 within-seed component contrasts. Its recorded grader identity, `79fce07334a03d852941f160cf9903e7ad51fcb5a475c0d7656da1d04967805a`, matches the current `score_seed_replication.py`. Both E475 runner attempts are `finished`, exit 0, teardown verified, with no failure, and their recorded `result.json` hashes match the output files. The original E471 result is also tied to its successful sealed attempt. The grader binds each run to its sealed source/input manifest and checks current file identities, selected exports, checkpoint hashes, and official parquet target alignment.

I independently recomputed official global and target WP from selected exports for four arms per seed and reproduced the first two preregistered contrasts per seed exactly using 2,000 centered paired component-bootstrap replicates. This was a bounded CPU check under one CPU/2 GiB; I did not alter or rerun the grader. It verified twelve arm scores (four per seed) and six contrasts across all three seeds. The checked contrasts were:

- `m10__adamw - zero__adamw`: +0.006526, -0.010406, -0.000818 for seeds 271828, 271829, and 271830.
- `m10__muon_adamw - zero__muon_adamw`: -0.008393, -0.018628, -0.014395 for those same seeds.

The full result shows no stable broad optimizer conclusion on this exposed panel: Muon versus AdamW changes direction by seed for some alpha settings, while the positive-alpha contrast versus zero is positive for Muon in all three runs and near zero for AdamW in the third seed. These are descriptive within-seed patterns from one fitted realization per seed; intervals condition on the historically exposed 16 components. Do not pool seeds into a training-seed-generalization interval, rank a universal optimizer, or treat these scores as a public forecast.

The post-score class diagnostic reads `label_class_diagnostic_only` only from the fixed exposed test-node cohort, repeats those evaluation labels over each sequence's selected-row count, and computes per-class official scores. No class label is present in the fit path or used to route predictions, assign model weights, or construct an aggregate public score. Each class has four exposed components; there are no class confidence intervals. It remains a descriptive stratification, not a public-mixture estimate.

Artifact identities:

- `score_seed_replication.py`: `79fce07334a03d852941f160cf9903e7ad51fcb5a475c0d7656da1d04967805a`
- `THREE_SEED_ACTIVATION_RESULT.json`: `2b28b3252bf75560e105380ee44bf5fc775b6ea463e2b2d59f05ead70fb02a27`
- Shared metric helper: `b85eb93af079d1b003b1fcca2e474d39c5bb8942d95b90a699e13d8da4ea3011`
