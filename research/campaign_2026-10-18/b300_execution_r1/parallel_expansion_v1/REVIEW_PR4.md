# PR4 field-age successor review

**Decision: conditional GO for a new, fit-only B300 successor; NO-GO for serving export until the streaming feature state is integrated and parity-tested.** This review uses the immutable source commit `a266b036b02e8e012c6eb31072291dad618a169d`. It does not claim that this review ran a profile or fit. Root’s separate density profile is an admission prerequisite and remains subject to its own receipt.

The original PR4 draft is a proposal for comparing `group_age` with `field_age`; the PR itself adds no implementation. The feature generators are present in the pinned source. The draft’s proposed 12-epoch WSD/peak-LR-0.001 recipe and historical FP16/two-epoch jobs differ from the current six-epoch R01 labels-only recipe. This campaign must therefore be registered as a new cold-start successor with new experiment IDs, not as a faithful completion or extension of PR4’s historical runs.

The native features are equal width. `group_age` emits one reset-local clock for each of seven raw-field groups, resetting when any field in that group changes. `field_age` emits 112 clocks, one per exact raw feature. Both emit `log1p(age)` with the source cap of 10,000 and start each sequence at age zero. The feature builders exist in NumPy, Torch, and `StreamingFieldAgeFeatures`; training assembles `raw112 + event99 + research_features`, before TBPTT chunking. The primary comparison therefore changes the age mechanism while holding suffix width, fit recipe, fold, seed, and data membership fixed.

The source’s plain-GRU `enc=linear` input projection adds `512 × width` trainable parameters. With the R01 baseline at 211 inputs and 2,210,306 trainable parameters, either age arm has 323 inputs and 2,267,650 parameters (+57,344). The matched PR14 comparison has 291 inputs and 2,251,266 parameters (+40,960): `causal_innovations:newest` retains the moment features while zeroing the newest 16-channel innovation suffix; `causal_innovations:none` retains that suffix. These exact counts follow the source projection formula and agree with the root-reported full-shape profile displays of 2.27M and 2.25M.

Use the canonical R01 six-epoch, fixed epoch-5, strict-FP32 labels-only recipe: hidden512/layer1 GRU; batch8/chunk2000; AdamW at 1e-4, beta2 .999, weight decay .01; OneCycle with .05 warmup; gradient clip1; WMSE with chunk denominator; seeded batch order and deterministic execution; train on the fold-excluded train+valid membership; no outer evaluation during fit; no teacher, propensity input, initializer, or inherited optimizer state. Each arm gets its own normalization fitted only from that fold’s training batches. Keep the full R01 checkpoint and export policy fixed across arms.

The feature-only command delta is:

```text
--research-features group_age --research-feature-control none
--research-features field_age --research-feature-control none
```

For PR14’s distinct matched control, use `--research-features causal_innovations --research-feature-control newest`; for moments plus innovation use the same mode with `--research-feature-control none`. Build the rest of each command from the frozen R01 `train_command` builder; the panel helper intentionally records only the feature delta.

The 30 paired cells per arm use seeds 1101–1108 on folds F0/F1 and 1101–1107 on F2/F3 (8/8/7/7), with seed-major order so each early wave spans all four folds. Pair members share fold, seed, GPU, and membership. PR4 and PR14 are separate comparisons, each with its own two arms and experiment identity. The panel records the PR lineage but marks both as successor experiments; it does not reuse a PR4 parent or claim a historical PR4 result.

**Causality is adequate for offline fitting, but serving is not ready.** Sequence feature generation compares each row only with earlier rows, resets at each sequence, and is performed before within-sequence training chunks. Verify prefix invariance and equality across whole-sequence, arbitrary-chunk, and one-row updates; include state reset and save/reload. The current `StreamingGraph`/`initial_state` rejects non-moment research modes, including both age modes, even though the standalone streaming helper exists. For `causal_innovations`, the streaming update path is implemented but still needs full-source/stream parity. Do not label any endpoint export official or serving-ready until the relevant streaming state is wired into the actual graph and those checks pass.

Admission is separately root-owned. While both old families are live, reserve four measured slots per GPU (two R01 and two PR15); each family releases its two only after verified canonical completion, inactive unit, empty cgroup, and no owned actors. Under a measured five-actor total cap, this permits one new fit per GPU when both are live, three when only one family remains, and no more than three new jobs per GPU after both drain. The helper assigns fresh jobs two same-NUMA CPUs from 80–159 and rejects stale density, owner, duplicate-cell, wrong-GPU, and over-capacity observations. These rules do not change the frozen watchers or controllers and do not themselves launch work.

Before fit admission, require the sealed source/data/split identities, clean cell ledger, source-built command, full-shape profile receipt for all four variants, aggregate resource receipt covering old and new units, and root-issued authority. Preserve the current R01/PR15 workers, known foreign GPU actors, and their resource ownership. If age wins offline, serving remains blocked until the streaming integration gate passes; if only an early fold or seed moves, report that as exploratory evidence rather than a campaign result.

## Source anchors

- [PR4 draft source contract](source_contracts/pr4.json) and [PR14 draft source contract](source_contracts/pr14.json)
- Frozen source paths at `a266b036b02e8e012c6eb31072291dad618a169d`: `src/wnn/research_features.py`, `src/wnn/training/setup_model.py`, `src/wnn/training/inputs.py`, `src/wnn/training/batches.py`, `src/wnn/training/evaluation.py`, `src/wnn/training/normalization.py`, `src/wnn/streaming.py`, and `src/wnn/nn.py`
- [Frozen R01 recipe](../deployed_review_v4/source/recipe/RANK01_RECIPE.json) and [canonical command builder](../deployed_review_v4/source/code/fit_recipe.py)
- [Pure panel/admission helper](code/panel.py) and [CPU tests](code/test_panel.py)
