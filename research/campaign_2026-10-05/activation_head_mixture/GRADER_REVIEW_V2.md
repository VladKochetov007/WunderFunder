# Independent E477 grader review, version 2

**Disposition: the earlier label/mask defect is fixed, but I would not approve real scoring until the checkpoint normalization buffers are checked.** This review supplements the preserved `GRADER_REVIEW.md`; it does not change the fitted actor, scorer, test suite, or prior report.

The current scorer returns the full `(20,000, 2)` raw target array alongside the full `(20,000,)` `need_prediction & is_scored` mask in `read_test_sequence` ([`score_head_mixture.py`](score_head_mixture.py:436)). The evaluator uses that mask once when selecting predictions, and `score_sequence_moments` validates aligned full-sequence shapes and finite selected values before clipping. A bounded synthetic check monkeypatched the parquet reader, exercised this exact full-mask path with 20,000 rows and 1,429 selected rows, and matched the resulting pooled score to official `global_wp`. The same check exercised finite paired-contrast and four-arm interaction intervals on synthetic component moments. Clean-CWD `--help` and `py_compile` also pass. No real test labels, fit checkpoints, scoring run, or GPU were used.

## Findings

1. **Blocking: strict checkpoint load can replace the scorer’s normalization buffers without an identity check.** `ScoringRNN` initializes `mean=0` and `std=1` ([`score_head_mixture.py`](score_head_mixture.py:283)), while the input cohort is already normalized from fit-only rows in `read_test_sequence` ([line 455](score_head_mixture.py:455)). The checkpoint is then loaded strictly ([line 642](score_head_mixture.py:642)), but no subsequent assertion verifies that the loaded `model.mean` and `model.std` remain identity buffers. A checkpoint with altered finite buffers would therefore apply a second, unregistered transform to inputs. After loading each state, require exact zero mean and unit standard deviation, as the independent E474 scorer does.

2. **Runner evidence is improved and correctly allows an unchanged historical parent OOM baseline.** The scorer checks the attempt’s status, exit, signal list, teardown, exact runtime map against sealed source hashes, unchanged worker and parent memory-event snapshots, exact manifest artifact set, and artifact hash/size ([`score_head_mixture.py`](score_head_mixture.py:78)). The actual attempt’s parent OOM counters are nonzero but unchanged; the current equality check accepts that history while separately requiring worker OOM counters to remain zero. The sealed source and input files are also hashed before the grader reads test rows. One additional fail-closed improvement would be to run `validate_spec` on the seal and check the recorded systemd `unit_properties.Result`/`ExecMainStatus`; the present exact attempt-spec equality and digest do not independently validate the seal’s internal `code_sha256` field.

3. **The normalizer is independently derived from fit rows only.** The scorer recomputes fit feature mean and scale from the 24 fit groups and requires exact equality with the saved fit-only cohort receipt before reading test labels ([`score_head_mixture.py`](score_head_mixture.py:409)). It then applies that normalizer once to test features. This boundary is sound provided the loaded checkpoint buffers are confirmed as identity, as above.

4. **Causal inference and contrast construction look correct.** Each sequence starts with no recurrent state; the scorer carries state across contiguous 256-row chunks and calls `eval()` so the post-activation dropout is disabled. The selected raw predictions are finite-checked before metric clipping. Paired contrasts use component resampling, and each four-arm dropout interaction recomputes all four pooled scores under one shared component-count vector. Seed results remain separate and the secondary comparisons are labeled as such.

5. **The recorded fixed seed has a reproducibility detail across bootstrap helpers.** Pair contrasts call `paired_component_bootstrap`, which canonicalizes component identifiers with sorted `np.unique`; the custom interaction helper uses the cohort’s source order. Both are valid paired component bootstraps, but the same seed does not map each draw index to the same concrete components across pair contrasts and interactions. If the report intends one identical resample stream for every contrast family, canonicalize component order once and use one shared multiplicity matrix throughout.

## Checks

- Current scorer SHA-256: `cd32f9d52535feb31897758e15c794cacfbf6dd9801ecf76f0d6a7b9d8080791`.
- Clean working-directory `--help` passed.
- Bounded synthetic full-label/full-mask and metric/contrast checks passed.
- No real scoring, fit, test-data read, or GPU work was performed.
