# Age readout screen grader adapter (R1)

This directory contains an independent, admission-gated scorer for the six-cell
`age-readout-screen` panel. It reuses only the frozen official-score and paired
component-bootstrap helpers from the older online-relationship grader. Its
artifact contract is specific to this recipe; it does not import the old
grader's arms, lags, recipe status, output naming, or parent-cache assumptions.

## Frozen inputs

- Recipe: `research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json`,
  SHA256 `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`.
- Expected cells, in recipe order: F0/1001 A, G, F; F1/1002 A, G, F. All six
  frozen output directories and standard filenames must match the recipe.
- Official targets: recipe-pinned `cache/valid_targets.npz`, SHA256
  `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- Producer sidecars are independently anchored by
  `research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json`,
  SHA256 `0dd9f9cec9bf54a4c2abdaeb835ad13f099e201a4b578714838014e39eb4d11e`.
- The cache hash pass is pinned to its read-only authorization, spec, and completed
  guard: `CACHE_VERIFY_ADMISSION_R1.json` `052424d4d631bce24741ea0b2f75218618e78ca04b9be812ed3a616a42ccfff6`,
  `CACHE_VERIFY_SPEC_R1.json` `ebf1f1fa163f46bbae4e690bdf4507b4f4caca4351d75af42c57a18138fee8a6`,
  and `CACHE_VERIFY_GUARD_R1.json` `f5fec8727cdbfc449995dc3e50fc04af70432941870257ec8f7646d5a0e13e12`.
- Scoring helpers: `online_relationship_grade_core.py` SHA256
  `b8480f1ac9f9068e076363486076f1965eeb65c364099c891fa25b2dd03533da`, with
  `grade_prediction_bank.py` `fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e`,
  `src/wnn/evaluation.py` `96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e`,
  and `src/wnn/gmetric.py` `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa`.

## Admission and bundle contract

The CLI requires the root-created `wnn.age-readout-grade-admission.v1` receipt
and its independently supplied SHA256. It fails closed unless `status` is
`admitted`, `allow_partial` is false, the metric and recipe SHA match, the
ordered six `required_cells` match, and the receipt pins both the bundle
manifest and local cache-verification receipt. The receipt must also bind the
fit-source hashes, this adapter's source hash, the frozen score-helper hashes,
and the requested resource limits.

The Kaggle wrapper invokes the extracted package with the root receipt SHA:

```sh
python /kaggle/working/disk_scratch/age_screen/grader/age_readout_grade.py \
  --bundle-root /kaggle/working/disk_scratch/age_screen \
  --bundle-manifest /kaggle/working/disk_scratch/age_screen/bundle_manifest.json \
  --admission /kaggle/input/age-grade-admission/GRADE_ADMISSION_R1.json \
  --admission-sha256 <root-pinned-receipt-sha256> \
  --output /kaggle/working/disk_scratch/age_screen/AGE_READOUT_GRADE_R1.json
```

The root admission receipt is a separate read-only Kaggle input, outside the
archive, to avoid a circular manifest hash. The wrapper must stop if the
receipt or its pin is absent and must pass the same extracted package root to
the grader.

The bundle manifest is `wnn.age-readout-grade-bundle.v1` and maps each original
absolute `source_path` to a safe relative `bundle_path`, with exact SHA256 and
byte length. Duplicate aliases, path traversal, any symbolic link, unmapped
source identities, or changed bytes stop grading. Candidate outputs
are resolved from the frozen recipe paths. Parent prediction arrays and their
sidecars are resolved from the root's transport map. Candidate provenance,
checkpoints, split/subset manifests, producer sidecars, cache-verification
attestation, target archive, grader, and code dependencies are hash-checked
from the same map.

The local cache-verification receipt attests that the root already hashed the
four original fit/evaluation cache payloads and sidecars and the two parent
prediction payloads before packaging. Kaggle does not receive or rehash those
large sequence caches. The scorer checks this receipt against each candidate's
fit/evaluation cache and parent ancestry fields, and compares all producer
sidecar hashes with the earlier frozen parent-cache milestone. Parent model
weights are not reloaded on Kaggle; the original parent provenance JSON,
checkpoint/provenance hashes, normalization identities and fit/evaluation
membership are matched to the recipe and frozen sidecars. This is a locally
attested cache identity, not an independent remote rehash of the omitted arrays.

Before it opens the target NPZ, the adapter validates all six candidate and two
parent prediction sidecars, exact source-qualified row-reference order,
membership and ancestry, NPY header/file-length coverage, SHA256, and finite
prediction values. It also checks that the transported target file has the
recipe-pinned content SHA and byte length; transport mtime is recorded but is
not represented as the original source mtime. Only then are target labels and
masks loaded and aligned by row group.

The report contains eight fixed endpoints (A/G/F and parent on both folds) and
five predeclared contrasts per fold: F−A, F−G, A−parent, G−parent, and
F−parent. Each contrast reports both targets and all four fixed row-position
bins, plus an overall official-WP dependency-component bootstrap interval.
Fold deltas are shown individually and as an equal-weight point mean; there is
no pooled significance interval. The intervals condition on fitted endpoints,
do not include training variability, and the folds are exposed development
data. These readout results do not establish a causal clock interpretation or
native serving promotion.

The adapter was prepared source-only. It does not inspect prediction values,
open labels, run a quality grade, or create a result until root admission.

## Synthetic validation

Run the bounded no-label checks with:

```sh
python3 grader/test_age_readout_grade_synthetic.py \
  --root /path/to/wnn_connectome_starterpack \
  --tmp-root /path/on/physical-disk/age-readout-test-scratch
```

The fixture calls the actor's real prediction-sidecar and provenance builders,
then validates a complete synthetic candidate through the grader. It also
checks source-qualified split overlap, missing candidate coverage, changed
prediction/source bytes, Fortran-order NPY rejection, symlink rejection,
transported target content pins with a changed mtime, main-path failure before
the target loader, and equivalence of the grader's official WP wrapper to
`global_wp`. Its labels and artifacts are generated in a temporary directory
on the approved physical scratch disk (the synthetic metric labels stay in
memory); the real target archive and candidate predictions are not opened. All
runtime candidate flags must declare strict FP32, TF32 off, autocast off, and
deterministic algorithms enabled. CPU thread counts and, for CUDA, the 4 GiB
allocator cap and measured peak counters must fit the declared resource limits.
The completed run and retry accounting are recorded in
[`SYNTHETIC_TEST_RECEIPT_R1.json`](SYNTHETIC_TEST_RECEIPT_R1.json).
The frozen recipe, six attested cache sidecars, and both parent prediction and
provenance chains were also checked without opening targets in
[`ACTUAL_PARENT_METADATA_CHECK_R1.json`](ACTUAL_PARENT_METADATA_CHECK_R1.json).
