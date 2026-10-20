# Teacher cache source audit R1

Status: source-only audit completed before the authorized narrow producer/profile extension. No SSH, checkpoint load, feature/target array read, GPU operation, prediction export, cache write, or commit was performed for this audit. Root remains owner of all execution and S5 acceptance.

## Bound source and teacher identities

The audited source checkout was clean at `af21d6d8409c46b36830f0bdddd50e67ee17025f` (`/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019`). Its 164-file teacher manifest is `0c83367d4ee3afd7e9f382111060d8c125b52420ce27cd5cc309d338287af7fd`. The audited cache builder is tracked by that commit but is not listed in the runtime manifest, so it is separately pinned below. The target binding is `REBUILD_RECIPE_BINDING_R2.json` SHA-256 `4a107af9189935f8f9732b0b2e6822e7883fbde2c44e00f5dda5eb73ae3abe39`; the student binding is `STUDENT_RECIPE_BINDING.json` SHA-256 `81662d8391fdc327bfe52bb5684533056af71dd467342b1f1c2375d854335171`.

The current source-qualified split is `/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json`, SHA-256 `c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`. The accepted S3 and S4 records each report 10,535 fit-train groups, 1,405 fit-valid groups, and 468 held-out evaluation groups. `load_campaign_split` binds source/index hashes and graph components; the T3 cache must contain only its exact fit memberships. S5's terminal acceptance and actual selected checkpoint digest were not read and remain pending root verification; no value is inferred here.

| Run | Root acceptance SHA-256 | `model.pt` SHA-256 | selected `model_ep2.pt` SHA-256 | tensor fingerprint |
|---|---|---|---|---|
| `fg19_t3_h384x2_e3_s3_v1` | `7c5ce63b087a15c309eeada24f0dd8df983d28f9931e86862347ca87b04282cc` | `67bbfb80bf915938efb9d961389d5f324ffd4e251862cc6d634d994596afd1dd` | `58e62541b9305169b94301d2e4f6643741d6a25f0712bc7b7cbf020de6b76241` | `6a868dfe732313f57aab75de33b83faad9336ec48a22395ec2a6aefc5a767902` |
| `fg19_t3_h384x2_e3_s4_v1` | `37e484f2ca507c11f45cb67299e6ba093046caffa3380a37dc0c84c53110a9e3` | `e20ce56668c9633141e461373cf6ea1f05d4ea2eaf50b6d9b3dbd48833221c5a` | `a59a747176263fc0ecdfcc584f78cf77b13465ec4645ca4bb5d9992380865a36` | `1deed8c26d044b87cb2af28ae724c0bf2e10994ece39a05752ca1395e6e28aab` |
| S5 | pending | pending | pending | pending |

The accepted S3/S4 identities above are source evidence only; the root-controlled cache admission must re-read the three accepted records, verify their actual digests and exact stop/population state, then provide the actual S5 pins. `build_distill_cache.py` currently checks checkpoint SHA against run provenance but does not bind a checkpoint to a root acceptance record.

## Canonical producer findings

The existing `scripts/build_distill_cache.py` (file SHA-256 `2db2d905bd6970b271d26fb4291b571c288789b0821a57af76b7ecc5738c1cd8`) is not a safe direct producer for this target. Its current path requires at least four sources spanning `FAMILY_ORDER` and a coefficient sum of `0.8125`; `combine_teacher_chunk` independently requires all four families. For three original `strong` sources weighted `0.1` each, its homogeneous result would be `0.3 × mean`, not the required `0.8125 × mean`. Changing the three coefficients to force a sum of `0.8125` would falsify their recipe records and still fail the four-family guard. Its emitted generic metadata also lacks the T3 profile fields required by the existing validator.

The existing consumer validator already recognizes `cleanfold_homogeneous_strong_t3_v1` and requires exactly three distinct strong teachers at epoch 2 with original coefficients `[0.1, 0.1, 0.1]`, source coefficient subtotal `0.3`, a separate homogeneous multiplier `0.8125`, formula `float32(0.8125 * float32_mean(three_epoch2_predictions))`, no construction-time clipping, and a homogeneous two-column target. Keep its existing consumer recipe record unchanged (`hidden=256`, `layers=2`, `epochs=6`, fixed epoch 5); the current H256 parent’s four-epoch recipe remains separately bound. The consumer clips cached teacher values to `[-2,2]` in the loss, after the cache is consumed at scale `1.0`; `prepare_batch` applies no extra multiplier at scale one. Therefore the required sequence is raw FP32 teacher predictions → FP32 arithmetic mean in explicit S3/S4/S5 order → one `0.8125` multiplication into cache → consumer scale `1.0` → loss-time clip. The cache must not clip individual teachers, store FP16, average/rescale twice, or apply any per-teacher normalization.

Prediction input can already be produced fit-only without reading labels or masks: `predict_run.py` with `--campaign-split-manifest` uses `FeatureOnlySequences` for both `train` and `valid`, and selects `split.train_groups` / `split.valid_groups`. These are the desired memberships, including no held-out valid groups. The builder currently instead requires prediction sidecars in the full target-index order. The narrow producer extension should accept exactly the fit-only sidecar groups, reject duplicate, missing, and outer groups, and align values by sidecar row-group identity into canonical target-index fitting order. It should write no held-out rows. This avoids requiring full-valid exports and avoids inference on the 468 evaluation groups.

The generic builder memory path is already bounded: prediction `.npy` files are opened with `mmap_mode="r"`, finite-checked in 64-sequence slices, copied only for each configured sequence chunk, and cache outputs are FP32 memmaps. It does not load all teacher predictions into Python RAM. Its generic chunk combiner creates several unnecessary targets, so the narrow T3 path should form only the two-column homogeneous chunk. The existing generic consumer loader memory-maps that target and copies only selected sequence rows.

`predict_run.py` loads trusted campaign checkpoints with `torch.load(..., weights_only=False)`, which preserves the repository’s native `TorchVersion` artifact compatibility; retain this behavior. With `--inference-precision fp32`, it calls `configure_strict_fp32` (TF32 off), marks the model for FP32 inference, and the canonical `predict` path disables CUDA autocast. Its normal production command exports a full requested split and has no bounded-profile mode; running it merely to learn resource use would already write a persistent prediction array. The authorized extension will add an opt-in, profile-only path over the first 16 current F0 fit-train groups, using the same B16 × 20,000 rows × 211 features, chunk 4,000, strict FP32 inference, and zero loader workers. It must read only features, perform no parameter update, write no prediction array/checkpoint, and emit only scalar identities, finite counters, CUDA allocated/reserved counters and configured cap, elapsed time, and process RSS.

## Exact prospective outputs and capacity

With the accepted fit counts, each run's required predictions are:

| Output | Shape | FP32 payload |
|---|---:|---:|
| train | `(10535, 20000, 2)` | 1,685,600,000 bytes |
| fit-valid | `(1405, 20000, 2)` | 224,800,000 bytes |
| per run total | 11,940 sequences | 1,910,400,000 bytes |
| three runs total | 35,820 sequences | 5,731,200,000 bytes (5.336 GiB) |
| final train cache | `(10535, 20000, 2)` | 1,685,600,000 bytes |
| final fit-valid cache | `(1405, 20000, 2)` | 224,800,000 bytes |
| final cache total | 11,940 sequences | 1,910,400,000 bytes (1.779 GiB) |

Prospective prediction directory: `/home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1/`, containing one `pred_<RUN>_{train,valid}.npy` and JSON sidecar per run/split. Prospective cache directory: `/home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_homogeneous_cache_v1/`, containing only `main_homogeneous_{train,valid}.npy` and metadata. Both must be absent before their create-only writers run. The combined payload is about 7.115 GiB; preserving the current builder's default 25 GiB physical-disk free floor requires at least 32.112 GiB free before prediction creation, plus actual metadata/checkpoint/log overhead and safety margin. No disk measurement was made.

The permitted production exports, after root acceptance/stop and fresh source, membership, output-absence, disk, and GPU-headroom checks, are one serial pair per run:

```text
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
python scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
python scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
python scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
python scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --output-root /home/grinchenkoeg/computer-vision-homework-20261018/data/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1
```

The separate bounded profile command after the authorized extension is the S3 first-fit-batch CUDA profile; it reads exactly the first 16 fit-train features and emits no cache payload:

```text
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --profile-first-fit-batch
```

The final cache builder needs an explicit `cleanfold_homogeneous_strong_t3_v1` profile argument and three source-qualified root acceptance/checkpoint pins. No safe invocation of the frozen generic builder exists; do not run a guessed four-family/weight-coerced command. The old `build_final_distill_cache.py` (SHA-256 `42ecff59a3fd5c3f532fede9977fd008703d0d604dde7dea975c90488aa53cc5`) is H390-specific, pins another split/propensity lineage, and admits all-data/held-out inputs; it is not applicable.

## Frozen source pins inspected

These runtime files match the 164-file manifest unless marked `commit-pinned` (the generic builder is omitted from the manifest but tracked at the clean source commit):

| File | SHA-256 |
|---|---|
| `scripts/build_distill_cache.py` (commit-pinned) | `2db2d905bd6970b271d26fb4291b571c288789b0821a57af76b7ecc5738c1cd8` |
| `scripts/predict_run.py` | `cf49d4ebcc1ff0946fa65bf13fdb5a7d4f54207d8c9d24d5a6da85674a72d8a3` |
| `src/wnn/distillation.py` | `5d68b03204b9413d2412533a7aa75ae0759276dacbbed4035b846da1d38350e2` |
| `src/wnn/distillation_cache.py` | `008fd6615ac5b627273f419072933cb95dc85fcd6c30eaffd6f1fb2df53ab0eb` |
| `src/wnn/training/evaluation.py` | `015c61a9281e5e31d91179edb4a1020a6978784941715b6a9c0db21149bb8345` |
| `src/wnn/resources.py` | `95fdc839ff1561622ac06c465cfc9a01370130d99be1e832b3a33c79bcb37930` |
| `src/wnn/splits.py` | `24024fbb82d33afdc2c709912404f7c1c26504a0816a050428c9f66e04e951fe` |
| `src/wnn/training/batches.py` | `c0b252cbd48028f874763dbaaaaa60aef1a38a148241f11f6e4e8433e3b02b21` |
| `src/wnn/training/losses.py` | `5859a4d40f13e91f50d34dfaed42d12dcfa2e229a85721ffc68bb5d69e1cfa66` |
| `src/wnn/training/cache_identity.py` | `0919b062d56b19d59b974a595f7412401029ba6b01a57ae968405336f3404d1f` |
| `src/wnn/final_distillation_cache.py` | `d649fe87a7ee2da53d60fa0350aa8a95cdf1b5046fd128e35814e199a37d82af` |

Authorized next code scope is limited to the existing canonical combiner/builder plus a bounded profile-only mode in `predict_run.py`. Keep legacy four-family behavior unchanged. Root supplies all accepted S3/S4/S5 acceptance, `model.pt`, and selected `model_ep2.pt` pins; profile and production admissions remain root-controlled.

## Implementation follow-up R1

The paragraph above records the audit-time pending state. Root subsequently verified that S5 was accepted and stopped and supplied the frozen execution contract, SHA-256 `dbcbe194f3ceb16d499f1c68bc8d862a16c32af54a13f57db75284691d8aabb9`. This local preparation still performed no prediction export or cache build. The contract binds the accepted S3/S4/S5 `model.pt`, selected epoch-2 checkpoint, provenance, tensor fingerprint, split, source order, and exact target arithmetic. The builder checks the contract bytes and corresponding acceptance/checkpoint/provenance files before inspecting prediction artifacts.

The canonical builder now has an opt-in `cleanfold_homogeneous_strong_t3_v1` path. It accepts only ordered strong S3/S4/S5 epoch-2 sources at their original `0.1` metadata weights, validates exact fit-only train/valid sidecar membership, maps prediction row order to canonical target-index fit order, and writes only two-column FP32 homogeneous memmaps. It computes the fixed-order FP32 mean then multiplies by `0.8125` once; no teacher clipping, cache clipping, FP16, or consumer rescaling is introduced. Its metadata remains compatible with the existing T3 consumer validator and records the historical six-epoch/fixed-epoch-5 student contract. The opt-in path verifies the root-pinned files and provenance but does not deserialize the checkpoint or reload optimizer/ancestor state. Legacy four-family validation and combination remain on their prior path.

For the commands below, `PROJECT` is `/home/grinchenkoeg/computer-vision-homework-20261018` and `B` is its registered physical-disk data root, `/home/grinchenkoeg/computer-vision-homework-20261018/data`. The root stages the compact acceptance files and contract under `$B/research/full_gru_tweaks_r1/teacher_cache_v1`; their `records/...` paths are resolved under that directory. Prediction exports and the combined cache live directly under physical scratch at `$B/research_tmp/...`.

After the root's fresh acceptance/stop, source, fit-membership, GPU, physical-disk, memory-headroom, and create-only output checks, the approved producer CLI is one bounded profile and six serial exports. The profile writes only scalar evidence to `PROFILE.json` and does not persist predictions:

```sh
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 train --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --device cuda --inference-precision fp32 --loader-workers 0 --profile-only --profile-output "$B/research_tmp/fg19_teacher_cache_profile_v1/PROFILE.json"
```

The production export CLI is issued once for each `(RUN, SPLIT)` pair below. It uses the accepted model and native source checkpoint loader, exports only the fit train/valid groups, and creates a `.npy` plus sidecar per pair:

```sh
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 train --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
python scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 valid --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
python scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 train --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
python scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 valid --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
python scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 train --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
python scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 valid --epoch 2 --campaign-split-manifest "$PROJECT/splits/fold0.json" --inference-precision fp32 --loader-workers 0 --output-dir "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" --create-only
```

After exports are accepted, the concrete cache command is:

```sh
python scripts/build_distill_cache.py \
  --data-root "$PROJECT" \
  --teacher-cache-root "$B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1" \
  --checkpoint-root "$PROJECT/artifacts/runs" \
  --cache-profile cleanfold_homogeneous_strong_t3_v1 \
  --cleanfold-contract "$B/research/full_gru_tweaks_r1/teacher_cache_v1/preparation/TEACHER_CACHE_EXECUTION_CONTRACT_R1.json" \
  --cleanfold-contract-sha256 dbcbe194f3ceb16d499f1c68bc8d862a16c32af54a13f57db75284691d8aabb9 \
  --acceptance-root "$B/research/full_gru_tweaks_r1/teacher_cache_v1" \
  --split-manifest "$PROJECT/splits/fold0.json" \
  --output "$B/research_tmp/fg19_t3_fit_homogeneous_cache_v1" \
  --teacher 'strong=fg19_t3_h384x2_e3_s3_v1@ep2*0.1' \
  --teacher 'strong=fg19_t3_h384x2_e3_s4_v1@ep2*0.1' \
  --teacher 'strong=fg19_t3_h384x2_e3_s5_v1@ep2*0.1' \
  --homogeneous-family strong \
  --expected-coefficient-sum 0.8125 \
  --chunk-sequences 8 \
  --output-profile homogeneous-only \
  --output-budget-gb 30 \
  --min-free-gb 25
```

The builder's independent preflight refuses an existing cache path or partial directory and preserves the configured 25-GiB free-space floor. At the known fit counts, the three teacher exports plus cache contain 7,641,600,000 FP32 payload bytes; root must remeasure actual physical free space and safety margin after source staging and before production admission. The builder reads predictions by mmap, validates them in bounded chunks, combines eight sequences at a time, and writes only `main_homogeneous_{train,valid}.npy` plus metadata. The existing consumer validator accepts the resulting absolute cache path.

Focused local verification completed without GPU, real-data, or checkpoint reads: the original 13 focused semantic/legacy tests passed; the synthetic legacy-ancestor closure regression passed again against the final builder; and 15 edge-case cases passed for strict row-group identities, FP32 overflow, and dangling output symlinks. Ruff format/check and `git diff --check` passed. The final combined cold import/read-sentinel smoke passed for 100 runtime WNN modules, both CLIs, and the helper; receipt: `/tmp/teacher_cache_cold_import_receipt.json`. No actual prediction or cache outputs exist from this work. Changed cache files are pinned here for root review:

The final narrow hardening preserves the legacy path: T3 fit-only sidecars and the semantic alignment helper now accept only actual integer identities (not booleans, floats, or strings); the FP32 combiner rejects a nonfinite result under a bounded NumPy error state before returning a chunk to its writer; and the opt-in T3 output preflight uses `lexists` before resolving the path so a dangling output symlink is rejected.

| File | SHA-256 |
|---|---|
| `scripts/build_distill_cache.py` | `e0d494019357e790576629b4731198fcbe2b80572b32e38ac7afc065bd6328b7` |
| `src/wnn/cleanfold_t3_cache.py` | `5f89ac1023d1b35f4d1ba02a39727e30ec1ff1d09fac28730039716188d5c8d4` |
| `tests/test_cleanfold_t3_cache.py` | `2f0ca5858c605d79e7bc24c89a74e30005216270645bba57c2229918ea8e6fbc` |
