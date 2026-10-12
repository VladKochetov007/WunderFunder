# First-cohort clean F3 teacher delivery

Status: source audit and private asset staging complete; the Kaggle source bundle is not yet sealed or uploaded, and no F3 teacher fit has run. This packet resolves the first-cohort teacher prerequisite in [the approved strategy execution plan](../PLAN.md) using only the project's existing trainer, Kaggle bundle, and Kaggle worker tools.

## Exact endpoint owed

Produce independently initialized H384, two-layer GRU, Muon-all teachers for canonical fold 3, seeds 3, 4, and 5, trained for three epochs with fixed epoch-2 checkpoints. Each teacher must export FP32 predictions on its fit-train membership, fit-valid membership, and full-valid feature-only membership, with sidecars bound to the exact checkpoint and fold-3 split. F0 matching teachers and caches already exist; this note only covers the missing F3 members.

The local artifact audit found no exact F3 endpoint for any of the three seeds. The closest H384 F3 candidate uses Muon-rnn and is not a clean T3 match; the H256 candidates are not substitutes. The F3 teacher recipe is scratch initialized, matching the existing clean F0 lineage. The existing `c20261003_combo_s3_f3@ep3` checkpoint is an unrelated H256 comparator and is not a teacher parent.

## Source-qualified data and inputs

The canonical fold-3 manifest is `research/campaign_2026-09-29/artifacts/splits/fold3.json`, SHA-256 `78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343`. It assigns 10,539 train-fit and 1,368 valid-fit groups, excludes 505 evaluation-valid groups, and has 1,873 full-valid groups. The F3 propensity caches bind model `c20260929_prop_f3`, checkpoint SHA-256 `8898ef3d31533058c354a16d00632b7c7adca461eaa813f0bed1f59a1fef59b8`, and the 10,607/1,873 source rows. Full file identities and the staged upload manifest are in `INPUT_BINDING_R1.json` and the local private staging directory recorded there.

Only the fold-specific propensity checkpoint, propensity arrays/sidecars, and fold manifest are staged as the new private asset payload (about 1.0 GB). The shared chains, targets, portable manifest, and raw parquet retain their existing private asset / competition dataset identities; the final kernel bundle must map every external input explicitly and verify its exact mounted bytes before importing the trainer.

## Profile and resource review

The prior F0 H384/Muon-all T4 profile completed 5 warmup plus 25 measured batches in 397.86 seconds, with 4.80 GB observed worker RAM and 3.50 GB observed GPU memory. It is a shape reference, not a full-fit maximum or ETA. The F0 local fit times (4,310.6 / 4,396.6 / 3,485.9 seconds for seeds 3/4/5) are different-hardware measurements and are not used as a cloud runtime guarantee.

Exact live Kaggle quota at audit was 35,838 GPU-seconds remaining, 72,000 TPU-seconds remaining, with zero reserved; refresh was 2026-10-17. One F0 T4 measurement implies the three-fit plus nine-export request must be staged with measured profile guards and terminal artifact collection; no project sub-ceiling or paid capacity is authorized. Kaggle's current CLI metadata describes `NvidiaTeslaT4` as T4 x2 by default; placement and device count still must come from the runtime inventory. P100 is not selected because current Kaggle metadata does not list it as a current accelerator and recent CLI reports describe P100/default-image incompatibility. See [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) and [accelerator options](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

The old F0 R3 packet is preserved as provenance only: it requires two distinct T4s, enforces a 5 GB profile threshold, and inherits a 6,000-second kernel / 5,200-second worker timeout. Those guards are not carried over as the F3 authorization. A reviewed successor must profile on the actual allocated device(s), retain a documented physical-fit margin, and allow enough worker/kernel lifetime for the measured profile, fit, export, and finalization. Root review of the exact sealed source, entry point, slug, mounted-input hashes, and resource estimate remains the dispatch gate.

## Evidence limits and next phase

Fold 3 is repeatedly exposed development evidence, not a fresh holdout. The teacher outputs enable the strategy plan's amplitude, innovations, and detach comparisons; they do not by themselves establish a strategy gain. No Kaggle object has been created or uploaded from this packet, and the staged `artifacts/kaggle_assets/...` directory is private local preparation only. The next owner step is to seal and offline-validate the F3 source/spec bundle, have root review it, then profile, fit, collect terminal outputs, and bind the prediction sidecars to the resulting checkpoints.
