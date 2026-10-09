# H230 audit of the reported 0.6674 heterogeneous-student recipe

Status: source audit is complete enough for root review. No full Maxim-source fit was launched. The own-code all-data parent profile is staged separately and remains held until the independent H230 family-off grade finishes.

## Pinned endpoint

The reported endpoint is pinned to `origin/research/creative-iteration-20260930` at commit `833aaf541155c591e930bbc6c640b8817e01a536`. Its `wave20/public-result.json` identifies submission `26n9pg4y`, ZIP SHA256 `86020dcf384430ae5fd3a848c8a8b4533cd81031caad3b0feec7647c4a5491f9`, score `0.6674`, and explicitly labels the score as user-reported. The package is a 0.5/0.5 pair of fixed-epoch-1 h256 GRU students; the package manifest binds both exported NPZ members and their hashes.

The source that produced this endpoint is `wave6/scripts/train_rnn.py` (SHA256 `26b2b96fe2d132d068f7afcc906d11aa28fbb1ab955d006125841cbd6e1cc9aa`) plus `wave6/scripts/compress_model.py` (SHA256 `8a618b49e23bd1302f5cbe1c79ddbb04ea921f44908eb3583a2d414fa038463e`). The checked `origin/quant-probes` trainer is a different path and source (`scripts/train_rnn.py`, SHA256 `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f`); it is not the 0.6674 endpoint's trainer.

The final student command is recorded in `wave18/final_student_lane.sh` (SHA256 `718a41d130f71168cfa00144fa70785c2bd4fd6cf6cf1ca99c84c0d0f95c2802`). For each seed 5/6 it initializes `fbc1k_s{seed}@ep3`, trains h256, two-layer GRU students with batch 16 and chunk 2000 for two epochs, and mixes the clipped label and clipped teacher cache at 0.3/0.7. It uses AdamW, LR `1e-4`, WD `.01`, FP16, OneCycle scheduling, `--fast`, propensity power 2 plus `.03`, true-mask epsilon `.03`, structural and next-event auxiliary losses `.3` each, and fixed zero-based epoch 1 for final export. The training command includes every supplied validation row with `--include-heldout-valid`; its `holdout`-named training metrics are therefore in-sample and are not held-out evidence.

The main cache is the unnormalized source sum

`0.30 × strong + 0.10 × full_s3 + 0.10 × full_s4 + 0.125 × raw_s3 + 0.125 × raw_s4 + 0.0625 × orgbase`,

whose coefficient mass is `0.8125`. `prepare_full_teachers.py` (SHA256 `6af31d8a5b7aca818bbf45c4328fd02afe8ad2f1d99c0da075fa3cacfdecee4b`) constructs this sum for train and valid. Family targets are: the strong teacher; the mean of full s3/s4; the mean of raw s3/s4; and organizer baseline. The trainer clips the main labels and main teacher to ±2 before mixing, but the family targets remain raw. Its family loss is hard-coded to `.1`; it averages four family losses using the same `|clip(y,2)| × main_valid` row weights. The auxiliary family outputs are discarded at export.

## Parent and teacher lineage limits

The shell and manifest evidence identifies the parents and teacher predictions, but the tracked branch does not provide checkpoint hashes or complete training receipts for the source parent and teacher weights. `fbc1k_s5/s6@ep3` are described in the branch research notes as full-data d256 “Vlad recipe” Muon runs (boot horizon 8, chunk 1000), with later full fine-tuning. The full and raw teacher lanes predict `ffull_s3/s4@ep1` and `frawW_s3ft/s4ft@ep1`. Their fitted checkpoint identities, full argv, training logs, and per-checkpoint hashes are absent from the endpoint manifest. The `orgbase` teacher is referenced as `artifacts/orgbase.npz`, but the endpoint snapshot does not bind its artifact hash or origin receipt. Treat each missing ancestry item as unresolved; do not substitute teammate weights or the unverified `orgbase` artifact in an own run.

The package does bind the final distilled weights: `weights_s5.npz` SHA256 `45ac50f7324b8e26f37eee97b975e2643134e54ab2399077de8778f931abc2bf` and `weights_s6.npz` SHA256 `f6f9702fc1645d7f60a456d3fb91b80149a4ed785489a8b3b333d7a6d244c2dd`. These prove the delivered pair contents, not the identities of its parents or cached teachers.

## Differences from the independent H230 implementation

| Dimension | Reported endpoint | H230 seed-3 candidate/control |
|---|---|---|
| Code | Branch wave6 trainer and its `CompressModel` helper | Independently written E529 `h230_source_v2`; source manifest SHA256 `697d78e5a5ac5461e876486caa8b278298283a213ed03432c43a856fd7f1bd58` binds 54 files |
| Fit membership | All usable train and all valid rows, including reserved valid rows; no local holdout claim | Fold-0-clean fit; the independent membership audit confirms candidate fit 0 of the 468 scored groups |
| Initializer | `fbc1k_s5/s6@ep3`; exact weight hashes unavailable in the branch snapshot | Our own `c20261003_combo_s3/s4_f0@ep3`; seed-3 checkpoint SHA256 `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e` |
| Main teacher sources | Strong `.30`; full `.20`; raw `.25`; organizer baseline `.0625`; total `.8125` | Strong `.30` from our combo s3/s4; full `.20`; raw `.25`; own wide same-fold baseline `.0625`; total `.8125` |
| Family composition | One strong model; full and raw paired means; organizer baseline | Own strong pair mean; own full/raw paired means; own wide-baseline pair mean |
| Main target / loss | `.3 × clip(y,2) + .7 × clip(source_sum,2)`; propensity weights; WMSE | Same main target mass and loss form, independently implemented and frozen in E529 |
| Family-head parameterization | Separate `Linear(hidden,8)` attached after the shared pre-output feature block | Eight rows appended to the final output `Linear`; same feature input and family-loss normalization, with different initialization/RNG behavior |
| Fit/checkpoint | Two epochs, one-cycle, fixed epoch 1; code evaluates the all-data fit on in-sample valid data | Two epochs, fixed epoch 1; fit-time evaluation disabled; fold-0 groups excluded from fitting |

The seed-3 H230 fit and matched family-off retry are complete; the independent grader owns the frozen comparison and was asked to finish that grade before any new profile consumes headroom. The user-reported H230 public single is `0.6649`; that number alone cannot isolate family-loss value or explain the difference to the all-data pair.

## Smallest useful causal sequence

1. Finish the frozen, independent H230 candidate-versus-family-off grade. Interpret it only on its exposed fold-0 development set and alongside its membership review.
2. Admit the one-batch own-code all-data parent profile recorded in the companion JSON. It exercises the exact H230 source, own checkpoint initializer, full `train+valid` membership option, and B16/chunk-2000 update shape. It writes no score or checkpoint. The grade currently owns the local resource window, so the profile has not launched.
3. After root reviews the recipe packet and fresh resource/source checks, fit independently owned parent/teacher sources over all supplied train+valid rows. Generate all-row teacher caches only from those own checkpoints and preserve exact source coefficients. A full-data fit is a final-training run, never a clean local evaluation.
4. Compare fixed-epoch-1 all-data student pair against a matched family-head-off pair under the same source mixture, membership, initializer ancestry and seed pair. Reuse the completed H230 grade as mechanism evidence, not as a public all-data effect.
5. If the paired endpoint warrants further budget, isolate teacher-sum normalization (`.8125` source mass versus normalized sum), family-target clipping (raw versus ±2), source-family composition, student budget (2 versus 4 epochs), and architecture scale/depth as separate registered arms. Predeclare checkpoint selection and retain all members; do not choose a seed or recipe from an in-sample final-fit score.

The branch result has a useful transferred hypothesis—multi-family recurrent teachers can guide a same-size student through distinct family targets. The current audit does not verify the branch's missing parent/teacher weights, reproduce its public score, or establish that any single recipe difference caused the observed score gap.
