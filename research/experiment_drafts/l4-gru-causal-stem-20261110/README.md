# L4 GRU causal input stem screen — 10 November 2026

**Status:** authorized six-fit design with committed implementation `336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2`. Capsule source SHA is `b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188`, manifest SHA `482d13529e8c65386e6f74e7e81f607eeb3b0ed0795beff44902bf703e7d342e`. The focused/existing CPU model suite passed18 tests; Ruff and the training structure checker passed. Independent review, six actual L4 profiles and fit approval remain pending. No scientific fit or quality gain is claimed yet. ROOT's authority is recorded in `C:/Users/admin/Documents/Codex/artifacts/l4-causal-stem-20261110/AUTHORITY_R1.json`.

The screen adapts the causal input-stem idea to the existing raw112 GRU256x2 recipe. It compares an identity/no-stem setting (`input_stem_kernel=0`) with per-channel residual causal FIR kernels 1 and 16, paired across seeds 203301 and 203302. K16 sees the current standardized row plus at most its preceding 15 rows. Those rows must carry across chunk boundaries and reset at each whole-sequence boundary. No future row is available.

This is a conceptual descendant of PR74, the B300 Transformer input-stem draft, and the older ledger E-168 Conv16-before-GRU experiment cited by PR74. It is not a reproduction of either recipe: E-168 reset history at chunk boundaries, while this plan tests carried history on a GRU under the six-epoch L4 recipe. The PR74 metadata records its own profiles/fits as unrun at publication. It does not supply evidence for the present L4 source or result.

## Frozen six-cell panel

| Arm | Seed 203301 | Seed 203302 | Stem parameter delta per model |
|---|---|---|---:|
| Plain (`K0`) | `plain_s203301.json` | `plain_s203302.json` | 0 |
| K1 | `k1_s203301.json` | `k1_s203302.json` | 112 |
| K16 | `k16_s203301.json` | `k16_s203302.json` | 1,792 |

K1 and K16 are not parameter-count matched; K16 has 1,680 more stem weights than K1. This screen estimates the effect of the complete stem recipe, including parameterization and optimizer interaction. It does not identify a pure lag-length effect.

Each JSON is copied from the frozen PR77 shared-control config with only `seed` and `model.input_stem_kernel` changed. All other fields remain fixed: six epochs, 38,760 updates, GRU256x2, raw112 input, FP32, one L4 rank, batch32, chunk1000, accumulation1, AdamW3e-4/weight-decay0.01, clip1, warmup5%, warmup-cosine-v2/min-LR-ratio0.1, all-fit train normalization over10,325 sequences, all-after-warmup mask, standalone two-target WMSE, all-epoch checkpoints and configured final epoch6. No teacher, KD, hint, reference, cached output, or valid ancestry is permitted.

Raw config SHA-256 values:

| Cell | SHA-256 |
|---|---|
| plain_s203301 | `0353a460479ac044a61265dfa90704a774449f7ef62af694212ff0f49d03d467` |
| plain_s203302 | `9d185a0b820616005bd24485cf789f659a075e7584a8257c1bb6d2f86baca751` |
| k1_s203301 | `8dc5aee84b9e68ceba1b2253fe1a28423f051c14659320e5110ffe847c8fcdf2` |
| k1_s203302 | `eb8ad24fd4a445573360067e5a65aa6325e42119e41f4f2afbe2266d1a899a1b` |
| k16_s203301 | `55a6f952f13d923817eda6c3dbf7a002934ff8e481f6367c6c1c82d8e754a534` |
| k16_s203302 | `a6672a34223f88d5bebfe9712ed3fb09192df6176239de5aeaf3dce2bd527bc0` |

These are raw file hashes, not loader-normalized hashes. ROOT must record loader-normalized config identities after pinning the actual source. Configs are not source-admitted yet.

## Gates, timing and evidence

ROOT owns the existing native RPC, source capsule, profile/fit approvals, GPU assignment, process signals and collection. The six new fits use GPUs0–5 only, one fit per assigned GPU. Existing SSL children on GPUs6/7 remain untouched. The authorized wall clock is four hours from the first actual new-device profile: at most three hours for fitting and one hour for full-valid evaluation/checkpoints/collection. At most six source-bound 30-update L4 profiles are authorized. Source, split, config bytes and device must be frozen before profiling; each fit waits for its own actual profile/reload evidence and ROOT approval. No profile timing or six-fit feasibility estimate is claimed here. The hard output cap is 2.5 GB with at least 4 GB free disk; aggregate owned RSS cap is 128 GiB. Local CPU preparation is capped at two cores. No added SSH, downloads, rentals, billing, retries, resumes or automatic follow-up fits.

Before any profile, the pinned source must prove: strict FP32 and exact 0/1/16 kernel dispatch; zero-initialized stem residual preserving the raw input; bitwise-equal common GRU/backbone initialization across the three arms for each paired seed; independent fresh seed initializations; K16 prefix causality; carried last-15-row equivalence across arbitrary chunk boundaries; reset at every sequence boundary; and native RNG capture/restore consistency. These are required gates, not completed tests. A CPU stub or source inspection alone cannot stand in for actual L4 profile/reload receipts.

Primary endpoint is fixed epoch6 `global_wp_clip2_v1`, with full official validation on all1,873 sequences and3,528,273 scored rows. Report the paired K16−K1 score delta for both seeds and their descriptive mean; report K16−plain and K1−plain as secondary contrasts and retain both target components. No validation-based checkpoint selection or tuning. The official chronological train/valid split fits only on10,325 train sequences; the full valid panel is repeatedly exposed development evidence, not a fresh holdout. The source/data receipt must confirm the native project has the same official membership before admission.

ROOT should preserve every epoch checkpoint, fixed-E6 predictions and grades, exact sequence/row coverage, source/config/split/GPU receipts, update and optimizer diagnostics, per-epoch train/full-valid curves, wall time, peak memory/disk, and archive hashes. A source SHA/commit and exact native project paths are not in `AUTHORITY_R1.json`; do not fill them from an uncommitted or guessed source tree. Independent review is required before costly fits. No result, public gain, or upload is claimed.

## Separate PR76 CPU callback qualification lead

This is a fit-free and separate deployment check; it is not part of the six-fit comparison. The local cached PR76 head commit `f95f5b5671e7b6d55342192dccae78c2f37eb045` contains the exact receiver sources and manifests at `research/submission_candidates/tl3-lstm-complement-20261110/`. The file path is absent from the current sparse checkout but readable from that local Git object with `git show <commit>:<path>`. The PR76 files include `RUNBOOK.md`, `launch_server.sh`, `Dockerfile.receiver`, `receiver.sh`, `receiver_io.py`, `receiver_parity.py`, `validate_cpu.py`, `solution.py`, `lstm_cpu.py`, `recipe.json`, `INPUTS.json`, `BUILD_RECEIPT.json`, `PR53_MANIFEST.json`, and `PR53_FIXED_MIXTURES_REPORT.json`. The cached `pr76-files.json` records the source blob SHAs.

The PR76 build receipt identifies `kotiki-tl3-w18-lstm10-cpu-20261110.zip`, 14,011,657 bytes, SHA-256 `766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e`. That ZIP/handoff binary was not found in the local `artifacts/next70-20261110` tree and has not been fetched or rehashed here. The expected portable handoff name in the runbook is `cpu-receiver-handoff-20261110-r2.tar.gz`. The ZIP embeds the served callback weights; the input manifest traces the LSTM checkpoint to `full-primary-configured-final.zip` (archive SHA `c01fcefb1aa7a8b05a9a3b4f7206c61e677e9ba9c0394fdcfe396cafeb5b5a1b`) and `lstm.pt` (SHA `842c6d5602c32052045d7567fe09365aa358eab7cc5251d7cfb8744b25c61c90`). The TL3 comparison baseline is produced in the same callback pass; it is not a separate fitted endpoint.

`INPUTS.json` pins the official split SHA `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`, valid Parquet length5,615,509,128 bytes and footer SHA `4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b`; it has1,873 sequences/37,460,000 total rows/3,528,273 scored rows. `PR53_MANIFEST.json` content SHA is `ed15cde830f5cf543e0b84bc986c703351ac55f273c8769f6de1d5ea6d92c7a9`; PR76 inputs content SHA is `2b751fa835e0c895631a6f167eda6a10880c956bae17d79f581f65f15e5fe190`, and recipe SHA is `881c125a78ecf97e8b59986401309b0bd9b2f0b955ee1dcdcb7ae38e82d85ac2`.

The pinned runbook's sequence is: verify the staged handoff `sha256sum --check SHA256SUMS`; bind a single CPU and an existing exact Docker image; run `bash launch_server.sh profile` and inspect `outputs/profile/profile.json` (`profile_passed` must be true); then run `bash launch_server.sh full` and inspect `outputs/full/full.json`. The full stage pairs the candidate callback with the served TL3 base over all official valid rows. The runbook's Docker build downloads dependencies, so ROOT must use an already approved handoff/image or obtain separate authorization before that step. No CPU command, model execution, upload or network action was performed as part of this packet.

## Local progress publisher

[`reporting/publish_l4_stem.py`](reporting/publish_l4_stem.py) reads the six shared-dashboard mirrors, verifies each cell against ROOT's exact profile-admission SHA, and reports phase/update/loss/preclip-gradient/latest full-valid target WP every30 minutes and at each terminal receipt. It emits paired K16−K1, K16−K0, and K1−K0 target deltas only after all six fixed-E6 terminals and exact full-valid events pass. It reuses PR77's local evidence readers, idempotent marker state, and the allowlisted `ProgressGitHub` client, binding the client to ROOT's CLI-selected PR. The reporter uses the new root's `live-dashboard/bridge-state.json` for mirror age and the existing next70 `live-dashboard/native` mirror for cell files; if bridge time is in the future after a host clock adjustment, age is reported as unknown.

ROOT supplies the actual fit-admission path/SHA and PR number at launch. The command shape is:

```powershell
python research/experiment_drafts/l4-gru-causal-stem-20261110/reporting/publish_l4_stem.py `
  --root C:\Users\admin\Documents\Codex\artifacts\l4-causal-stem-20261110 `
  --monitor-root C:\Users\admin\Documents\Codex\artifacts\l4-next70-20261110 `
  --profile-admission <ROOT-local-admission.json> `
  --admission-sha256 <ROOT-admission-sha256> --pr <ROOT-selected-pr> `
  --maxseconds 14400 --state-file C:\Users\admin\Documents\Codex\artifacts\l4-causal-stem-20261110\reporting\publisher-state-pr<ROOT-selected-pr>.json
```

ROOT provides GitHub credentials only in the publisher child environment. No live publisher run or GitHub post was performed here; CLI help and Ruff passed. A local-mirror dry-run remains pending the exact ROOT fit admission and PR number.
