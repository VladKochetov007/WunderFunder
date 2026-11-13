# Since-scale profile admission review

**Decision: profile gate passes for the four labels-only since-scale followups.** This is an engineering/admission conclusion only; the 30-update profiles do not establish a quality gain, and validation has already been exposed by earlier work.

## Recipe match

The frozen panel is `since-scale-followup-r1/PANEL_MANIFEST_R1.json` (SHA-256 `37f959c0a3d8bcdfc8b7db31fc47e19135ff11ac5d4d13e5c5079078b4a4ed5e`). Its four configs are SHA-bound in that manifest. The actual canonical preflight receipt, collected as `cluster-all-experiments-20261112/l4-rpc/results/167-age-preflight-profiles-r1.json`, reports `changed_fields: ["since_weight"]` for each same-seed comparison. The active baseline config hashes are `996204e336c77dc2b2c27bbd2094c14a7ad05084a89ad81a116c197c40320226` (seed 203401) and `7ed0d8d6fc9dba5095fd9361bfbb84b981b0efd4d53b420266bccca2dc97bfe3` (seed 203402).

The only scientific parameter change is `since_weight`: 32→8 or 32→128, at seeds 203401 and 203402. `since_floor=0.25`, K16 stem, 2×256 GRU, FP32, AdamW, six epochs, batch 32, 1,000-row chunks/horizon, normalization over all 10,325 fitting members, and the optimizer/schedule controls remain fixed. In the pinned source, the scale applies to the row-weight rule `clamp(1 / (1 + since / since_weight), min=since_floor)`.

The active controls and new profiles report source identity `29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511` and official split `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. Each profile receipt contains 10,325 fitting members, all tagged `train`, and the same 1,873 `valid` evaluation members. The common data fingerprint is `4987ed5f263cf3c655714a7af3e792605716f6cb256c1c96474eb46ae9869323`. The separate chronological protocol is preserved; `seq_ix` is not used to order data.

## Actual profile evidence

The four embedded receipts in `l4-168-native-wave-status-r1.json` each report `profile_complete`, 30 updates, zero errors, passed reload, one L4/world size 1, and equality of model, optimizer, scheduler, and RNG across reload. Peak reserved memory is 1,883,242,496 bytes per profile. Elapsed seconds were 3.585 (age8/seed203401), 3.617 (age128/seed203401), 3.613 (age8/seed203402), and 6.847 (age128/seed203402). The last profile was slower under a foreign GPU load noted by the operator; memory and reload gates still passed.

The R2 fit-admission launch record binds these exact profile receipt hashes: age8/203401 `623abd37cb8558aa28f1807b0ca791317fc83cdfeaec7c60d86fdb3fdd6c1eb7`; age128/203401 `33d61499c4f21a1e797dbb6381faeadb860663203198c25a16a3c1f97148f0b8`; age8/203402 `0d1248b131697a9aa70877164976bbe46c2dec3abdf83299f096b7a6ce4b2970`; age128/203402 `0ba843b6094a00473665e94e93fa2510789b5e3717e05088d2ac98d0d0bc6df4`.

The profile identity has no teacher-cache provenance and `kd_weight=0`; no teacher cache is passed by this labels-only controller. The runtime logs may contain a nonzero field named `kd_loss`/`distillation`: it is a diagnostic, not an applied KD term here. The configured KD coefficient is zero, and the objective receipt uses the standalone-label objective. Do not read that diagnostic as evidence of teacher use.

## Launch and guard review

- `l4_age_wave_r1.py` pins the source capsule, manifest hash and split hash; rechecks the config hashes; checks the shared `CLOCK_R1.json` fit deadline; and bounds each process timeout by the time remaining. The fit admission record `170-age-admit-fullfit-r2.json` was created at epoch `1794508770.13`, before the shared fit deadline `1794528790.58`.
- The controller CLI accepts only `profile-age` and `fit-age`. It selects the four entries whose `teacher_cache_required` is false, requires exactly four, and does not append `--teacher-cache`. The per-cell check binds the config's seed and KD weight to the manifest. KD-required cells cannot pass the four-cell gate in this labels-only invocation.
- Sorted cell dispatch sets `CUDA_VISIBLE_DEVICES` to 4, 5, 6, and 7. Profiles report one visible L4 each; the subsequent native health snapshot records the four new fits on GPUs 4–7 while the four existing controls remain active.
- Existing output directories and phase control directories are rejected (`FileExistsError` / `exist_ok=False`); the fit command has no resume flag and requires an explicitly approved profile receipt. The four fit outputs are fresh paths, so this is not a restart.
- The first admission attempt (`native_age_admit_fit_r1.py`) used the wrong membership key `fitting_members` and failed before writing approved profile copies or spawning the fit controller. The preserved R2 successor correctly checks `fit_members`, membership counts/tags, source/split, reload equality and memory before writing the four approvals. Its collected launch record binds the four profile receipt hashes and the same fit deadline.

Reviewed source hashes: `since-scale-followup-r1/l4_age_wave_r1.py` `be39b3f9e7de6ebfa9819833def1663d49d1a0521e74f5f7aed3a5e921ccf1ad`; `native_age_preflight_start_r1.py` `f478ccc0ce15045dde0e2261059275602e6d78a069309550c9b6132fd7814baa`; `native_age_admit_fit_r2.py` `a14784ebd51168700a3a9fe5257866d95b7186d347b6c31f6b90e967f49d64b9`. R1 is preserved separately with SHA-256 `393e409d6d65fb3f86eba2cf16e3365968f4ca6c70b3e71f493bbc3b16cb3fbd`.

## Limits

The controller reads the common `CLOCK_R1.json` at runtime rather than pinning its hash inside the panel manifest. The actual profile and fit-admission records use the same deadline, and no clock change is evidenced here; future reuse should retain the shared-clock immutability check. Profiles are resource and configuration evidence only. No full-fit endpoint, independent validation result, significance test, or submission-quality claim is admitted by this review.
