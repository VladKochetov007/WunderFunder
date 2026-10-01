# Critic review v1: 2026-10-01 code (independent implementation critic)

Scope: `bind_fold_teachers` + distill under clean folds, `PooledCorrMemory` / `--loss gcorr_pool`, `ft_gclp.sh FT_STAGES=head`,
`joint_head_ft.py`, `screen_score.py`, `jobs/d256_fold.sh`. I read only this repo's working tree (no teammate branches).
I ran every check on CPU under `WNN_CPU_PCT=7 WNN_MEM_MAX=4G CUDA_VISIBLE_DEVICES='' scripts/limited`:

- `check_units.py` -> `check_units.out` (PooledCorrMemory gradient/numerics/fp16, aux-memory sharing, head-only freeze)
- `check_teachers.py --hash` -> `check_teachers.out` (teacher sidecars, row alignment, fold provenance, d256 args vs parent)
- `check_screen.py` -> `check_screen.out` (screen scorer vs frozen report, fold guard, duplicate folds, component ids)

## Verdicts

| # | item | verdict |
|---|---|---|
| 1 | `bind_fold_teachers` / distill under campaign split | **PASS**: no eval-label path. Minor hardening L1, L3 |
| 2 | `PooledCorrMemory` + `--loss gcorr_pool` | **PASS** for math and numerics. Latent bug M1 (not triggered by current jobs) |
| 3 | `ft_gclp.sh FT_STAGES=head` | **PASS** |
| 4 | `joint_head_ft.py` | **PASS** for fold safety, freezing and the joint-vs-solo match. Interpretation caveat M2 |
| 5 | `screen_score.py` | **PASS**: reproduces 0.6895232393809754 exactly. Missing guards L2 |
| 6 | `jobs/d256_fold.sh` | **PASS**: recipe equals the since parent except hidden 256 and distill |

## Findings (severity-ranked)

**M1 (medium, latent): one `PooledCorrMemory` instance is shared by the main head and every `--aux` shift head.**
`scripts/train_rnn.py:1083-1087` routes `objective()` to `pool_memory` for every call. `train_rnn.py:1300-1301` also calls
`objective()` for each `--aux` shifted-target head. Each call overwrites `self.memory`, so the next chunk's main-target pool
contains the aux heads' sums (other targets, other prediction columns) and gets decayed once per call instead of once per
chunk. `check_units.out: aux_calls_mutate_main_memory=true` shows it. Current exposure is zero: `ftpool_folds.sh` inherits
`aux=""` from the since parents. Any future `--aux 20,200 --loss gcorr_pool` run would be silently wrong.
Fix: `ap.error` when `--aux` is combined with `gcorr_pool`, or keep one memory per call site (e.g. a dict keyed by column slice).

**M2 (medium, interpretation): the solo control in `joint_head_ft.py` isolates the objective, but it is not the incumbent head FT.**
Joint and solo match each other on groups (`sorted(valid_groups)`), loader order (seeded generator, same seed in
`joint_heads.sh`), steps, OneCycle schedule, AdamW, clip and AMP. Both differ from the production head stage (`ft_gclp` stage 1):
lr 2e-4 vs 5e-4, no `aux_struct`/`aux_next` terms (the production head stage keeps both at 0.3 and they shape the shared
`head.0`), non-fused AdamW, and a different loader RNG. So joint minus solo is a clean effect estimate, but "joint beats
reference" requires the untouched reference arm in the same screen. The joint co-adaptation also holds only for the exact
equal-weight mixture of those k members. Adding a member or reweighting voids it.
Fix: screen three arms (reference, solo, joint) and promote only on joint > reference with joint > solo.

**L1 (low): `bind_fold_teachers` does not check row order or checkpoint identity.** `train_rnn.py:65-85` checks the run
name, split, manifest hash and prediction bytes. It does not check `side["row_groups"] == cache/<split>_targets.npz["row_group"]`,
`side["epoch"] == teacher checkpoint_epoch`, or `side["checkpoint_sha256"]` against the teacher provenance. All three hold
today for every exported teacher (`check_teachers.out`: s3_f0/s4_f0 train+valid and s3_f1 valid are all true, and prediction
hashes match). Empirical alignment on 12 fold-0 fit sequences: aligned wp 0.435 vs off-by-one 0.036. Parquet targets equal
cache rows. Fix: add the three comparisons; the frozen `campaign_score.validate_member` already does this.

**L2 (low): `screen_score.py` is missing guards the frozen scorer has.**
(a) Duplicate folds are accepted: `{"folds":[0,0]}` scores 936 sequences, each fold-0 row counted twice
(`check_screen.out: duplicate_fold_refused=false`). The frozen scorer raises "evaluation rows duplicated across folds".
(b) No `side["row_groups"] == targets["row_group"]` check. (c) The targets cache is not hash-verified. (d) No check that
`selected_rows > 0`. What works: the fold guard (`wrong_fold_member_refused`), the mixture (float64 accumulate then float32,
same as frozen), and the pooled bootstrap. Component ids are global (`load_campaign_split` recomputes them from the
single `cache/chains.npz` graph), and evaluation components are disjoint across folds (checked), so cross-fold pooling
is valid. Fix: add (a) and (b), a few lines each.

**L3 (low): teacher files are read after the hash check, and workers reopen them each epoch.** `bind_fold_teachers` hashes each
teacher once at startup. `Sequences` then mmaps `pred_<teacher>_<split>.npy` inside DataLoader workers, which are
non-persistent and reopen the file every epoch. `predict_run.py` replaces files by rename. If the same teacher name is
re-exported during a student fit, later epochs silently train on different targets while provenance records the old hash.
The same applies to `screen_score.bound_member`. Fix: re-hash at the end of the fit, or open the memmaps in the main process
before workers fork. Don't re-export a teacher while a student that uses it is running.

**L4 (low): `--distill` silently ignores `--loss`.** `train_rnn.py:1223-1227` always uses wMSE toward
`(1-a)y + a*teacher` (weights `|clip y| * main_valid`) whenever teacher columns are present. `d256_fold.sh` uses
`--loss wmse`, so there's no effect today. Fix: `ap.error` when `--distill` is set and `--loss != wmse`.

**L5 (low, cosmetic): the gcorr_pool loss value is inflated by 1/share.** At steady state the logged loss is about
(1-rho)/(1-decay): about 50x at 0.98 and about 200x at 0.995 (the `ftpool_folds.sh` setting). A chunk with zero weight logs
about 5e5 while its gradient is exactly 0 (`check_units.out: zero_weight_chunk`). Gradients are unaffected. Loss curves
can't be compared across `--loss` settings.

**L6 (low): `joint_head_ft.py` housekeeping.**
- `mkdir(exist_ok=False)` runs after training (`:130`), so a name collision throws away the GPU fit. Check at startup.
- Members are validated only by manifest hash (`:48`), not by fixed checkpoint and no-eval like `validate_campaign_checkpoint`.
- Outputs carry no job record or their own fit rows, so they can't enter the frozen `campaign_score` (screen-only).
- `model.train()` on the frozen trunk is harmless only while dropout = 0.

**Notes (no action):**
- The correlation losses don't clip predictions to [-2,2] (the metric does). This predates tonight and matches `gcorr`.
- Code drift: the d256 student runs a `train_rnn.py`/`nn.py` whose hashes differ from those recorded in the since-parent
  job. On the default path, the changes relative to the 09-30 base snapshot are the `structural_loss` refactor (identical
  to the old inline CE when `encoded_spread is None`, checked by reading) and instrumentation only.
- Outside scope: `jobs/final_d256.sh` reuses `cache/pred_ab67_final_since_s{3,4}_*.npy` from 09-27 without a sidecar.
  Its teachers' `model.pt` comes from legacy best-on-holdout selection (no `checkpoint_policy` in their args). The final
  student's holdout number is therefore not clean. That only matters if anyone quotes it.

## Item detail

**1. Distill fold safety: PASS.**
- The student reads teacher rows only through `Sequences(train_split, train_idx, teachers)` and
  `Sequences("valid", extra_valid, teachers)` (`train_rnn.py:959-965`).
- `train_idx` is within `campaign.train_groups`, `extra_valid` equals `campaign.valid_groups`, and `assert_fit_groups`
  rejects component overlap (`:880`). The eval loader is built without teachers and not at all under
  `--no-eval-during-fit`.
- Teacher predictions on evaluation rows exist in the file but are never indexed. Teacher predictions on fit rows are
  in-sample, which is acceptable.
- Row order: `predict_run.py` iterates `cache/<split>_targets.npz["row_group"]` without shuffling and writes position i.
  `Sequences` maps group to position through the same npz, so the two agree (verified above).
- `used` splits (`:901`) cover exactly the splits read. FT stages do not inherit `distill` (`inherit_init_arguments`
  field list).

**2. PooledCorrMemory: PASS.**
- The gradient equals the whole-pool Pearson gradient restricted to current rows, divided by share = n_cur/n_pool
  (max relative error 3e-16).
- Memory with decay d is equivalent to row weights d^age.
- decay=0 reproduces `weighted_pooled_corr_loss`.
- Cancellation from float64 raw sums: 1e-6 relative error at mean/sd = 5e4, negligible at realistic mean/sd <= 10.
- fp16 predictions produce finite fp16 gradients under a 65536 loss scale. The loss runs outside autocast and returns
  float32, so it works with GradScaler.
- Memory persists across chunks, batches and epochs through the closure (created once at `:1083`), as intended. It is
  updated even on AMP-skipped steps, which is negligible.

**3. FT_STAGES=head: PASS.** Stage 1 is unchanged. With `FT_STAGES=head` the script exports `${out}h@ep0` and exits.
`--freeze` is parsed out and applied to stage 1 only, and stage 2 always uses `--freeze ""`. An empty `runner` array is safe
under `set -u`.

**4. joint_head_ft: PASS.**
- Only `head.*` receives gradients; the GRU and `inp` get none (`check_units.out`). This is the same trainable set as
  ft_gclp's `inp,gru` prefix freeze for linear-encoder RNNModel.
- State is carried per member across chunks and detached each chunk. It fits only `split.valid_groups`
  (`assert_fit_groups`).
- The checkpoint keeps `args`/`n_in`/`readout`/`head.2.weight`, so it loads in `predict_run.py`. The sidecar inherits the
  parent manifest hash.

**5. screen_score: PASS.** Pooled wp matches exactly (0.6895232393809754), as do all four per-fold dicts and the by-class
dicts. The 8 member prediction hashes equal the frozen report's bindings. The run takes about 40 s on about 1 core.

**6. d256_fold.sh: PASS.**
- Arg diff against `c20260929_since_s3_f0` shows `hidden` 128 -> 256 and `distill` only. The `sync_free`, `fused_adam`,
  `pin_memory` and `prop_cache` differences come from `--fast` expanding after parse. New arguments sit at no-op defaults
  (`pool_decay`, `spread_aux_mode=original`, `decay_init`, `memory_arm`, `quote_cache_root`).
- The FT uses seed 3, which matches the incumbent FT for both s3 and s4.
- Operational: teacher exports exist only for fold 0 (complete) and partly for fold 1 (s3_f1 train is still `.partial`,
  s4_f1 has none). `d256_fold.sh 1|2|3` fails closed at `bind_fold_teachers` until `teacher_preds` covers that fold.
  The current job script only exports folds 0 and 1.
