---
kind: review
status: complete
created_utc: 2026-11-14T12:37:10Z
updated_utc: 2026-11-14T12:43:50Z
hypothesis_ids: []
experiment_ids: ["quality_r2:q14_kd90_s4", "local_r1:logged Muon cells"]
evidence_paths:
  - research/campaign_2026-11-14/quality_r2/CONTRACT_R1.json
  - research/campaign_2026-11-14/quality_r2/run_fit.sh
  - research/campaign_2026-11-14/local_r1/run_cell.sh
  - research/campaign_2026-10-29/teacher_track_r1/strict_launch.py
  - /home/vlad/development/wunder/research_tmp/pr49_wt (commit 188c0788e571c6f981cdc6827fef0d50bd38cf14)
  - research/campaign_2026-11-14/quality_r2/PRECISION_AMENDMENT_R2.json
  - research/campaign_2026-11-14/quality_r2/precision_fix_r1/CUDA_CHECK_R1.json
  - research/campaign_2026-11-14/quality_r2/precision_fix_r1/check_precision.py
  - research/campaign_2026-11-14/quality_r2/run_strict_fit.sh
  - research/campaign_2026-11-14/quality_r2/q14s_kd90_profile_s3_r2.fit.log
  - /home/vlad/development/wunder/research_tmp/q14_strict_source (commit 48776fba8af26b16b7e6813def7ccf571f56c1ac)
---

# Precision audit: legacy Muon is not strict FP32

## Question or claim

Do the `local_r1` and current `quality_r2` legacy `--optim muon` fits honor the strict-FP32 policy through the optimizer, or does a runtime wrapper/config turn on `strict_fp32`?

**Finding for the original frozen runs: no.** The model forward path is FP32 with autocast disabled, but legacy Muon Newton–Schulz orthogonalization uses BF16 on CUDA. No launcher, recipe argument, or logged fit config supplies optimizer-group `strict_fp32` for those cells. The launchers' TF32/cuDNN controls do not change this optimizer option. A subsequent isolated source fix and its bounded CUDA checks are assessed below; they do not change the original runs' precision identity.

## Mechanism and method

Read-only source and recipe audit. The frozen trainer is `/home/vlad/development/wunder/research_tmp/pr49_wt` at commit `188c0788e571c6f981cdc6827fef0d50bd38cf14`. Inspected optimizer construction, precision contexts, both local launch paths, the quality launcher, and the separate TL3 Muon path. A Python standard-library JSON read of fit-log argument records found 18 `local_r1` plus one `quality_r2` Muon record; all had `optim="muon"`, `precision="fp32"`, `resume_state=null`, and no `strict_fp32` field. No Torch import, training-state inspection, test, fit, GPU work, network access, or code execution from the trainer was used.

## Evidence and interpretation

### Legacy `--optim muon`: the omitted flag defaults to BF16 Newton–Schulz

- In frozen [`src/wnn/training_dynamics.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training_dynamics.py:113), `common` contains only LR, betas, and weight decay. The ungrouped legacy branch constructs `MuonAdamW(..., **common)` at lines 115–117; the grouped branch does the same at lines 123–146. Neither forwards `args.precision` or `strict_fp32`. The optimizer-switch constructor at lines 216–244 also omits it.
- [`src/wnn/optim.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/optim.py:73) declares `strict_fp32=False` by default and stores that default in each group at lines 85–95. With that value, `orthogonalize` explicitly casts CUDA gradients to BF16 at lines 32–48. The step path calls that non-strict branch at lines 145–161. Strict mode would instead validate FP32 parameters, gradients, state, and Newton–Schulz intermediates at lines 22–29 and 112–159.
- The models themselves remain FP32: [`setup_model.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/setup_model.py:110) moves the model to the device without a dtype cast. [`optimization.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/optimization.py:108) sets `use_autocast` false when `args.precision == "fp32"`; [`forward_backward.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/forward_backward.py:27) applies that setting to model forward. Thus the accurate identity is **FP32 model forward + BF16 Muon orthogonalization on CUDA**, not end-to-end strict FP32.
- The shared recipe explicitly selects `--precision fp32 --optim muon` ([`recipe_args.sh`](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/recipe_args.sh:3)). `quality_r2/run_fit.sh` invokes `strict_launch.py` ([lines 9–15](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/run_fit.sh:9)); `local_r1/run_cell.sh` uses that wrapper for train-only runs and `final_launch.py` for `FINAL` runs ([lines 8–13](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/local_r1/run_cell.sh:8)). Both wrappers set CUDA matmul/cuDNN TF32 false and cuDNN RNN FP32 precision to `ieee`; neither passes `strict_fp32` to the optimizer. `NVIDIA_TF32_OVERRIDE=0` also does not set that Python optimizer-group option.
- The trainer also calls `configure_strict_fp32()` when `precision == "fp32"` ([`setup_lineage.py` lines 202–203](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/setup_lineage.py:202)). That helper only disables CUDA matmul and cuDNN TF32 ([`resources.py` lines 80–85](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/resources.py:80)); it does not set optimizer-group `strict_fp32` or Newton–Schulz dtype.
- [`strict_launch.py`](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-29/teacher_track_r1/strict_launch.py:13) prints `STRICT_FP32` beside backend flags, but does not inspect the optimizer group or Newton–Schulz dtype. [`final_launch.py`](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/local_r1/final_launch.py:15) has the same precision controls. Neither explicitly calls `torch.set_float32_matmul_precision("highest")`; the strict optimizer guard requires `highest` if strict mode is enabled.
- This is not merely an absent log annotation: all 19 relevant saved argument records are cold starts (`resume_state=null`), select the legacy optimizer, and lack a strict option. There is no resumed optimizer group that could have carried a saved strict flag into these runs.

### Historical TL3 teacher path is separate and strict

The teacher record [`tt29rep_fbc1k_s5.fit.log`](/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-29/teacher_track_r1/tt29rep_fbc1k_s5.fit.log:2) declares `optim="tl3-muon"`, `precision="fp32"`, and no resume. In the frozen trainer, the `tl3-muon` branch passes `strict_fp32=(precision == "fp32")` ([`training_dynamics.py` lines 99–112](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training_dynamics.py:99)). Its separate builder sets `ns_dtype=torch.float32` for that flag ([`tl3_optimizer.py` lines 238–279](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/tl3_optimizer.py:238)); strict steps validate the runtime environment and FP32 operands. This correct TL3 path does **not** carry over to later fits logged as `optim="muon"`. In particular, the `tt29_parent_s3` record and the current student cells select legacy `muon`.

### Correction to the runtime-identity claim

The frontier's “strict runtime identity verified” statement is supported only for the flags printed by the launcher. It does not establish strict optimizer precision. The logged training results remain valid observations of the recipe that actually ran, but must not be labeled as strict-FP32 Muon results. This audit does not inspect model/training-state artifacts; that work remains with the root as directed.

## Limits and validity

This is a source/config audit, not a numerical measurement of the affected checkpoints. It establishes the executed optimizer mode from frozen source, launch commands, and saved argument records; it does not estimate how much BF16 Newton–Schulz changed any score. The TL3 comparison establishes the separate code path and teacher config, not equivalence between TL3 Muon and legacy Muon.

## Follow-up: isolated correction and test scope

Root's CPU-only training-state check confirmed the old `q14_kd90_s4` optimizer groups had `strict_fp32=False`. The fit stopped after two complete epochs and a partial epoch 2; there is no fixed ep3 endpoint, and conditional delivery is disabled. Preserve its score only as an observation of FP32 model forward with BF16 Muon Newton–Schulz. It is not a scientific rejection and must not be relabeled strict FP32.

The isolated correction is commit `48776fba8af26b16b7e6813def7ccf571f56c1ac` in `/home/vlad/development/wunder/research_tmp/q14_strict_source`, based on the frozen source. Its only source change is `src/wnn/training_dynamics.py`: both legacy Muon constructor branches now pass `strict_fp32=(precision == "fp32")` ([patch artifact](precision_fix_r1/strict_factory.patch)). The source worktree was clean at that commit. It does not alter `switch_optimizer`; this matches the current frozen recipe, whose `optim_switch` is empty. The scope is therefore adequate for the direct and head-grouped non-switch factory paths used here, but it is not a general fix for dynamically switching AdamW to Muon.

The already-run CUDA check in [`check_precision.py`](precision_fix_r1/check_precision.py) exercises both factory branches (`head_lr_mult` 1 and 2), three optimizer steps per branch, and 24 strict Newton–Schulz calls on an RTX 3060 Laptop GPU with Torch 2.14.0+cu130. It asserts strict dispatch on every call, CUDA FP32 gradients/results, FP32 optimizer state, and the strict kernel's internal dtype/environment guards. The FP16 control remains non-strict. [`CUDA_CHECK_R1.json`](precision_fix_r1/CUDA_CHECK_R1.json) records pass and zero scientific fits; [`negative_old_factory.log`](precision_fix_r1/negative_old_factory.log) records the expected failure against the old factory. The structure-check receipts report 124 files and zero errors before and after. These are root-produced artifacts; no tests were executed by this audit.

**Adequacy:** this directly verifies the defect and correction for both non-switch construction branches, including actual CUDA Newton–Schulz execution. It is sufficient for the frozen recipe's optimizer-factory scope. Remaining limits are no `optim_switch` coverage, no optimizer-state resume round-trip, and no full quality fit or score on the corrected source. The test model is a small GRU used to exercise the real factory and optimizer; it is not a model-quality replication.

Profile R1 was rejected because profiling early exits cannot save epoch continuation state ([log](q14s_kd90_profile_s3.fit.log)). Root reports corrected `profile R2` is running. Its log records `precision=fp32`, legacy `optim=muon`, `save_training_state=false`, `max_batches=6`, and 120 completed optimizer updates with valid throughput evidence. The wrapper removes only `--save-training-state` from the shared base arguments in profile mode ([`run_strict_fit.sh` lines 9–10](run_strict_fit.sh#L9)); it retains the frozen recipe and adds the declared profile limits. This is engineering/throughput evidence, not a validation score or quality endpoint. The strict optimizer guard ran on those updates; the launch path does not set matmul precision to `highest` explicitly, so portability continues to rely on the guard rejecting any runtime where that invariant is absent.

The CUDA check is a standalone campaign artifact rather than a committed CI test. For a lasting regression guard, carry its direct/grouped factory assertions into `tests/test_training_dynamics.py`. The current check's reduced-precision negative control uses FP16 (not a separate BF16 case); neither limitation affects the frozen recipe, which is FP32 and has no optimizer switch or resume.

## Decision

Keep the original source, logs, checkpoint, score, and disabled-delivery state unchanged. The isolated patch and CUDA checks are adequate for the current no-switch precision defect. Do not mix corrected-source endpoints with the old BF16-NS controls or describe old results as strict FP32. Any later quality claim requires complete matched fits from the corrected source and the existing campaign gate; profile output alone cannot promote an endpoint.

## Source records

- Frozen contract and current recipe: `research/campaign_2026-11-14/quality_r2/CONTRACT_R1.json`, `recipe_args.sh`, `run_fit.sh`, and the 19 fit-log argument records described above.
- Local launch paths: `research/campaign_2026-11-14/local_r1/run_cell.sh` and `final_launch.py`.
- Separate TL3 teacher record and launcher: `research/campaign_2026-10-29/teacher_track_r1/tt29rep_fbc1k_s5.fit.log` and `strict_launch.py`.
- Corrected-source evidence: `research/campaign_2026-11-14/quality_r2/precision_fix_r1/strict_factory.patch`, `check_precision.py`, `CUDA_CHECK_R1.json`, `negative_old_factory.log`, `research/campaign_2026-11-14/quality_r2/run_strict_fit.sh`, and both `q14s_kd90_profile_s3*.fit.log` records.
- Frozen trainer source: `/home/vlad/development/wunder/research_tmp/pr49_wt`, commit `188c0788e571c6f981cdc6827fef0d50bd38cf14`.
- Corrected isolated trainer source: `/home/vlad/development/wunder/research_tmp/q14_strict_source`, commit `48776fba8af26b16b7e6813def7ccf571f56c1ac`.
