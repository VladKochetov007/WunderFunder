---
kind: synthesis
status: paused_jobs_landing
created_utc: 2026-10-25T16:55:00Z
hypothesis_ids: [H-LOSS25-EVENT-HAZARD-R1, H-LOSS25-UV-AUX-R1, H-LOSS25-RUNAGE-MEASURE-R1, H-LOSS25-PR32-BLEND-PROBE-R1, H-LOSS25-FAITHFUL-NLL-R1, H-C3MIX-PUBLIC-WEIGHT-R1, H-I1LEAD-HELD-R1, H-OPT25-MUON-FP32-R1, H-OPT25-CAUTIOUS-R1, H-OPT25-ADEMAMIX-R1, H-OPT25-CLIP-REGIME-R1, H-OPT25-MUON-MECHANISM-R1]
---

# October 25 research session — synthesis and handoff

User-directed session (resources: 5 GB VRAM, 30 GB RAM, 10 CPU cores; later "utilize gpu",
"go crazy with research", "utilize Kaggle"; CPU reserved for EDA/theory). Ended by the user:
"let current experiments land gracefully, wrap all the research progress … stop research loop,
just finish with already running jobs." The 30-minute research loop was deleted; no new
admissions. Two jobs were left to land (see **Handoff**).

## 1. Headline results

1. **Strict-FP32 Muon beats AdamW, including an LR-tuned AdamW, and we know why.**
   On the fast cold recipe (H512×1 GRU, 10% frozen memberships, 3 epochs, fixed CP2, F0/s1001
   and F2/s1002): Muon vs AdamW 1e-3 **+0.0038 / +0.0043**; vs AdamW's optimum (5e-4)
   **+0.0026 / +0.0035**; vs AdamW 2.5e-4 +0.0039 / +0.0049 (all intervals exclude zero).
   Removing only the Newton–Schulz orthogonalization (RMS-matched momentum, `--muon-update rms`)
   returns it to AdamW level (−0.0040 vs Muon in both cells). Mechanism: AdamW collapses the GRU
   gates (45–48% of update-gate activations saturated, frozen units with timescales ~5×10⁵ rows,
   forming already in epoch 0); Muon keeps gates soft (3%, q99 τ ≈ 170 rows). AdamW faces a
   collapse-vs-undertraining trade-off over LR; Muon escapes it. Spectral capping of AdamW's
   gate blocks recovers only ~40% of the gap. Muon's step (0.2) and depth (1 layer) are on a
   plateau. [Optimizer panel](optimizer_panel_r1/README.md) · [mechanism panel](mechanism_panel_r1/README.md) ·
   [saturation panel](saturation_panel_r1/README.md) · [LR ladder](lr_ladder_r1/README.md) ·
   [diagnostics](optimizer_panel_r1/diagnostics/README.md) · [theory](optimizer_panel_r1/OPTIMIZER_THEORY_R1.md).
2. **Policy violation found and fixed:** torch 2.14 defaults cuDNN RNN to TF32
   (`cudnn.allow_tf32=True`, `cudnn.rnn.fp32_precision='tf32'`); every local/B300 "strict FP32"
   GRU fit used TF32 recurrences (Kaggle T4 runs were true FP32). New `--strict-fp32-kernels`
   costs no measurable quality (−0.0011 / −0.0001). Use it in all new recipes.
3. **Loss-family diagnosis:** |y|-weighted MSE is Bayes-aligned with pooled WP, so distance
   changes (NLL, Pearson, Huber, KD variants) cannot move the target; NLL's test was clipping-
   confounded. Per-target normalization is a no-op (t0 mass 0.496). Class-3 up-weighting is
   negative in two regimes. Clipping at 1.0 helps (removing it −0.0025; 0.3 neutral) as implicit
   robust batch reweighting. [Loss review](loss_ideation_r1/README.md) ·
   [class-3 weighting](class3_public_mix_r1/README.md) · [clip panel](clip_regime_r1/README.md).
4. **Information audits (negative):** label change at book events is trivial quote arithmetic
   (AUC 1.0); causal microstructure hand features add no transferable residual information
   (replicates Sept E-092/E-096); FTT's 52 "lead" inputs were fingerprinted as i1 values held at
   the last i0 book event, but adding them to the GRU hurts (−0.0049 / −0.0016).
   [Class-3 reframe](class3_reframe_r1/README.md) · [FTT fingerprint](class3_reframe_r1/FTT_LEAD_BLOCK_FINGERPRINT_R1.md) ·
   [i1 lead screen](i1_lead_panel_r1/README.md).
5. **Dynamics auxiliary (cost-deferred Oct-21 panel) executed:** inconclusive (mean +0.00015;
   consistent gain only in rows 99–500). [Dynamics panel](dynamics_panel_r1/README.md).
6. **PR32 (teammate B300 Transformers) memorize:** val weighted R² ≈ 0.14 vs train ≈ 0.995;
   1.54B val WP fell 0.37→0.27 (F0); fits stopped by the teammate; analysis and requests posted
   on PR32.

## 2. Full experiment ledger (this session)

| Experiment | Fits | Result | Status |
|---|---:|---|---|
| Class-3 weight 1.43, warm LR3e-5 (F0) | 1 | −0.00006 | null |
| Class-3 weight 1.43, stage-6 LR1e-4 (F0) | 2 | −0.00022 | closed negative |
| Dynamics aux 0 vs 0.05 (F0/F2) | 4 | +0.00046 / −0.00016 | inconclusive |
| i1 lead block (F0/F2) | 4 | −0.0049 / −0.0016 | closed negative |
| Muon / Cautious / AdEMAMix (F0/F2) | 6 | +0.0038/+0.0043; −0.0014/−0.0011; −0.0045/−0.0038 | Muon supported |
| Clip none / 0.3 (F0/F2) | 4 | −0.0018/−0.0032; −0.0011/+0.0001 | keep clip 1.0 |
| Mechanism panel (F0/F2) | 12 | see headline 1–2 | supported two cells |
| Saturation panel (F0/F2) | 6 | cap +0.0011/+0.0020 (Muon +0.0027/+0.0023 over it); Muon .3 ≈ .2; H512×2 ≈ ×1 | done |
| AdamW LR 2.5e-4 (F0/F2) | 2 | −0.0013/−0.0014 vs 5e-4 | AdamW optimum ≈5e-4 |
| Muon stage-6 production lineage (F0) | 1 | **landing** | running |
| Kaggle reproduction R3 (F0/F2, seeds 2001/2002, T4) | 4 | **landing** | running |

All fits on exposed folds, one seed per cell; paired component-bootstrap intervals condition on
the fitted models. Zero-fit audits and EDA used CPU only.

## 3. Operations and incidents

- Three laptop restarts (12:31 clean, 12:52 abrupt with no logged cause, 13:53 clean); partial
  fits discarded and refit; thermal guards tightened to freeze ≥92 °C / thaw ≤84 °C.
  [Ops note](../unified/operations/VSCODE_CPPTOOLS_AND_RESETS_20261025_R1.md).
- VS Code cpptools indexed 18k third-party headers (5.9 GB, >100% CPU); workspace excludes cut it
  to 31 MB. Allowlisted [RAM guard](../unified/operations/ram_guard/README.md) runs as `wnn-ram-guard`.
- gpu-unlock (third-party kernel module) not installed: RTX 3060 Mobile (10de:2520) explicitly
  unsupported, and it only accelerates FP16/BF16/TF32, not our FP32 training.
- Kaggle: R1 and R2 workers stalled re-hashing ~35 GB of parquet per worker in
  `verify_portable_source` (located by faulthandler stack dumps); fixed by a shared, lock-
  protected digest cache in `scripts/kaggle_traced_train.py`. First versions of new kernels twice
  failed at provider startup (empty log) right after dataset creation; identical re-pushes ran.
  [Kaggle reproduction](kaggle_muon_repro_r1/README.md).
- External reviewers: Grok 4.7 and OpenCode (DeepSeek/GLM/Nemotron free) for ideation; Grok,
  Codex Astra xhigh and an Opus red-team subagent for the optimizer theory (Gemini-4-Argon failed:
  insufficient funds). [Reviews](optimizer_panel_r1/external/README.md).

## 4. New code (all tested, ruff/structure checks pass)

`--optim cautious_adamw|ademamix` (`src/wnn/optim_variants.py`); `--muon-update rms` and
`--muon-rms` (`src/wnn/optim.py`); `--strict-fp32-kernels`; `--spectral-cap/--spectral-cap-every`
(`src/wnn/spectral_cap.py`); research feature `i1_held_i0book` (`src/wnn/i1_lead_block.py`);
Kaggle `scripts/kaggle_traced_train.py`; `.vscode/settings.json` cpptools limits.

## 5. Handoff — jobs left landing (no further admissions)

1. **Muon stage-6 production lineage** (`wnn-muon6`, guard `wnn-muon6-thermal`): strict-FP32
   Muon on the warm distilled student recipe (init `c20261003_combo_s3_f0@ep3`, 6 epochs, LR 1e-4,
   fixed CP5) vs reused AdamW control 0.6812108. Its runner exports and grades automatically →
   `optimizer_panel_r1/grade_stage6_muon_f0_R1.json`. Caveat (Opus): it starts from AdamW weights
   whose update gate is already collapsed, so it tests transfer, not the cold mechanism.
2. **Kaggle R3** (`vladkochetov07/wnn-muon-repro-20261025-r3-runner` v2): four fits, fresh seeds,
   T4, true FP32. Collect at terminal status:
   `kaggle kernels output vladkochetov07/wnn-muon-repro-20261025-r3-runner -p research_tmp/kmr3_out_v2`,
   copy `results/*/artifacts/runs/*` run dirs into `artifacts/runs/`, export with
   `scripts/predict_run.py <run> valid --epoch 2 --campaign-split-manifest <fold json>
   --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name <run>`,
   then grade AdamW vs Muon per fold with `scripts/screen_score.py`.

## 6. Next steps when research resumes (ranked)

1. Collect both landing jobs; if Muon transfers to the warm production student and reproduces on
   Kaggle, plan a **full-data strict-FP32 Muon retrain** of the student/teacher lineage and a
   matched package comparison (complete endpoint, CPU replay, size/runtime).
2. Make `--strict-fp32-kernels` mandatory in new recipes (ADR/policy note).
3. Muon at full data and with distillation in a cold recipe (the mechanism was shown cold, 10%).
4. Remaining optimizer ideas: Muon on input/head matrices with norm matching; per-block
   recurrent clipping; Codex's direction×size shadow-state 2×2.
5. Endgame items from the Oct-24 value review still stand (upload the two quota ZIPs under a
   pre-declared rule; FTT source and lead-feature code from the teammate; PR32 blend probe).
