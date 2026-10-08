# Fold0 Fast Screen Protocol — October 8, 2026

## 1. Fold0 Training Membership & Validation Group Exclusion

**Split Definition:** research/campaign_2026-09-29/artifacts/splits/fold0.json:09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d

The fold0 split enforces:
- **Train groups (clean fit):** 10,535 components (exactly matched by `train_groups` in provenance)
- **Valid-fit development groups:** 1,405 components (fit-time evaluation only, not held-out)
- **Evaluation groups (held-out):** 468 components (frozen, never touched during fit)
- **Total sequences:** ~1.87M rows across all groups; ~20k rows per evaluation component (valid_group count: 1873 total, 468 marked evaluation, 1405 marked fit-dev)

**Flags for pure fold0-train-only, no eval-group leakage:**
```bash
--split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
--train-source train+valid \  # Include fit-valid groups in training
--no-eval-during-fit \        # Never load or evaluate on evaluation groups
--valid-fold 0                # Signal fold0 choice
```

No `--include-heldout-valid`, no `--fit-groups-manifest` override. Propensity cache must match the manifest:
```bash
--propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03
```

## 2. Prediction & Scoring Pipeline

**Latest working reference:** H272 direct-label readout (used in H272_S4_CLEANFOLD_V2 grade 2026-10-08).

**Prediction step** (scripts/predict_run.py; cite: scripts/evaluate_bandit_continuation.py:92-95):
```bash
cd /home/vlad/development/wunder/wnn_connectome_starterpack && \
.venv/bin/python scripts/predict_run.py {RUN_NAME} valid --epoch {CKPT}
# Output: cache/pred_{RUN_NAME}_valid.npy (shape: [num_valid_rows, 1], fp32 or checkpoint precision)
```

**Scoring step** (scripts/evaluate_bandit_continuation.py; cite: gmetric.py METRIC_ID + WARMUP constants):
```bash
.venv/bin/python scripts/evaluate_bandit_continuation.py \
  --candidate-run {RUN_NAME} \
  --candidate-epoch {CKPT} \
  --split research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --grader-spec research/campaign_2026-10-06/bandit_continuation/local/C1V2_C0_PAIRED_GRADER_SPEC.json
# Outputs: {RUN_NAME}_grade.json (metric: global_wp_clip2_v1, warmup: 99 rows, evaluation: 468 groups)
```

**Metric definition** (src/wnn/gmetric.py): global_wp_clip2_v1 uses:
- Clipping: ±2 on predictions and labels  
- Row weights: original magnitude-based (abs(clip(y,-2,2)))
- Component bootstrap: 2000 replicates, seed 271828, over 468 evaluation components

## 3. Base GRU256×2 Config + Screen Knobs

**Strongest established local baseline:** E502 C0 (cite: research/campaign_2026-10-06/bandit_continuation/local/E502_C0_PAIRED_v2.json)

```bash
# Base fixed flags (always use):
--arch gru --hidden 256 --layers 2 --dropout 0.0
--chunk 2000 --batch 16 --loss wmse
--lr 0.0001 --wd 0.01 --optim adamw --sched onecycle
--clip 1.0 --grad-stats
--aux-struct 0.3 --aux-next 0.3
--distill-boot-h 0
--init c20261003_combo_s3_f0@ep3 \  # or s4 for seed4
--fast --pin-memory --prop-cache --fused-adam
```

**Screen knobs (vary per hypothesis):**

| Knob | S-fidelity (≈10–15 min) | M-fidelity (≈30–45 min) |
|------|-------------------------|------------------------|
| `--epochs` | 1 | 2 |
| `--checkpoint-epoch` | 0 | 1 |
| `--train-frac` | 0.5–0.6 (5,500–6,300 groups) | 1.0 (10,535 groups) |
| `--seed` | Fixed (3 or 4) | Fixed (3 or 4) |
| Expected wall time (RTX 3060) | 600–900 sec | 1100–1500 sec |

**Rank correlation S ↔ M:** UNKNOWN — historical fold0 gains at 20% fraction have failed full-fold confirmation. Recommend M-fidelity for any mechanism with non-trivial improvement or code changes.

## 4. Matched-Control Rules & Bootstrap

**Same-seed pairing:** Candidate vs. reference (e.g., C0 S3 vs. C0 S3) requires:
- Identical seed (same `--seed` value)
- Identical membership (same `--train-frac`, same training sequence order via DataLoader RNG)
- Identical initialization (same `--init` checkpoint and epoch)
- Identical evaluation mask (468 evaluation groups, 99-row warmup)

**Bootstrap intervals:** Use scripts/evaluate_bandit_continuation.py; it applies paired component-level bootstrap (2000 replicates, centered, conditional on fixed checkpoints). Cite: src/wnn/evaluation.py paired_component_bootstrap.

**Tool for paired intervals:** Existing evaluate_bandit_continuation.py handles full grading. For manual component-level confidence intervals, see: research/campaign_2026-10-06/bandit_continuation/local/C1V2_C0_PAIRED_GRADER_SPEC.json lines 78–84 (bootstrap_replicates, seed, unit="frozen connected evaluation component").

## 5. Ten Untested Hypothesis Arms (Flag-Only Changes)

All are pure train_rnn.py flag diffs; no code changes required.

| ID | Mechanism Family | Flag Diff from Base | Implementation State | Cost |
|----|------------------|----------------------|---------------------|------|
| H-265 | optimization_geometry | `--optim muon --head-lr-mult 3 --momentum 0.95` (Muon final-stage transfer, frozen parent) | Muon already in optim.py; new --head-lr-mult & momentum binding needed | 2M |
| H-266 | optimization_horizon | `--sched wsd --wsd-decay 0.2 --wsd-final 0.01` (WSD stable cooldown) | Existing scheduler; frozen beta1 confound by design | 2M |
| H-267 | stochastic_exploration | `--sched sgdr --restarts 3 --grad-noise 0.05` (finite reheat + noise pulse) | Existing SGDR; new bounded plateau-triggered policy not yet coded | 2M + code |
| H-268 | distillation_target_geometry | `--distill bd26_h230_fold0_teacher_family --distill-alpha 0.25` (teacher amplitude rescale) | Existing cache transform; freeze clipping order needed | 2S |
| H-269 | component_interaction | `--distill bd26_h230_fold0_teacher_family --distill-alpha 0.5 --loss wmse_gcorr` (own-best + heterogeneous blend endpoint) | Existing blend/cache; new target identity algebra | 2S |
| H-270 | auxiliary_representation | `--aux-struct 0.5 --aux-weight 0.8` (residual family supervision; amplitude increase) | New independently authored residual cache; existing heads | 2S |
| H-271 | stochastic_exploration×disagreement | `--aux-sign 0.2 --sched sgdr --restarts 2` (antithetic teacher-disagreement dither) | New epoch-aware target transform deterministic algebra | 2S + code |
| H-273 | representation_diversity×distillation | `--distill-cache-variant homogeneous --distill-family-weight 0.5` (raw-input family contrast) | Cache variant selection exists; true-raw family never fitted | 2S |
| (New A1) | learning_rate_decay | `--warmup-frac 0.1 --lr 0.0002 --sched onecycle --restarts 2` (early warmup + restart cycle) | Existing sched; no new code | 2S |
| (New A2) | batch_geometry | `--batch 32 --chunk 1000 --update-rows 1000` (double batch, half chunk, macrostep) | Existing detach/update machinery; numerics TBD | 2S |

**Notes:**
- H-265, H-266, H-273: proposed by Astra 2026-10-08; code-change dependencies listed separately
- H-267, H-271: require bounded implementation (code flag + algebra); estimated as "2S + code"
- (New A1, A2): not yet registered; pure existing-flag combinations for exploratory pairing

## 6. S-Fidelity Screen Template (Copy-Paste Ready)

**Train command:**
```bash
RUN_NAME="screen_test_seed3_s1"
CKPT=0  # epoch 0 fixed checkpoint
cd /home/vlad/development/wunder/wnn_connectome_starterpack && \
.venv/bin/python scripts/train_rnn.py \
  --name "$RUN_NAME" \
  --device cuda \
  --arch gru --hidden 256 --layers 2 --dropout 0.0 \
  --chunk 2000 --batch 16 --loss wmse \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle \
  --clip 1.0 --grad-stats \
  --aux-struct 0.3 --aux-next 0.3 --distill-boot-h 0 \
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --train-source train+valid \
  --no-eval-during-fit \
  --valid-fold 0 \
  --init c20261003_combo_s3_f0@ep3 \
  --seed 3 \
  --epochs 1 \
  --checkpoint-policy fixed --checkpoint-epoch 0 \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --fast --pin-memory --prop-cache --fused-adam
```

**Predict + grade:**
```bash
.venv/bin/python scripts/predict_run.py "$RUN_NAME" valid --epoch $CKPT && \
.venv/bin/python scripts/evaluate_bandit_continuation.py \
  --candidate-run "$RUN_NAME" \
  --candidate-epoch $CKPT \
  --split research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --grader-spec research/campaign_2026-10-06/bandit_continuation/local/C1V2_C0_PAIRED_GRADER_SPEC.json
```

**Expected wall time:** ~600–900 sec on RTX 3060; metric output appears in artifacts/runs/{RUN_NAME}/grade.json.

---

**Document version:** SCREEN_PROTOCOL_20261008  
**Fold:** 0 only; 468 held-out evaluation groups  
**Metric:** global_wp_clip2_v1 (src/wnn/gmetric.py)  
**Split manifest path:** research/campaign_2026-09-29/artifacts/splits/fold0.json (sha256: 09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d)
