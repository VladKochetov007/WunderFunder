# Faithful Student Stage Readiness (H230 Seed99/Seed9)

## Question 1: H230 Seed3/Seed4 Student Configuration

**Argv source**: `/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/bd26_h230_all_data_student_gru256_seed3_v1/provenance.json`

**Parent checkpoint**: `artifacts/runs/bd26_h230_all_data_parent_b16_ep3_retry3_v1/model_ep3.pt`
- Parent recipe: All-data B16 parent trained from sealed source-v2, ep3 checkpoint warm-start.

**Distill cache root**: `/home/vlad/development/wunder/wnn_connectome_starterpack/cache/bd26_h230_fold0_teacher_family_all_rows_v1`
- `distill_cache_all_data: true`
- `distill_cache_variant: "source"` (teacher target is weighted sum)

**Student training hyperparameters**:
- `alpha: 0.7` (distill_alpha)
- `lr: 0.0001` (learning rate)
- `epochs: 2` (epochs)
- `batch: 16` (batch size)
- `chunk: 2000` (chunk size)
- Family loss: `distill_family_weight: 0.1` (four-family loss weight is 0.1)
- Family target mode: **`"raw"`** (line 62, provenance.json) - no ±2 clipping on family targets
- Aux losses: `aux_next: 0.3`, `aux_struct: 0.3`
- Optimizer: `adamw`; Scheduler: `onecycle`
- Propensity weighting: `"c20260929_prop_f0"` with `prop_power: 2.0`, `prop_eps: 0.03`

---

## Question 2: Faithful Student on Seed99 Muon Parent

**Source tree to use**: `h230_source_v4_faithful_muon`
- Location: `/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon`
- Basis: sealed source-v3 loaded+extended with seed99 checkpoint; fresh state + AdamW + OneCycle per ADR0023

**Parent checkpoint path**:
- `artifacts/runs/bd26_h230_strong_parent_seed99_muon_local_v1/model_ep3.pt` (verified present)

**Draft argv** (by analogy to seed3):
```bash
python scripts/train_rnn.py \
  --name "bd26_h230_strong_student_seed99_muon_v1" \
  --device cuda \
  --arch gru --cell gru --hidden 256 --layers 2 --dropout 0 \
  --seed 99 \
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --valid-fold -1 --eval-source dev --train-source "train+valid" \
  --train-seqs 10607 \
  --init "bd26_h230_strong_parent_seed99_muon_local_v1@ep3" \
  --epochs 2 --checkpoint-policy fixed --checkpoint-epoch 1 --no-eval-during-fit \
  --batch 16 --chunk 2000 \
  --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --propensity "c20260929_prop_f0" --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 \
  --aux-struct 0.3 --aux-next 0.3 \
  --distill-cache-root cache/bd26_h230_fold0_teacher_family_all_rows_v1 \
  --distill-cache-variant source \
  --distill-alpha 0.7 --distill-family-head --distill-family-weight 0.1 \
  --distill-family-target-mode raw --distill-boot-h 0 \
  --fast --save-training-state
```

**Uncertain flags:**
- Exact `--name` format (draft: `bd26_h230_strong_student_seed99_muon_v1`)
- Whether distill_cache_all_data should be explicitly set (not in command, may be inferred from cache membership)

---

## Question 3: Teacher Cache Composition

**Cache main_source** = weighted sum (unnormalized): `0.30 strong + 2×0.10 full-family + 2×0.125 raw-family + 0.0625 baseline = 0.8125`

**Does cache provide the teacher target the student needs?** YES.
- `main_source_train.npy` and `main_source_valid.npy` contain the already-blended teacher predictions
- Recipe does not require parent's own predictions inside the cache (teacher is pre-computed weighted sum)
- Metadata fields: `"membership_mode": "all_supplied_train_valid"`, `"heldout_claim": false`

---

## Question 4: Raw vs. Clipped Family Target Switch

**Existing flag**: `--distill-family-target-mode {raw|clip2}` (CLI argument; default `"clip2"`)

**Code location**: `/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v3_all_data_distill/src/wnn/distillation.py` line 180

**Function signature**:
```python
def family_targets_for_loss(targets: torch.Tensor, mode: str = "clip2") -> torch.Tensor:
    # mode: "clip2" preserves historical behavior (targets.clamp(-2, 2))
    # mode: "raw" keeps teacher values unchanged
```

**For one-line treatment**: The flag is already implemented as a CLI argument; no code change needed. Pass `--distill-family-target-mode raw` to replicate seed3/seed4 behavior.

---

## Question 5: Measured Wall Time and Export Commands

**H230 student fit times** (seed3/seed4):
- Seed3: epoch 0 = 978.3 s, epoch 1 = 989.3 s → **total ~1968 s (33 min)**
- Seed4 (retry1): epoch 0 = 972.8 s, epoch 1 = 955.2 s → **total ~1928 s (32 min)**

**Log files**: 
- `/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/runtime/logs/h230-all-data-student-gru256-seed3-v1-20261008.log`
- `/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/runtime/logs/h230-all-data-student-gru256-seed4-retry1-v1-20261008.log`

**Submission export/packaging**:
```bash
# Build native extensions (x86-64-v2 and v3)
scripts/build_submission.sh RUN_NAME

# Steps inside:
1. Docker build native extension (Dockerfile.build)
2. For each level (v2, v3):
   - maturin build --release --features extension-module
   - Extract .so files from .whl
3. Export weights to npz: torch model.pt → np.savez(weights.npz)
   - Converts model checkpoint to float32 numpy: {k: v.float().numpy() for k,v in ck["model"].items()}
4. Package: solution.py + weights.npz + v2/*.so + v3/*.so → submission.zip
5. Verify ABI: python3 scripts/check_submission_abi.py submission.zip
```

**Package command** (line 26 in build_submission.sh):
```bash
cd pkg && python3 -m zipfile -c ../submission.zip solution.py weights.npz v2 v3
```
