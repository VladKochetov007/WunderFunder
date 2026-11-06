---
kind: experiment
status: implementation_prepared_remote_acceptance_pending
created_utc: 2026-11-06T00:00:00Z
updated_utc: 2026-11-06T00:00:00Z
hypothesis_ids: [H-B300-TRANSFORMER-SWIGLU-20261106]
experiment_ids: [E-B300-TRANSFORMER-SWIGLU-20261106]
evidence_paths: []
---

# Matched SwiGLU Transformer teachers on B300

**Code and launch packet prepared; no receiver checks, profiles, fits, cache exports, inference or scoring have run.** The four-teacher panel and the separately allocated four-student transfer panel are both proposals. Current actual counts are zero. The branch starts at `research-team` commit `7e407431`; the implementation commit and PR URL will be added after publication.

This packet exposes optional `ff_activation` (`gelu` by default or `swiglu`) and `ff_hidden` options in the sequence Transformer. Historical recipes omit both keys: their loaded configuration, GELU state-dict keys, parameter creation order and default forward remain unchanged. SwiGLU computes `silu(gate(norm(x))) * up(norm(x))`, then applies the existing output projection. Its gate and up matrices are separate parameters and separate Muon matrices.

## Why this experiment exists

Earlier team plans already proposed SwiGLU: the [causal Transformer / distillation idea](../../unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md) uses a 768-wide, 2048-intermediate SwiGLU model, and the [large dynamics idea](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md) lists SwiGLU as its backbone. The current causal sequence runtime is GELU. This work implements and tests that planned mechanism in the current bounded B300 teacher and student pipeline; it is not a novelty claim.

The primary source, [Shazeer, “GLU Variants Improve Transformer”](https://arxiv.org/abs/2002.05202), reports that some gated feed-forward variants improved quality over common ReLU/GELU Transformer sublayers in its sequence-to-sequence experiments. That is a reason to test a gated numerical-sequence block, not evidence of a gain on this competition.

The comparison holds the FFN matrix count and leading dense FFN GEMM work constant. For width 768, GELU uses `768×3072` and `3072×768`; SwiGLU uses two `768×2048` projections and one `2048×768` projection. Both have 4,718,592 FFN matrix weights and the same leading FFN matrix multiply count per token. SwiGLU adds an elementwise SiLU and product, has 1,024 more FFN bias values per block, and runs three independently optimized FFN matrices instead of two. This is not an exact total-parameter, elementwise-FLOP, optimizer-work or throughput match.

The two teacher arms use the same fold and seed, but their tensors are not bitwise paired: the FFN shapes differ and consume different random draws. Pairing is by fitting membership, seed, schedule, compute budget and evaluation membership. Any observed effect belongs to that implemented parameterization, initialization and precision recipe.

## Frozen panel

| Arm | F0 / teacher seed | F2 / teacher seed | `ff_hidden` |
|---|---|---|---:|
| GELU reference | `gelu_f0_s101001` | `gelu_f2_s101002` | 3072 |
| SwiGLU candidate | `swiglu_f0_s101001` | `swiglu_f2_s101002` | 2048 |

Each teacher uses T768×24, 12 heads, context 1024, chunk 256, accumulation 4, batch 1 per rank, eight B300 ranks, 30 epochs, Muon LR `3e-4`, 5% warmup, cosine floor 0.1, matrix decay 0.01 and clip norm 1. The model uses BF16 autocast with process-wide cuBLAS BF16x9 emulation; master parameters, gradient accumulation, loss, optimizer state and exported predictions remain FP32. Exact-window efficient attention and compiled Transformer blocks are explicit. This follows the [24 October precision amendment](../b300-transformer-distillation-runtime/PRECISION_AMENDMENT_20261024.md). BF16x9 may affect Muon GEMMs and is not IEEE-FP32 parity.

All configs set `normalization_scope=all_fit` so every normalization value comes from that fold's complete fitting membership. `normalization_sequences=32` remains only for compatibility with the sequence recipe schema. No hints are exported. Final epoch 29 is fixed; the outer score cannot select an epoch. Complete-sequence resets, warmup rows, source-qualified folds and the canonical evaluator stay in the existing runtime.

| Planned panel | New fits | Reused endpoints | Cost |
|---|---:|---:|---:|
| Initial teacher comparison | 4 total; 2 GELU and 2 SwiGLU | 0 | 120 teacher epochs |
| Separate output-KD transfer follow-up | 4 total; 2 students from each teacher family | 0 | 120 student epochs |

The initial idea screen is the four-teacher panel. The four T128×2 student fits are a separate allocation, not part of the initial screen and not an automatic successor. The same F0/student seed 1001 config is used under both teacher caches; F2 uses the same seed 1002 config under both caches. Each student uses one B300, context 128, 30 epochs, AdamW LR `3e-4`, BF16/BF16x9, and an equal label-plus-output-KD objective (`kd_weight=1`, `hint_weight=0`). There is no labels-only student control, so the follow-up measures whether these teacher families produce different KD students; it does not measure KD benefit over labels alone.

All F0/F2 evaluation groups are exposed development evidence. Fold and seed change together, so two paired cells cannot separate fold from seed variability, establish significance, or estimate public transfer. The `+0.0005` WP threshold below is an allocation trigger only.

## Files

- [PR.md](PR.md): the eleven-section scientific and execution contract.
- [recipe.json](recipe.json): IDs, costs, cells, source pin, decision rule and evidence limits.
- [configs/teacher/](configs/teacher/): four full teacher recipes, GELU/SwiGLU × F0/F2.
- [configs/student/](configs/student/): the two exact student recipes shared within each matched fold.
- [run.sh](run.sh): explicit receiving-host checks, per-cell profile/fit/evaluate/cache/bundle stages; it has no queue and never approves profiles.
- [validate_teacher_cache.py](validate_teacher_cache.py): read-only binding of the actual eight-rank fit cache to its one-rank parity export, selected teacher checkpoint, fold/seed and current canonical split. It records the parity receipt and manifest hashes for each student stage.
- [compare_scores.py](compare_scores.py): official-score validation and a fixed 2,000-draw component bootstrap for one paired fold; intervals are conditional on these fits.
- `tests/test_sequence_swiglu.py`: receiver-only checks for GELU compatibility, FFN matrix-budget arithmetic, config rejection, causal/chunk behavior, gradients, checkpoint reconstruction and BF16 compiled execution. They are intentionally unrun here.
- `scripts/check_sequence_muon_ddp.py`: the two-rank receiver check now accepts `--ff-activation gelu|swiglu` and checks semantic optimizer ownership for each.

The cache parity stage reuses the canonical [R4 cache checker](../b300-transformer-distillation-runtime/high_compute_r4/cache_parity.py), which checks actual full-fit arrays, manifests and checkpoint ancestry. `scripts/train_sequence.py`, `scripts/predict_sequence.py` and `scripts/score_sequence.py` remain the canonical trainer, exporter and official scorer.

## Receiver run sequence

Use the full committed source SHA after publication. Set canonical source-qualified split paths and a receiver-owned output root on an approved local disk, then call one stage per command. The `check` stage records `findmnt`, `df` and `lsblk` output; the receiver must confirm the selected mount has the required capacity and is the intended persistent physical storage before proceeding. This packet does not start a queue or grant GPU allocation.

```bash
export PACKET="$PWD/research/experiment_drafts/transformer-swiglu-20261106"
export WNN_EXPECTED_COMMIT="<full reviewed source commit SHA>"
export WNN_DATA_ROOT="<canonical data root>"
export WNN_SPLIT_F0="<canonical F0 manifest>"
export WNN_SPLIT_F2="<canonical F2 manifest>"
export WNN_RUN_ROOT="<dedicated physical-disk output root>"
export CUDA_VISIBLE_DEVICES="<eight allocated B300 IDs or UUIDs>"
bash "$PACKET/run.sh" check
```

With eight allocated B300s, run all required stages for one teacher cell at a time. For each teacher, review all three original unapproved profile receipts; preserve them unchanged. Only the exact eight-rank production receipt can match the full teacher fit. After the human review, copy `teachers/<cell>/profile-eight/profile_receipt.json` to `teachers/<cell>/approved_profile.json` and set `approved=true` with a review note. The code validates its complete fit identity before training.

```bash
bash "$PACKET/run.sh" teacher-profile-one gelu_f0_s101001
bash "$PACKET/run.sh" teacher-profile-eight gelu_f0_s101001
bash "$PACKET/run.sh" teacher-profile-peak gelu_f0_s101001
# Human reviews receipts, then copies only the exact eight-rank production receipt and approves that copy.
bash "$PACKET/run.sh" teacher-fit gelu_f0_s101001
bash "$PACKET/run.sh" teacher-evaluate gelu_f0_s101001
bash "$PACKET/run.sh" teacher-cache-eight gelu_f0_s101001
bash "$PACKET/run.sh" teacher-cache-one gelu_f0_s101001
bash "$PACKET/run.sh" teacher-cache-parity gelu_f0_s101001
bash "$PACKET/run.sh" teacher-bundle gelu_f0_s101001
```

Repeat those same named commands explicitly for `swiglu_f0_s101001`, `gelu_f2_s101002` and `swiglu_f2_s101002`, using eight allocated devices. Do not wrap scientific cells in a shell loop. Once both scores for a fold are present, compute its paired delta:

```bash
uv run --no-sync python "$PACKET/compare_scores.py" \
  --candidate "$WNN_RUN_ROOT/teachers/swiglu_f0_s101001/fit/score.json" \
  --reference "$WNN_RUN_ROOT/teachers/gelu_f0_s101001/fit/score.json" \
  --output "$WNN_RUN_ROOT/teacher-pair-f0.json"
```

Repeat for F2 with its two `s101002` teacher cells. Consider the separately allocated student panel only if the arithmetic mean of these two fixed deltas is at least `+0.0005`, neither delta is negative, and all receiver gates pass. The comparison is two exposed cells, not a significance test.

After any separately allocated student panel, use the same fixed comparator for its F0/F2 student pairs; it reproduces pooled WP and reports conditional component intervals, without making a labels-only KD claim:

```bash
uv run --no-sync python "$PACKET/compare_scores.py" \
  --candidate "$WNN_RUN_ROOT/students/student_swiglu_f0_s1001/fit/score.json" \
  --reference "$WNN_RUN_ROOT/students/student_gelu_f0_s1001/fit/score.json" \
  --output "$WNN_RUN_ROOT/student-pair-f0.json"
```

Repeat for F2 using the `s1002` student cells.

The student panel requires a separate one-B300 allocation, using the same `WNN_RUN_ROOT` so the actual teacher caches remain available. For each student, run `student-profile-one`, then `student-profile-peak`; after reviewing both, copy `students/<cell>/profile-one/profile_receipt.json` to `students/<cell>/approved_profile.json` and set `approved=true` with a review note. Run `student-fit`, `student-evaluate`, and `student-bundle`. For example:

```bash
export CUDA_VISIBLE_DEVICES="<one allocated B300 ID or UUID>"
bash "$PACKET/run.sh" student-profile-one student_gelu_f0_s1001
bash "$PACKET/run.sh" student-profile-peak student_gelu_f0_s1001
# Human reviews both receipts and approves a copy of the exact one-rank production receipt.
bash "$PACKET/run.sh" student-fit student_gelu_f0_s1001
bash "$PACKET/run.sh" student-evaluate student_gelu_f0_s1001
bash "$PACKET/run.sh" student-bundle student_gelu_f0_s1001
```

Repeat those commands explicitly for `student_swiglu_f0_s1001`, `student_gelu_f2_s1002`, and `student_swiglu_f2_s1002`. Each stage revalidates the actual teacher config, source, checkpoint, cache manifests, full parity receipt and current canonical split, and writes a binding receipt under that student's `cache-bindings/` directory. The fit bundle includes its production/peak profile receipts, approved receipt, score and every cache binding (with the embedded parity receipt). Teacher bundles include their three profiles, approved receipt, score, cache manifests and cache parity receipt. Keep the receiver acceptance bundle from `$WNN_RUN_ROOT/acceptance` alongside the cell bundles. `CUBLAS_EMULATE_SINGLE_PRECISION` must be unset for FP32 receiver checks; the script exports it as `1` for these BF16/BF16x9 profiles and fits.

## Status and limits

The receiver still needs the targeted sequence tests, two-rank DDP/Muon checks for both activations, one/eight-rank and peak-LR profiles for each teacher config, and a one-rank plus peak-LR profile for each student config. A human reviewer must inspect the real hardware/software, finite-update evidence, memory headroom, exact data/cache lineage and profiles before writing an approved profile receipt. The trainer validates that receipt against the fit identity. This preparation did not test B300 compatibility, memory fit, compilation, cache parity, learned quality or runtime.

The teacher comparison can motivate only a separately allocated T128×2 transfer panel when mean SwiGLU-minus-GELU teacher WP is at least `+0.0005` with no fold regression and the receiving profiles pass. This threshold does not establish significance. A teacher gain without student transfer narrows the tested transfer recipe; a negative teacher screen narrows these widths, schedule and precision. Neither outcome closes all gated-FFN or Transformer directions. No leaderboard effect is known.
