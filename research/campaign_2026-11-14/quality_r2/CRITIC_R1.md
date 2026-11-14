# Bounded critic — quality R2

**Scope:** static audit only; no fits or data inspection. Recommendations are not launch or promotion authority.

## 1. KD90 seed-4 identity

[`local_r1/run_cell.sh`](../local_r1/run_cell.sh) sources the frozen [teacher-track recipe](../../campaign_2026-10-29/teacher_track_r1/recipe_args.sh) and strict launcher. The saved [seed-3 config](../local_r1/l14_kd90_s3.fit.log) uses fixed ep3, PR49 teacher `tt29rep_fbc1k_s5`, alpha .9, boot 0, raw weights, 10,325 train sequences, the frozen split, and FP32. The declared q14 recipe matches; its runtime config was unavailable, so execution identity remains unverified.

**Falsifier:** q14’s logged config/provenance must differ only in name and seed. Any teacher, split, membership, precision, endpoint, or recipe mismatch voids the paired interpretation. Compare ep3 with same-seed `l14_kd50_s4`; valid is exposed screening data.

## 2. One-factor variants and prior evidence

In PR49 [`training/losses.py`](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/losses.py), `--distill-boot-h 8` uses the teacher’s t+8 output in place of the origin label only when the label stays unchanged through t+8; otherwise it falls back to the origin label. The result is blended with the current teacher at alpha .9; loss weights remain from the real label/mask. **Cheap falsifier:** measure weighted eligible-row coverage and future-vs-current teacher change; negligible exposure means little treatment.

`--eval-ema` selects EMA weights for scoring/saving ([PR49 outputs](/home/vlad/development/wunder/research_tmp/pr49_wt/src/wnn/training/outputs.py)); it does not change the KD loss. Decay `.999` is already default: the sole treatment is enabling evaluation EMA. Keep `--ema-reference-rows 0`; this tests checkpoint smoothing, not EMA-teacher distillation.

Prior-work check: [H-180](../../hypotheses.jsonl) proposed alpha × horizon 0/8; [E-338](../../experiments.jsonl) says the h8 screen was **not run**, and E-334’s `0.0001` is target-bias diagnosis, not student quality. The fbc1k teacher combines h8, EMA, and multi-teacher changes. E-423/H-071 tests row-clock decay (−0.000574, interval crosses zero), not `eval_ema`; the teammate EMA gain also changes training duration. No direct KD90 one-factor result appears in these ledgers.

**Exact flags** (append to the shared recipe; separate arms):

```bash
# control / q14
--distill tt29rep_fbc1k_s5 --distill-alpha 0.9
# bootstrap-only arm
--distill tt29rep_fbc1k_s5 --distill-alpha 0.9 --distill-boot-h 8
# EMA-evaluation-only arm
--distill tt29rep_fbc1k_s5 --distill-alpha 0.9 --eval-ema --ema-decay 0.999 --ema-reference-rows 0
```

Hold all else fixed. One seed is a screen; replicate only a survivor with a distinct paired seed inside the remaining cap. Keep h8 and EMA separate.

## 3. Final-data trust and causal scope

[`run_cell.sh FINAL`](../local_r1/run_cell.sh) switches to `train+valid`, `--include-heldout-valid`, 10,607 train sequences, and `--no-eval-during-fit`, intentionally fitting the reserved validation block. [`final_launch.py`](../local_r1/final_launch.py) disables the membership guard unconditionally; a wrong invocation can bypass it, so safety depends on reviewing the wrapper’s logged arguments.

Require logged all-data membership, no fit-time evaluation, fixed endpoint, unchanged source, and teacher-prediction provenance/coverage for every fitted group. Any score is in-sample only: these fits materialize packages, not quality or causal evidence. Rank recipes on predeclared train-only pairs and exposed-valid results; retain final fits as packaging provenance. Public transfer awaits user upload and score.
