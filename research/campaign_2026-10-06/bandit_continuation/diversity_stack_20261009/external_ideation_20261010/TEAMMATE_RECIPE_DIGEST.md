# Teammate recipe digest: best public submission (0.6674)

Read-only mining of the teammate refs. No checkout, merge, training or python execution.
Refs (short names used below):

- `CI` = `origin/research/creative-iteration-20260930`
- `AR` = `origin/archive/2026-10-01`
- `QP` = `origin/quant-probes`
- `CE` = `origin/cpp-engine`

## Identity and attribution caveat

- The best recorded public score in the teammate material is **0.6674**, submission `26n9pg4y`, file `distilled_h256_pair.zip` (24m57s, 640 MB). Source: `CI:research/CURRENT_STATE.md` lines 197-199; `CI:research/iterations/2026-10-05-wave20/candidate-manifest.json`.
- The wave20 artifacts are in a single commit touching them, `dfc4b31a` by Maxim (2026-10-06, "preserve October 1-6 experiments"). Author counts on `CI`: 103 Maxim, 123 Vlad Kochetov, 17 nb.
- The notes call the parent recipe the "Vlad recipe" (Muon, chunk 1000, stopped bootstrap h8): `CI:research/iterations/2026-10-02-slow-fast/README.md` lines 174-175, 202-203. The user's own recipe is therefore the parent of the 0.6674 lineage. Its own public score is recorded as 0.6660 (seed-4 combo) at `CI:.../2026-10-02-slow-fast/README.md` line 197.
- The two 0.6674 seeds are `w18final_distilled_s5` and `s6`, produced by `CI:research/iterations/2026-10-05-wave18/final_student_lane.sh`. Member list: `CI:research/iterations/2026-10-05-wave20/distilled_h256_pair/members.json`.
- Earlier public best on `AR`: `candV1_ft2` at 0.6641, the 4-member "C" ensemble plus 2-epoch head FT (`AR:snapshot/2026-10-01/SOTA_STATE.md`).
- A later user-confirmed two-stream pair (`distilled_twostream_h256_pair.zip`, `CI:research/experiments.jsonl` E-20261006-TWOSTREAM-PAIR-USER-SUBMITTED) has no public score in the inspected files. Not found: its public score.

## Recipe parameters (0.6674 student, seeds 5 and 6)

Command template: `CI:research/iterations/2026-10-05-wave18/final_student_lane.sh`, with `--train-source train+valid --include-heldout-valid --full-valid --eval-source holdout`.

| parameter | value | source ref:path |
|---|---|---|
| Public score | 0.6674 (submission 26n9pg4y, 24m57s, 640 MB) | `CI:research/CURRENT_STATE.md` L197-199 |
| Served members | 2 GRU members, coef 0.5 each, equal weights, no other models blended | `CI:research/iterations/2026-10-05-wave20/distilled_h256_pair/members.json` |
| Member architecture | GRU, hidden 256, 2 layers, dropout 0.0 (argparse default, not overridden) | `CI:research/iterations/2026-10-05-wave18/final_student_lane.sh`; `CI:research/iterations/2026-10-05-wave6/scripts/train_rnn.py` (argparse) |
| Student epochs | 2 (saved `model_ep1`, final epoch) | `CI:.../2026-10-05-wave18/final_student_lane.sh` |
| Parent init | `fbc1k_s{5,6}@ep3` (4th epoch of parent, warm start, not from scratch) | `CI:.../2026-10-05-wave18/final_student_lane.sh` |
| Student learning rate | 1e-4 peak, OneCycleLR | `CI:.../final_student_lane.sh`; `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (OneCycleLR) |
| Warmup / decay shape | OneCycle, warmup fraction 0.05 (default), final_div_factor 100 | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (`--warmup-frac`, OneCycleLR call) |
| Optimizer | AdamW (default; no `--optimizer muon` in student command), betas (0.9, beta2=0.999 default) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` |
| Weight decay | 0.01 (default) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (`--wd`) |
| Gradient clip | 1.0 (default) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (`--clip`) |
| Batch size (student) | 16 | `CI:.../final_student_lane.sh` |
| Chunk length / truncated BPTT | chunk 2000 rows; hidden state carried across chunks within a 20,000-row sequence, detached at each chunk boundary; state reset only if `--state-reset` (default 0) | `CI:.../final_student_lane.sh`; `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (chunk loop, `detach_state`) |
| Parent chunk / batch | Parent `fbc1k` chunk 1000 per notes; parent batch not stated in notes (argparse default 32 used by `final_mu.sh` template). fbc1k exact command: not found | `CI:research/iterations/2026-10-02-slow-fast/README.md` L203, L218; `CI:.../2026-10-02-slow-fast/final_mu.sh` |
| Loss | `wmse` (weighted MSE, weights from abs clipped target x propensity x valid mask) | `CI:.../final_student_lane.sh` |
| Propensity weighting | `prop_f0`, power 2, eps 0.03 | `CI:.../final_student_lane.sh` |
| Valid true-mask weight | 0.03 | `CI:.../final_student_lane.sh` |
| Auxiliary heads | aux-struct 0.3, aux-next 0.3 | `CI:.../final_student_lane.sh` |
| Distillation target | target = (1 - alpha) * label + alpha * teacher, clipped to [-2, 2]; alpha 0.7 | `CI:.../final_student_lane.sh`; `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (`tgt = (1 - alpha) * label + alpha * teach`) |
| Teacher cache | `w18final_teacher` (see teacher table below) | `CI:.../final_student_lane.sh`; `CI:.../2026-10-05-wave18/prepare_full_teachers.py` |
| Family teachers (aux family head, weight 0.1) | `w18final_fbc1k_s5`, `w18final_family_full`, `w18final_family_raw`, `orgbase` | `CI:.../final_student_lane.sh`; train_rnn.py family loss term |
| EMA weights | none in the student command (no `--ema-teacher` or `--eval-ema`) | `CI:.../final_student_lane.sh` |
| SWA | none (`--swa-frac` default 0) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` |
| Augmentation | none in the student command. Other flags exist (`--level-aug`, `--level-drop`, `--i1swap`) but are unused here | `CI:.../2026-10-05-wave6/scripts/train_rnn.py`; `CI:.../final_student_lane.sh` |
| `--fast` flag | sets sync-free, fused Adam, pinned memory and propensity cache (performance only, per code) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` |
| Seeds | 5 and 6 (`w18final_distilled_s5@ep1`, `s6@ep1`) | `CI:.../2026-10-05-wave20/distilled_h256_pair/members.json` |
| Serving | native C++ v3 path, int8 per the manifest validation string; 2 members | `CI:.../2026-10-05-wave20/candidate-manifest.json` |
| Validation of the members (not holdout for final fits) | fold0 seed3/4 pair 0.71815029; native int8 0.71806187 on 425 sequences; independent composite replication +0.00174 valid, +0.00168 train diagnostic vs own teacher | `CI:.../2026-10-05-wave20/candidate-manifest.json` |

Teacher `w18final_teacher` (cache = weighted sum of member predictions, `CI:.../2026-10-05-wave18/prepare_full_teachers.py`, `CI:.../full-teacher-manifest.json`):

| component | weight | epoch used for predictions | source ref:path |
|---|---|---|---|
| `w18final_fbc1k_s5` (strong parent, Muon x3 chunk 1000 boot8 per notes) | 0.3 | ep3 | `CI:.../2026-10-05-wave18/full_teacher_lane.sh` (strong: epoch 3) |
| `w18final_ffull_s3`, `w18final_ffull_s4` | 0.1 each | ep1 | `CI:.../full_teacher_lane.sh` (full: epoch 1); training command: not found |
| `w18final_frawW_s3ft`, `w18final_frawW_s4ft` | 0.125 each | ep1 | `CI:.../full_teacher_lane.sh` (raw: epoch 1); training command: not found (closest analog `CI:research/iterations/2026-10-02-slow-fast/final_rawW.sh`, Muon x6) |
| `orgbase` | 0.0625 | not stated | `CI:.../prepare_full_teachers.py` |

Observation to verify, not established: the teacher weights sum to 0.8125, not 1.0, and the script does not normalize them (`CI:.../prepare_full_teachers.py`). If unintended, the distillation target is shrunk toward zero by about 19 percent. The same weights are reused in wave21 (`CI:research/iterations/2026-10-05-wave21/prepare_teachers.py`).

Parent template (`fbc1k`, Muon family; notes give the description, the exact fbc1k command is not found):

| parameter | value | source ref:path |
|---|---|---|
| Model | GRU hidden 256, 2 layers, chunk 1000 (notes) or 2000 (template) | `CI:.../2026-10-02-slow-fast/README.md` L203; `CI:.../final_mu.sh` |
| Muon | Muon on hidden 2-D matrices (GRU weights, hidden head layer); AdamW on input projection, output head layer, mean/std and biases. Momentum 0.95. Muon lr = base lr x muon-lr-mult; mult 3 (fbc1k per notes) or 6 (template) | `CI:.../2026-10-05-wave6/scripts/train_rnn.py` (Muon split and `--muon-lr-mult`, `--muon-momentum`); `CI:.../2026-10-02-slow-fast/README.md` L203; `CI:.../final_mu.sh` |
| Base lr / epochs | base lr 1e-3 (default); 4 epochs (ep3 used) | `CI:.../final_mu.sh` (`--epochs 4`); train_rnn.py default |
| EMA | decay 0.999, `--ema-teacher 1.0`, `--eval-ema` | `CI:.../final_mu.sh` |
| Since-weighting | `--since-weight 32 --since-floor 0.25` | `CI:.../final_mu.sh` |
| Stopped bootstrap distillation | `--distill-boot-h 8` (Vlad recipe per notes) | `CI:.../2026-10-02-slow-fast/README.md` L174-175, L203 |
| Head fine-tune (V1-style, not used in 0.6674 student) | 2 epochs, lr 5e-4, gcorr loss, freeze inp,gru | `CI:.../final_mu.sh` (H2 stage) |
| Weight decay | 0.01 | train_rnn.py default; `CI:.../final_mu.sh` |

## Evidence on more epochs (numbers as recorded in the notes)

| claim | numbers | source ref:path |
|---|---|---|
| 12-epoch single fold-0 run overfits at wd 0.01 | best 0.6734, end 0.6462; wd 0.1 stayed about 0.674-0.676 | `CI:research/ideas/2026-09-27_regularization.md` L9; `CI:research/game_changers.md` L32 (E-153) |
| Six-epoch vs four-epoch EMA+since single model | +0.0016 pooled single-model WP | `CI:research/ideas/2026-09-27_synthesis.md` L36; `CI:research/2026-09-27-final.md` L11 |
| Seed-3 six-epoch in pure-chain pair | +0.001208 +/- 0.000512 (lambda 0.02, 831 sequences) | `CI:research/2026-09-27-final.md` L11; `CI:research/2026-09-27-four-hour-run.md` L41 |
| Four-epoch EMA transfer to public | +0.0017 public | `CI:research/2026-09-27-final.md` L11 |
| Averaging late checkpoints (epochs 3/4/5) | -0.0001 to -0.0002 vs fixed epoch 5; "extra training helped; averaging did not" | `CI:research/ideas/2026-09-27_synthesis.md` L54 (E-196) |
| Extra validation repetition (more steps, about 12 percent) | fold0 +0.0020 vs parent; pure-chain pair lift +0.000406 +/- 0.000929, failed gate | `CI:research/2026-09-27-four-hour-run.md` L70 (E-220) |
| Early two-epoch vs longer | "Two epochs was the only clear gain (+0.004)"; 3 epochs 0.6713 vs 2-epoch 0.6707 (saturated) | `CI:research/brief-2026-09-26.md` L8-9, L37 |
| Muon chunk 1000 doubles optimizer steps | Muon x6 seed-3 0.6750 (collapse); Muon x3 0.6933 (fold-level, seed 3 and 4 avg +0.0005) | `CI:research/iterations/2026-10-02-slow-fast/README.md` L175, L203-204 |
| Parent-to-student continuation | parent 4 epochs (ep3), student +2 epochs at lr 1e-4 (total about 6 epochs of training, not counting the teacher) | `CI:.../2026-10-05-wave18/final_student_lane.sh` |
| Public effect of 2-epoch head FT on V1 | 0.6640 to 0.6641 (+0.0001) | `AR:snapshot/2026-10-01/SOTA_STATE.md` |

Reading: the notes support a fixed six-epoch EMA+since recipe over four epochs, and a warm 2-epoch distillation continuation after a 4-epoch parent. They do not support unbounded epochs: the 12-epoch run degraded without strong weight decay, and validation-repeat-induced extra steps failed the pure-chain gate.

## Not found

- Exact command for `fbc1k_s5/s6` (parent with Muon x3, chunk 1000, boot 8): not found. Only the README description was located.
- Exact commands for `ffull_s3/s4` and `frawW_s3ft/s4ft` (teacher members): not found.
- Parent batch size for `fbc1k`: not found (default 32 assumed only for the template).
- Learning-rate schedule for the `ffull`/`frawW` teachers beyond OneCycle default: not found.
- Number of served models for the most recent (wave46 four-seed) candidate: not scored publicly in inspected files.
- Any explicit "Maxim" recipe distinct from the Vlad parent: not found in the mined refs as a separate config.
