# Local GPU research log, from 2026-11-14: cold one-stage distillation × since-weighting on a 40-minute GRU cell

**Precision correction and prospective panel amendment.** Independent source review plus saved optimizer groups found that the PR49 legacy Muon adapter used BF16 Newton–Schulz despite FP32 model kernels. Historical scores and ZIP checks remain actual observations, with corrected precision identity. The incomplete alpha-.9 seed4 fit and old automatic final-refit path were stopped. Corrected source [`48776fba8`](https://github.com/VladKochetov007/WunderFunder/commit/48776fba8af26b16b7e6813def7ccf571f56c1ac) passes both factory-branch CUDA checks and a 120-update full-recipe profile. The revised screen is six cloud fits (alpha .5/.9/.9+boot8 × seeds3/4, shared exact controls), plus a local alpha .9/1.0 seed3 pair using the reserve. Final refits remain conditional on new corrected-source comparisons; mixed-precision controls are never substituted. Interrupted cost remains charged inside the original local/cloud time caps. The older panel wording below is superseded by this dated amendment.

**Current follow-up:** strict local KD90 s3 is complete at .6915443853; pure-teacher s3 is running. The six strict cloud fits are queued. This branch now contains the two-branch Muon precision correction; optimizer-switch conversion remains out of scope. [Current PR description](PR.md) records the effective panel, checks and conditional release rules. Historical 18-fit results below are preserved with the corrected legacy precision label.

**Status: all 18 fits completed; submission collection and checks on November 14.** Owner: Vlad / Claude Code; collected by Codex. A live log of local-GPU experiments (one RTX 3060 Laptop,
6 GB) with all ideas, triage, code and per-epoch results. Updated as cells finish.
**Results table: [RESULTS.md](RESULTS.md)** (generated from the fit logs in this folder).

## Final collection (2026-11-14)

All 18 fitted endpoints are saved (16 train-only, two final train+valid). The GPU fitting queue is finished. [Submission handoff](SUBMISSIONS_R1.md), [checks and manifests](submission_checks/README.md), and [review status and limits](SUBMISSION_AUDIT_R1.md) own the current release status. No new fitting was needed for collection.

**Correction to the preliminary synergy claim below:** the since×KD interaction is +0.001587 on seed 3 and −0.001786 on seed 4 (mean −0.000100). Synergy did not replicate. KD on top of since32 gains +0.003756/+0.002299, mean +0.003028. Two close seed scores alone do not establish removal of training variability.

The fixed equal-weight train-only kd50 s3/s4 pair scores **0.691423** on all official valid, +0.000949 over the mean single. Alpha 0.9 is the best single screen endpoint at **0.691442** (one seed). Row-dependent alpha scores 0.691117; longer training, no auxiliaries, a second distillation round, H384 and since8 trail these endpoints. The 50%/75% data runs end at 0.621511/0.662904 after severe within-run regression; they do not isolate a clean data-scaling law.

The user reports the train-only kd50 single at **public 0.66108 / 15m27s / 628 MB**, ID `1qn2bx0q`. This is transfer feedback, not a numerical forecast for the final-data pair.

## Why this exists

Iteration speed. One cold fit of the recipe below takes about 40 minutes and reaches 0.687 on full valid, so
a factorial that needs an L4 wave can be answered locally in an evening. The first question is PR #92 B03:
do since-weighting and distillation add up or substitute?

## Recipe (frozen; `recipe_args.sh` BASE + PARENT, launched by `run_cell.sh`)

GRU H256×2 (0.98M parameters), canonical 211 derived inputs, Muon on recurrent matrices + AdamW elsewhere, LR 1e-3, onecycle
with 5% warmup, batch 32 × chunk 1000, 4 epochs, fixed last epoch, WMSE, `--since-weight 32 --since-floor .25`,
auxiliary next/struct heads 0.3, strict IEEE FP32 (`strict_launch.py`, TF32 off). Trainer source: PR49 legacy
runtime `188c0788e`. Fit on the 10,325 train-only sequences (local split `6d23a1b7…`, 282 linked sequences excluded),
evaluate all 1,873 valid sequences / 3,528,273 scored rows with `global_wp_clip2_v1`. Teacher for distillation:
our local reproduction of the PR49 `fbc1k_s5` teacher (valid 0.69142), feature-only train predictions;
target = (1−α)·label + α·teacher.

Differences from Semyon's L4 contract: Muon at LR 1e-3 instead of AdamW 3e-4, 4 epochs instead of 6, no K16
stem, auxiliary heads on. Numbers here are not directly comparable with PR #95/#96 cells; paired deltas are.

## Early seed-3 snapshot (superseded by final collection above)

Since × distillation 2×2, fixed epoch 4, full valid:

| | labels only | + distillation α 0.5 |
|---|---|---|
| no since | 0.68591 | 0.68808 |
| since32 | 0.68671 | **0.69047** |

- **The two do not substitute here; they reinforce.** Distillation alone +0.0022, since32 alone +0.0008,
  together +0.0046; interaction +0.0016. One seed: the interaction is about the size of seed noise, so the
  second-seed 2×2 (seed 4) is queued before anything is claimed. This is the opposite sign to our older
  E-351 (substitution, −0.0012), which used a warm-started child.
- **Seed 4 control** (`l14_ctrl_s4`, since32, labels): 0.68818 against 0.68671 at seed 3; seed spread on this
  recipe is about 0.0015. (Its queue entry says FAILED because `run_cell.sh` was edited in place while it
  ran; the fit itself completed and logged its fixed endpoint.)
- **Six epochs do not help the distilled recipe:** `l14_kd50_e6_s3` ends at 0.68887 (best epoch 0.68935)
  against 0.69047 for four epochs. Keep four.
- **The interaction builds up during training.** Per epoch, I = both − kd − since + labels is
  −0.0001, +0.0010, +0.0015, +0.0016: absent after one epoch, present from the second.
- **Cold one-stage distillation** beats our Oct 29 two-stage chain (parent → distilled child, 0.68997) in one
  40-minute fit and sits 0.001 below its teacher (0.69142).
- **Distillation removes the last-epoch regression** that label-only runs show.
- Zero-fit: the three local teachers' residuals correlate 0.995, so a three-teacher mean target is not run.
- Zero-fit (`blend_check.py`): the distilled student correlates 0.9956 with the served TL3C blend; adding it
  at weight 0.2 gives +0.0001 on valid, so no four-member ZIP is built.

## Early packaging snapshot (current handoff is SUBMISSIONS_R1.md)

| ZIP | Members | Valid | Checks |
|---|---|---|---|
| `submissions/local_20261114/kd50_s3_single.zip` (SHA `e5790b71…`, 4.6 MB) | `l14_kd50_s3@ep3`, train-only | Torch 0.69047 | team INT8 runtime; native vs Torch on a 64-sequence valid panel 0.66546 vs 0.66545 (RMS diff 0.6%); ABI, CRC, A-B-A reset pass; 0.42 s/sequence on the laptop |
| `submissions/local_20261114/final_kd50_s3_single.zip` (SHA `ba871f49…`, 4.6 MB) | `l14_final_kd50_s3@ep3`, same recipe on train + all valid (12,479 sequences) | none (valid is in the fit) | ABI, CRC, A-B-A reset pass; 0.41 s/sequence; predictions correlate 0.982 with the train-only ZIP on the 64-sequence panel, same scale (std ratio 0.99), all finite |

Built with the Oct 26 `package_tool.py` (`export`, `build`, `replay`) and `score_panel.py` here. A train+valid
refit of the same recipe (`l14_final_kd50_s3`, 12,479 sequences, launched by `final_launch.py`, which
disables the trainer's train-only guard for this sanctioned final fit; no held-out score exists for it) is
queued for a second ZIP. Reference pair for transfer:
tiedlead-s5 single, valid 0.69143, public 0.66246.

## Where the error sits (zero-fit, `band_calibration.py`, `l14_kd50_s3`, 400 valid sequences)

| Rows since the last i0 price event | Share of scored rows | Share of metric weight | WP |
|---|---|---|---|
| < 8 | 46% | 41% | 0.809 |
| 8–32 | 21% | 22% | 0.707 |
| 32–128 | 21% | 23% | 0.632 |
| ≥ 128 | 13% | 15% | 0.487 |

More than a third of the metric weight sits on rows 32+ rows after the last event, where WP is 0.49–0.63.
A per-band affine correction of the output does not help, so the gap there is information, not calibration.

## Operations

`run_queue.sh` runs `queue.txt` one cell at a time (a single fit already uses ~94% GPU, 3.0 GB VRAM); each
cell is a systemd scope with a hard memory cap and no swap. `thermal_guard.sh` freezes the fit at CPU ≥ 95 °C.
`make_results.py` regenerates RESULTS.md. Fit logs (`*.fit.log`) carry the full config, per-epoch gradient
statistics and valid scores. Weights and caches stay on the local machine.

## External ideation round 1 (Freebuff 0.2.x, DeepSeek V4.1 Flash, 2026-11-14)

Prompt: [FREEBUFF_TASK_R1.md](FREEBUFF_TASK_R1.md) (anonymized problem statement only). Output:
[FREEBUFF_IDEAS_R1.md](FREEBUFF_IDEAS_R1.md). Triage against the ledger:

| # | Idea | Status |
|---|---|---|
| 1 | Differentiable weighted-correlation loss | closed: H-018 refuted (E-023), gcorr fine-tunes tested |
| 2 | Ordinal/categorical head | closed: CE readout family H-046 and successors |
| 3 | Train on the unrounded event-clock mean | **new**; needs a zero-fit check of the quantization share of error first |
| 4 | Learned per-row noise scale (NLL) | tested: H-NLL-STRONG-GRU / H-LOSS25-FAITHFUL-NLL |
| 5 | Per-event increment heads | already in the recipe as `--aux-next` |
| 6 | Stateful TBPTT | already how the trainer chunks sequences |
| 7 | Density-ratio reweighting | refuted Sep 26 (n_eff ~ 1e-5) |
| 8 | Since-phase specialists | **new as since bands**; zero-fit diagnostic: valid WP by since decile (same as PR #93 C04) |

## External ideation round 2 (Freebuff, DeepSeek V4.1 Flash): distillation

Prompt [FREEBUFF_TASK_R2.md](FREEBUFF_TASK_R2.md), output [FREEBUFF_IDEAS_R2.md](FREEBUFF_IDEAS_R2.md).

| # | Idea | Status |
|---|---|---|
| 1 | Per-row mixing weight by label noise | trainer already has `--distill-alpha-far/-tau` (more teacher far from a label change); queued as `l14_kdfar_s3` (0.3 near, 0.8 far, tau 32) |
| 2 | Mean of three teachers | **killed zero-fit**: |y|-weighted residual correlations between the three teachers are 0.9952-0.9955 on 400 train sequences (`teacher_resid_diag.py`), so the mean target is no quieter than one teacher; cell removed from the queue |
| 3 | Cross-fitted residual gate (GBM on y - t) | open; needs the scored-row mask for train, which this cache does not carry |
| 4 | Second round with a label/teacher/student vote | open; needs a train export of the round-1 student |
| 5 | Teacher on rank-transported inputs | open; expensive, proxy-shift test first |
| 6 | Teacher fraction rising over epochs | open; needs a two-line trainer change |

Caveat on `teacher_resid_diag.py`: the train target cache has an empty mask array, so the diagnostic includes
unscored rows and its absolute WP (0.51) is not the scored-row WP. The residual correlations between
teachers do not depend on that.

## External ideation round 3 (Freebuff, DeepSeek V4.1 Flash): why the pair is super-additive, what to stack

Prompt [FREEBUFF_TASK_R3.md](FREEBUFF_TASK_R3.md), output [FREEBUFF_IDEAS_R3.md](FREEBUFF_IDEAS_R3.md).
It points out that two variance reductions on one target should substitute, so a positive interaction needs
another mechanism, and offers three: (E1) the teacher keeps the information of rows the age weight discards,
(E2) the pair leaves a residue of large-|y| teacher-disagreement rows, (E3) each knob delays a different
part of the over-fit so the last epoch stays productive.

| Item | Status |
|---|---|
| Per-epoch interaction (E3 diagnostic) | done from existing logs: grows from 0 to +0.0016, consistent with E3; but 6 epochs did not help, so it is not simply "more epochs" |
| Shared-seed check | all four cells share seed 3 (same init and data order), so the common seed effect cancels in I; seed 4 2×2 is queued for the across-seed read |
| B1 event-gated second layer | same as PR #93 C02; needs model code |
| B2 Muon | already in the recipe |
| B3 event-phase inputs | PR #92 B06 family; needs input code |
| B4 density-matched sequence sampling | trainer has `--class-balance`, but the train-only guard forbids it; skipped |
| B5 per-since-band affine readout | **killed zero-fit** (`band_calibration.py`, 400 valid sequences, 2-fold cross-fit): 0.70353 raw → 0.70324 by since band, 0.70316 by band × class, 0.70358 by class; the readout is already calibrated per band |
| B6 Huber on the mixed target | open; Ivan's PR #63 covers Huber on B300 |
| 12-run screen (six singles on the pair, then one A×B 2×2 at two seeds) | adopted in spirit: current queue = singles on the pair at seed 3 + the since×KD 2×2 at seed 4 |

## Next

The queue is complete. Deliver the checked ZIPs in SUBMISSIONS_R1.md for human upload. No additional experiment is launched by this collection.

[Quality update R3](QUALITY_UPDATE_R3.md): actual public receipts, completed strict pair/negative weight means and a hash-bound stronger-teacher target probe. New matched fitting is conditional; no new public or student gain is claimed.
