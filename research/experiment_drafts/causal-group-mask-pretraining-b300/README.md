# Causal group-masked pretraining for a GRU teacher

**Status:** [draft PR #21](https://github.com/VladKochetov007/WunderFunder/pull/21) is open against `research-team`; its initial published source commit is `642bfad821f83658061615a1f3ac3e2df45fb68e`. This is a proposed B300 experiment with an executable feature-only pretrainer and checkpoint converter. No B300 profile, fit, score, or submission has run.

The idea is to pretrain a 512-wide causal GRU to reconstruct one masked raw input group at a time using the other current inputs and past recurrent state. Each selected group is hidden over a 32-row block. Mask indicators are appended explicitly. At pretraining, canonical event features are rebuilt from the masked raw sequence so they cannot reveal the hidden current group. The prediction at row t never reads row t+1, targets, or scoring masks.

The paired task control and candidate use the same fitting-only normalization and initial task readout. The candidate copies its raw/event input projection and GRU from pretraining, drops the mask-indicator columns, and fine-tunes on full inputs. An 8-epoch cold control matches two additional supervised sequence passes; that is not a FLOP match because the masked pretraining model has a wider input and a reconstruction head.

## Files

- [PR.md](PR.md) contains the numbered research-team handoff, all 30 fold/seed cells per arm, official evaluation and interpretation limits.
- [Independent static review](../../unified/reviews/CAUSAL_GROUP_MASK_PRETRAIN_STATIC_REVIEW_20261018_R1.md) records source-hash and resume checks, the final pass after fixing shared-stat races and full-recipe validation, and what the review did not execute.
- [recipe.json](recipe.json) freezes hypotheses, arms, folds, schedules, masking, controls and artifact requirements.
- [pretrain_masked_groups.py](../../../scripts/pretrain_masked_groups.py) prepares fold-clean input statistics and trains the pretrainer or profiles a bounded number of real-shape updates.
- [prepare_masked_group_initializers.py](../../../scripts/prepare_masked_group_initializers.py) writes matched standard trainer checkpoints for cold and pretrained task fits.
- [masked_group_pretraining.py](../../../src/wnn/training/masked_group_pretraining.py) owns feature-only loading, the 13 group ranges, masked input construction and backbone transfer.
- [masked_pretraining_epoch.py](../../../src/wnn/training/masked_pretraining_epoch.py) owns one data pass, TBPTT chunk updates and per-update receipts.
- [masked_pretraining_loop.py](../../../src/wnn/training/masked_pretraining_loop.py) owns strict-FP32 training updates, gradient/LR logging, epoch checkpoints and resume.
- [masked_pretraining_statistics.py](../../../src/wnn/training/masked_pretraining_statistics.py) owns the fixed 64-sequence fold-clean statistics and hash verification.
- [masked_pretraining_reporting.py](../../../src/wnn/training/masked_pretraining_reporting.py) owns atomic resume checkpoints and update/profile receipts.
- [Source note](../../unified/sources/CAUSAL_GROUP_MASK_PRETRAIN_SOURCES_20261018_R1.md) reviews the primary literature and its transfer limits.
- [Idea card](../../unified/ideas/CAUSAL_GROUP_MASK_PRETRAIN_20261018_R1.md) records the mechanism, falsifiers and allocation rationale.

The scripts read the source-qualified fold manifest, use SequenceFile.read with targets=False and mask=False, and keep all statistics, initialization and fitting inside each fold's component-excluded membership. For endpoint grading, mount the approved `cache/valid_targets.npz` beside the dataset root; the fitter never reads it, and the existing official grader uses it after prediction export. The pair runner locks shared per-fold statistics so same-fold seed cells cannot race, and initializer conversion rejects any cached pretraining checkpoint whose complete model or optimizer recipe differs from this packet. Start with `bash run_pair.sh profile 0 1001` on one actual B300, then review its run log and memory receipt before authorizing quality cells. `bash run_pair.sh pair FOLD SEED` runs one paired C0/T1/C1 cell. The quality recipe has 90 supervised fits plus 30 pretraining fits; its wall time is unknown until the profile and a complete first cell are measured. The current handoff does not yet export/grade all endpoints or aggregate the 30-cell result, so those steps must be bound to the existing campaign evaluator before a quality claim.
