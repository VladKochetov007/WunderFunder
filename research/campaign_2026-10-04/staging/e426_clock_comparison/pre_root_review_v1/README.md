# E426 teacher-clock endpoint: fixed epoch-3 comparison

This packet stages predictions and a single paired screen for the already
registered fold-0 constant, aligned, and shuffled teacher-clock fits. It does
not fit models, choose epochs, or inspect validation scores to select an arm.
Each export reads only the predeclared `model_ep3.pt`, requests common FP32
inference, and writes all 1,873 VALID predictions. The E426 trainer records the
evaluation EMA in the checkpoint's ordinary inference `model`; the teacher EMA
clock affects training targets only. `checkpoint_model(..., use_ema=True)`
selects `evaluation_ema_model`, and the final checkpoint is written after each
full epoch. The validator checks this against the epoch-boundary training state
before scoring.

The paired endpoint uses the frozen `campaign_score.py` arithmetic on fold 0's
468 component-disjoint VALID evaluation groups, with full 20,000-row timelines,
the official `row_index >= 99 AND is_scored` mask, global clipped WP, and 2,000
paired component bootstrap replicates (seed 271828). The three registered
contrasts are aligned minus constant, aligned minus shuffled, and shuffled
minus constant (diagnostic). Per-class WP is reported as a guard against
aggregate masking; no class or epoch is used to select a result. No final
368-sequence holdout is included.

Exports and the score remain drafts. Root must first confirm the exact fit
attempts are terminal-successful, then separately seal/admit three serial
900-second GPU exports and the 900-second, 1-CPU/2-GiB CPU score job. No job was
launched while staging this packet.

`predict_run.py` uses the frozen standard GRU factory, loads checkpoint
normalization buffers and preprocessing flags, resets hidden state per
sequence through `predict`, and writes only the first two outputs for the
official metric. The pre-score validator requires all 23 model outputs to be
present in the checkpoint, confirms equality with the saved evaluation EMA,
and checks that the three fits share source closure, fit membership, data split,
normalization, and every training argument except the intended clock and run
name.
