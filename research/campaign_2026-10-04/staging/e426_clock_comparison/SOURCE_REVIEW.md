# E426 fixed epoch-3 export and score review

This source review confirms the proposed inference path but does not approve or
launch any worker.

`research/campaign_2026-10-04/staging/e426/scripts/train_rnn.py` routes
`--eval-ema` snapshots through `checkpoint_model(..., evaluation_ema_model=...)`
at each completed epoch. For the non-`shared` clocks, teacher and evaluation
EMA are separate modules; `--fast` constructs each as `FastEMA`. The ordinary
checkpoint `model` is serialized from that selected inference module, while
the epoch-boundary `training_state.pt` separately stores `evaluation_ema.state`.
The new pre-score gate requires these states to be tensor-equal, so a clock's
teacher EMA cannot be mistaken for the inference EMA.

The root `scripts/predict_run.py` reconstructs with the checkpoint's `args`,
`n_in`, and saved `head.2.weight` output width; it loads the complete state,
restores saved preprocessing/global inference settings, and forces FP32 under
`--inference-precision fp32`. Its prediction loop initializes recurrent state
to `None` per batch and carries it only across successive chunks of each
sequence. It writes all VALID rows and their sidecar, including checkpoint and
prediction digests and the full row-group list. The staged validator then
checks 23 outputs in each source checkpoint, canonical integer IDs, exact
checkpoint/sidecar/prediction linkage, and FP32 output shape.

The selected score path is the existing `screen_score.py` plus the frozen
`campaign_score.py`. The screen contains only fold 0; it therefore scores the
manifest's evaluation groups, not final holdout rows. The frozen scorer applies
the dense-sequence mask `row_index >= 99 AND is_scored`, exact global WP,
connected-component paired bootstrap, and class-stratified summaries. The
screen registers all three requested comparisons in one invocation. The score
wrapper rejects incomplete or altered exports before calling that frozen
scorer and checks that all four evaluation classes appear in the report.

The training provenance currently observed for constant and aligned records
the same 211-feature normalization digest, code-source-closure digest,
fit-membership identity, split manifest, 4-epoch horizon, and no-evaluation
fixed epoch-3 policy. The validator checks all training arguments match after
removing only run name and `teacher_ema_clock`. It also checks the authoritative
runner fit record is successful and its checkpoint/provenance digests match.
The three exporters, screen, and validator use the same `e426_clock_<clock>_fp32`
cache names. A stateless test checks this join and negative/missing training
state artifact hashes before the score job can be admitted. The validator now
requires the attempt record to bind the checkpoint, provenance, and
`training_state.pt` bytes.

The fold-0 evaluation panel is component-disjoint from these fits, but it is not
untouched: the teacher checkpoint lineage and the score panel have prior
campaign exposure. Interpret this only as a paired exploratory endpoint for
the registered teacher-clock contrast, not as a fresh generalization result.

Reviewed source hashes:

| Source | SHA-256 |
| --- | --- |
| `scripts/predict_run.py` | `de4da30aebba33cb72a535b516839e4e627d00f6d6943cb4b2acbca7440b7fee` |
| `scripts/train_rnn.py` | `672a31dfcd92fd5e1a8c6e1ce88368735903f6131f737527ca9f60fee5fc4575` |
| E426 staged trainer | `f5bb73da64b3d182de06757e5986ab54a3b8277cb1b2e46cd239471050fdc4a1` |
| E426 event-clock module | `46644386d35a8aac27f4eeb972bb185c26262eba1624aa8a62baa25019b11d6a` |
| `scripts/screen_score.py` | `6bbd064d30e9ab608f24386c8bb076f0356be028c02092b629ab9a342051d34e` |
| frozen `campaign_score.py` | `b6377a0e1d67030115c6d20a64ee00dc97ef15eaad7a806408a76ef70558ab39` |
| fold-0 manifest | `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d` |
| staged score validator after cache/provenance fix | `9f20576360ba7760ab9fa350a1413d1557f803bd1ec718fad45c934d32b81907` |
| stateless packet schema/provenance test | `dd0f4731087d9e588b528f0f7fc66efbd3d10089aaade2f8d6b4269ac649a525` |

The export drafts use three distinct cache names and never overwrite existing
prediction artifacts. Each allows 900 seconds at 8 CPUs, 24-GiB host RAM,
4.2-GB allocator / 4.8-GB stop. The paired score draft allows 900 seconds at
1 CPU and 2 GiB RAM with no GPU. These are path-only drafts; root still needs
to bind them to terminal fit artifacts, finish the timing-window gate, and
seal/admit them.
