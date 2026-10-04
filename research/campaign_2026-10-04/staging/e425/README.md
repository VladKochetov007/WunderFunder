# E425 / H106 soft prefix mixture staging

This isolated overlay tests whether a learned first-99-row gate benefits from
mixing every specialist's output during training, compared with H093's
straight-through hard top-one gate. It never changes the active root
`src/wnn/specialist.py` or `scripts/train_rnn.py`.

`src/wnn/specialist.py` computes the same causal running prefix and the same
shared and four expert GRUs for all arms. In `soft` mode, each row's
prediction is the base prediction plus the sum of all four expert predictions
weighted by the learned router softmax. The router sees only normalized input
rows through the current time; its context stops changing at row 99. Every
expert executes and receives an output gradient on every row. `learned`,
`hash`, and `universal` retain their existing forward paths. The soft route
adds no trainable tensors or state-dict keys.

The staged `scripts/train_rnn.py` preserves the ordinary combo objective and
optimizer loop. For soft-route fits only, it records route entropy, each
expert's pre-step gradient norm and nonzero-gradient count, and each expert's
measured post-step parameter delta and nonzero-update count. These are
diagnostics, not screening rules. `scripts/run_specialist_trial.py` prioritizes
the staged overlay and attaches hashes for all staged overrides and root WNN
dependencies to provenance. Use this entrypoint for any future E425 fit; do
not invoke the root trainer for the new route.

## Prepared verification

`tests/test_e425_soft_route.py` covers the exact 154,343-parameter budget and
identical state topology, bitwise forward/state equality between the staged
`learned` path and frozen root code under identical weights, causal and
post-warmup-stable gate probabilities, chunk and row-stream continuation,
fresh-state reset, direct gradients and measured updates for all four experts,
and equality with universal averaging under uniform router logits.

`tests/test_e425_trainer_cpu.py` drives the actual staged trainer through a
one-epoch CPU fixture with 200-row sequences and tiny hidden sizes. It retains
Muon, the training loop, EMA paths, and the staged diagnostics while replacing
only data/split providers. It expects four optimizer attempts, 800 routed
rows, and nonzero gradient and parameter movement for every expert. These
tests are prepared but have not been imported or run in this staging task;
the root runner should execute them under the agreed one-CPU/two-GiB cap.

## Matched controls and reuse boundary

The available E424 H093 v2 fits use the same seed-3 recipe and identical
training/evaluation memberships, normalization, propensity checkpoint, and
teacher arguments across learned, universal, and GRU96 controls. The learned
and universal specialists each have 154,343 parameters; the GRU96 control
has 152,855. Their fixed-epoch-3 checkpoint SHA-256 values are:

| Control | Run | Checkpoint SHA-256 |
| --- | --- | --- |
| learned hard gate | `c20261004_h093_learned_s3_final` | `4dca229567c1eef24ad4b185fabea1484f7723bad975adba692fe12c40be433b` |
| universal average | `c20261004_h093_universal_s3_final` | `8ef0f57c2ca443ceca20dfa1a111752c178469217cb64696219eb163a4de8ab3` |
| ordinary GRU96 x 2 | `c20261004_h093_gru_budget_s3_final` | `43af87e1c335557d5c4ccfafb9c55a985ad9338becb113dcc9419d446cf27578` |

These controls use the public-parent combo lineage: teachers
`ab67_final_since_s3` and `ab67_final_since_s4`, plus `prop_v1`, the full
TRAIN+VALID pool, Muon, chunk1000, stopped bootstrap8, alpha0.5, and four
epochs. Root verified this against
`artifacts/runs/c20261004_combo_s3_final_r1/provenance.json`. They are matched
to one another and share the intended full-data public parent lineage;
fold-specific `c20260929` teachers are counterparts, not newer replacements.
Reuse still requires exact fit-row membership, cache, prediction, and source
identity checks. The H093 learned v3 fixed-epoch prediction exists at SHA
`ea295807f5c8e01a37eff501c37f4552ecf2c6a9127957eaca78c6246eabb337`, and the
GRU96 prediction exists at SHA
`2a8b5a97699a3188250d5482c2d5630396df780b13e5a36e46511880119e74b7`. No
matching universal prediction artifact was present when this packet was
prepared. The H106 preregistration requires matching prediction provenance
before control reuse; that check remains open.

## Not established here

No model fit or GPU launch is part of this staging. Native production export,
the repository inference/prediction wrapper accepting a soft route, and
end-to-end prediction parity for a soft checkpoint remain unverified. The
current E424 recurrent prediction wrapper covers hard specialist and adapter
families, but its actual saved-model compatibility must be tested with a soft
checkpoint before claiming native export. Root source hashes must remain
unchanged until the active E424/E426 workers are finished.
