# Four teachers running; no quality result yet

PR: https://github.com/VladKochetov007/WunderFunder/pull/62.
Actual launch update:
https://github.com/VladKochetov007/WunderFunder/pull/62#issuecomment-6637653182.

The receiving authority runs 17:04:33–20:04:33 UTC on 4 November, including
45 minutes for collection. The new teacher fits use GPUs2–5, eight disjoint CPU
cores each, under detached owned-command controllers. The original two
Transformer teachers remain on GPUs6/7. No foreign process was changed.

| Arm | Seeds | Change from reused PR61 GRU256x2 controls |
|---|---|---|
| clip5 | 1001,1002 | Global clipping threshold 1 → 5 |
| aux_late | 1001,1002 | Keep existing .3 structural/.3 next coefficients through half of training, taper to zero on the final optimizer attempt; clip1 |

Both arms keep native211 inputs and all23 output rows, four epochs/fixedep3,
Muon on recurrent matrices, the original OneCycle recipe and strictFP32. Official
train10325/0valid fitting ancestry and fullvalid1873/3,528,273 scored rows are
unchanged. This validation is exposed development.

All four actual profiles passed: exit0, no timeout, 1360/1360 completed updates,
finite loss and complete preclip gradient checks, zero AMP skips/nonfinite
gradients, 26.21–26.24 sequences/s. Raw measured values are in
[ACTUAL_PROFILES_SUMMARY_R1.json](ACTUAL_PROFILES_SUMMARY_R1.json).
The measured train-only projection is `4 * 10325 / 26.21 ≈ 1576 seconds`;
35–40 minutes including validation, startup and checkpoints is an estimate.

The attempted no-update diagnostic failed before gradient measurements: its
1×T warmup mask was not expanded to B×T for structural-loss row weighting.
Original failed source/log/receipt remain in the runtime. No retry was run.
The frozen execution-failure branch selects aux_late from the prior hypothesis;
this failure provides no evidence of gradient conflict or model quality.
The working driver now expands the mask for a future identified attempt; the
frozen afc24 diagnostic snapshot and its failure remain unchanged. The corrected
driver was not run on the GPU in this allocation.

Diagnostic source is afc24ac1033215f7db2be94c03390aa59452748c. Fit source is
11b8de361b78d1650a811ccc8c138755bab7cc93. Independent source review found no
default-off numerical drift from the reused GRU controls. All launch identities
are joined in the actual receiver's `control/OBJECTIVE_PANEL_R1.json`,
`ADMISSION_R1.json`, `PROFILE_REVIEW_R1.json` and source/data bindings. The
[received panel copy](OBJECTIVE_PANEL_RECEIVED_R1.json) is JSON-compacted for
transport and is not the byte-identical remote panel hash authority.

Runtime: `/data2/lc-epanov_s/tabdata/runtime/l4-objective-20261104-r1`.
CPU-only monitor `control/objective_monitor_r1.py` forwards real train/valid and
gradient events into the existing dashboard and records receipt-owned process
resources plus GPU/free-disk telemetry. Training is detached from SSH. Actual
per-cell receipts, not the launcher's initial status, establish completion.

At most two W18 student fits remain conditional on the frozen teacher gate:
the best new two-seed fixedep3 mean must exceed the reused control mean by
at least .001. No teacher gain, student gain, public score or new submission is
claimed at this running stage.

[The terminal report collector](../../../scripts/l4_objective_report_20261104.py)
joins actual fit ownership, chronological provenance, history metrics and fixed
checkpoint hashes, then writes the comparison/curves and all-epoch teacher
weight archive. Teacher reporting does not replace a student/serving evaluation.
