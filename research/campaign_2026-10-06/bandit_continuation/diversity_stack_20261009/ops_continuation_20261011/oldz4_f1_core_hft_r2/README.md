# F1 core crop and head-only HFT successor

This packet prepares the already-reviewed F1 student CP1 for an exact 31-to-23
family-head crop, then a bounded actual-device HFT profile, a one-epoch
head-only HFT fit and a full-valid export. It is preparation only: every
mutating stage remains closed until its named root receipt is present and
matches the frozen contract. No stage in this packet has run.

The parent is `oldz4_h230_student_f1_s3_r4@ep1`, with checkpoint, provenance
and training-state bytes pinned in `SOURCE_INPUT_BINDINGS_R2.json`. Root's
actual consumer review is separately identified there. The crop preserves the
first 23 `head.0` rows and all other tensors and normalization buffers; it is
not a fit.

The HFT recipe matches the successful F0 HFT producer: fit only the 1,416
source-qualified, fold-eligible groups in `valid.parquet` with
`--train-source valid --full-valid --mask-weight`; exclude the 457 outer F1
evaluation groups. The 10,533 F1 train groups belong to the broader student
ancestry, not this HFT stage's direct fitting set. The exact frozen F0 argv
omits auxiliary CLI flags. The C337 initializer-preprocessing path inherits
effective `aux_struct=0.3` and `aux_next=0.3`; the export consumer checks these
values in actual F1 provenance. F1 was previously inspected and is robustness
evidence, not a fresh holdout.

## Validation and execution order

Run `validate-import` first with the zero-worker, single-thread bounded
environment in the contract command. It verifies the exact C337 tree, the
actual source-qualified split through the frozen loader, import roots bound to
MAIN and no device startup. The frozen snapshot's original manifest omits one
file that is present in its tree (`src/wnn/csrc/symreg_screen.cpp`); this packet
pins that exact extra path and digest separately and still requires exact
tree closure. It does not weaken the 52-file original manifest.

Then require a root CP1 terminal receipt and a separate root crop admission
before `--stage crop`. Verify the resulting write-once crop receipt before
`--stage validate-cli`; that command exercises the actual C337 parser up to
the device boundary without starting CUDA. `--stage profile` additionally
requires the profile-launch receipt. Its profile intentionally omits
`--save-training-state`, which C337 rejects together with profile-only early
exit flags. The full-fit argv retains `--save-training-state`.

The profile does not release a fit by itself. Root must interpret finite
actual-CUDA gradient/update evidence and resources, then issue separate
profile and fit admissions. After a successful fit, the export command
requires a root terminal gate and a separate export admission. It binds the
actual fit membership, CP/provenance/initializer and normalization, then
exports the complete `(1873, 20000, 2)` float32 valid array in canonical row
order, with sidecar and streamed finite checks. Output-level student/HFT
averaging would differ from the historical parameter-averaged `PARENT|HEAD`
mechanism; this packet does not define a blend.

The unsealed R1 draft is not part of this release; its two pre-freeze findings
are preserved in `R1_PRE_FREEZE_DIAGNOSIS_R2.json`. R2 corrects the source
snapshot's separately pinned extra file and the crop digest lookup at the
actual nested source-binding key. R2 ran only the separate temporary crop
fixture described above; no admitted crop, device profile, fit or export stage ran.

`tests/run_actual_cp1_crop_fixture_r2.py` runs the exact frozen F0 crop
producer and downstream verifier on the actual hash-bound F1 CP1 in a new
temporary project under `/tmp`. It writes only that temporary crop and a
fixture receipt in this packet; it does not use root admission or write to
MAIN artifacts. The receipt is consumer-compatibility evidence, not execution
authorization.
