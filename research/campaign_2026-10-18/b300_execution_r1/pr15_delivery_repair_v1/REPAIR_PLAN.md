# PR15 checkpoint delivery repair

The first four baseline fits completed their full three-epoch schedules. Their
old worker then failed before export because two sealed validators compared the
raw Torch archive hashes of `model_ep2.pt` and `model.pt`. The pinned trainer
`src/wnn/training/outputs.py::save_epoch` calls `torch.save` separately for
each name from the same checkpoint object. The B300 CPU-load check found exact
recursive equality of both payloads, including all model tensors and metadata,
while their archive bytes differed. This is a serialization-container false
positive; it does not call for another fit.

The repair is isolated in this directory. It leaves `pr15_host_v1` untouched.
The original freeze remains bound by `FROZEN.json` SHA-256
`c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b66510`,
`FIT_CONTRACT.json` SHA-256
`05aa087af1fdd68ac97146e6f56750157d4a33ebd4fce2a5ec2f5eb30cc6bcb8`, and
`fit_validation.py` SHA-256
`3359c4073d5b88b540acb713a9bdbb947673eacd67c050c59f7e54372bb56d8a`.
The immutable packet must continue to reject its raw-byte comparison; a future
corrected worker requires its own new packet and freeze manifest.

`checkpoint_identity.py` hashes the full loaded payload: mapping order and
metadata, container types, tensor dtype, shape, stride, storage offset, and
logical tensor bytes. Unsupported or object-dtype values fail closed. It also
records both original raw file hashes independently and checks each against the
trainer's `provenance.json`. Predictions remain pinned to the raw
`model_ep2.pt` hash. Thus payload identity resolves the archive encoding issue
without weakening source, recipe, fit-budget, or evaluation pins.

`delivery_resume.validate_completed_fit` verifies the original request and
canonical train/predict argv, all frozen packet member hashes, source tree,
scorer copies, fold identity, finite FP32 H512x1 state, recipe, source map,
three evaluation-withheld epochs, and exact per-fold optimizer totals. It
loads the separately frozen `host_v2` validator, whose `_check_checkpoint`
uses full-payload identity for the selected/fixed pair while retaining both
actual archive-byte hashes and checking each against trainer provenance. The
repair never changes or aliases the raw SHA-256 function. It hashes and
preserves the old failure receipt. Because the old worker never persisted its
successful outer phase record, the new fit recovery evidence records the
existing `actor_returned` envelope and fit-log hash, and leaves the outer exit
code and phase timestamps null.

The exact optimizer-attempt totals by fold are 44,760 / 44,790 / 44,850 /
44,640 for F0 / F1 / F2 / F3. Validation checks the existing full-run history
and checkpoint provenance against these counts; a 100-update shape profile
cannot satisfy a completed-fit check.

The function returns a validation object for the root-owned runner to persist
as a distinct recovery receipt. It never launches a process and never writes a
receipt. `delivery_commands` constructs only export and CPU-score argv; the
caller must use the original `_run_gpu_actor` and `_run_score` wrappers. The
export actor environment must prepend the verified source's `scripts` folder
to `PYTHONPATH`: the sealed `export_cell.py` adds `source/src`, while
`predict_run.py` imports `train_rnn` from `source/scripts`. A first export-only
attempt without that path failed at import, before producing predictions. Keep
that failed phase envelope/log, and use a new run directory and separate
recovery receipt for the retry. Do not call either trainer entrypoint.

After export and source-native grading, `validate_exported_delivery` rehashes
the checkpoint pair and provenance, checks the evaluation-only prediction
sidecar against fold membership and the fixed checkpoint's raw hash, checks
prediction bytes/shape/dtype, and invokes the unchanged PR15 validation of
`score.json`. Persist its result separately from the original worker receipt.
The source-native metric remains `global_wp_clip2_v1`.

The corrected future-worker packet is frozen separately at `host_v2/`. Its
fit validator uses `verify_checkpoint_pair`, returns both actual archive hashes
plus full-payload identity, and retains the existing model, provenance,
trainer-source, exact update-count, prediction, and native score checks. Its
export wrapper adds the pinned source's `scripts` directory to `sys.path` so
the unchanged `predict_run.py` can import `train_rnn`. The packet's
`FROZEN.json` and `EXECUTION_MANIFEST.json` bind this change while
`PARENT_PACKET.json` preserves the original v1 hashes. The v1 files and
original failure receipt remain unchanged.

The CPU identity suite is in `host_v2/test_checkpoint_identity.py`. This
Windows workspace has no Torch install, so those ten tests skip locally; the
same suite passed under the deployed CPU Torch environment before the repair
adapter was accepted. Local Ruff, Python compilation, v2 export-path and
contract tests pass. The local review snapshot omits the sibling
`b300-r01-v1/control` files referenced by the original v1 packet; the v2
packet verifies its complete copied file set and parent hash chain locally,
while completed-fit validation still runs against the deployed runtime tree.
