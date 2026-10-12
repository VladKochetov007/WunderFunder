# Recipe correction R2

**Status: corrected packet prepared for independent review. No training, export, grade, or provider run has launched. Root still owns fit release and fresh resource admission.** The R1 source, command JSON, and bindings remain preserved unchanged as audit history; do not launch R1.

The independent recipe critic found two launch blockers in the R1 command packet:

1. It passed `WNN_GPU_MEMORY_BYTES=4300000000`. In the frozen R3 `resources.py`, that value becomes an explicit PyTorch per-process allocation ceiling.
2. P2/P3 supplied relative `--resume-state` paths. The root resource controller runs workers with the source overlay as cwd, while `train_rnn.py` opens the state directly as a `Path`; those paths would not resolve to the main project's artifacts.

R2 fixes both without changing the scientific treatment. Its immutable R3-derived source overlay applies the same target-amplitude trainer hunk and replaces only the operational `src/wnn/resources.py` with main commit `ac6a577f3f209ae70b8dd0830c9d90175d3c1b88` (SHA256 `0b6280130a480f5547552df83cc4a15e7cfe88ad04498ca11c2ab2d54bb428e9`). This reviewed policy derives the allocator fraction from live free VRAM and a 1,000,000,000-byte safety reserve. R2 pins that reserve, fraction `1.0`, and one CPU thread in the worker environment; it passes no explicit byte ceiling and uses no external 5-GB VRAM watchdog. The measured shape profile remains the same GRU/batch/chunk/cache profile with 2.18 GB observed VRAM peak; root performs fresh admission.

P2/P3 now pass absolute resume-state paths under the main project. Other input roots in the trainer are resolved against `WNN_PROJECT_ROOT`; the exact wrapper and root bindings are in the R2 command packet. Model, target equation except scale, losses, label weighting, optimizer, schedule, sample order, checkpoint policy and export precision stay matched to the frozen R3 control.

## R2 hashes

- [Exact fit, export and grade argv](FIT_AND_EXPORT_COMMANDS_R2.json) — SHA256 `318dff806c73799c84b8c0a140824f427f0acb8b1b07c77d4098dc58725bbc06`.
- [Source and input bindings](SOURCE_AND_INPUT_BINDINGS_R2.json) — SHA256 `e4240b0d041072c97eefe03cfe1a281ad32bcbe7279f28dc8691d1c4a999ccb2`.
- [Two-file source delta](SOURCE_DELTA_R2.patch) — SHA256 `4e05bdeff2c022dcd00e9ff1b798203094f34c07766e3ae0dcadd182a3eda164`.
- [Complete source manifest](source_overlay_r2/SOURCE_MANIFEST_R2.json) — SHA256 `ac086fd55d9833492191507cbfba7f1d978a763011be698097bd582372402818`.
- R2 trainer SHA256 `9b3079b9477fb0a1913bc5b2d504831126bff0d4cf02aff2b14f39872f421333`; R2 allocator source SHA256 `0b6280130a480f5547552df83cc4a15e7cfe88ad04498ca11c2ab2d54bb428e9`.

The critic has been asked to re-review these exact R2 hashes. R2 remains unlaunched until that review and root's explicit release.
