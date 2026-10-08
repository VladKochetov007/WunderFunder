# E540 V11 fixed-checkpoint export

V11 is an export-only successor to V10. It reuses the same completed V8 linear and V9 PLE fixed-epoch-0 checkpoints, the same 512-component fitting provenance, and the same 468 feature-only evaluation groups. It performs no fit, profile, checkpoint update, ONNX export, or grading.

V10 terminated before data loading because its consumer required `binding.recipe_sha256`, a field absent from the frozen binding. The recipe digest was already independently bound by the runtime spec and checked through `WNN_E540_RECIPE_SHA256`. V11 removes only that redundant absent-field comparison; it retains the recipe environment check and the subset, split, source, checkpoint, sampler, normalization, knot, membership, authorization, and feature-only checks. The V10 terminal receipt and worker result remain bound in the V11 recipe/card.

The worker predicts only the fixed 468 groups and the feature reader requests neither targets nor masks. The original model-fit artifacts are immutable inputs. Root retains metric computation and scientific interpretation; the previously exposed groups are not a fresh holdout or public forecast.

The private artifact dataset is reused byte-for-byte from V10. The source packet remains pending root review and a new hash-bound authorization. This packet has not been uploaded or pushed.
