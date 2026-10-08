# E540 V10 fixed-checkpoint export

V10 is an export-only recovery for two fits that already completed. It reuses the V8 linear control checkpoint and V9 PLE candidate checkpoint at fixed epoch 0, then writes new FP32 predictions for the same 468 evaluation groups. It runs no profile, training, checkpoint update, metric or grade.

V9's PLE fit completed 72,996 of 72,996 optimizer updates with zero skips, and its fixed checkpoint and provenance were collected. Its worker failed in a post-fit consumer check because `history.train_seconds` was 559.7 while sibling `provenance.fitting_seconds` was 559.8. The trainer writes rounded history timing before checkpoint serialization and rounded sibling provenance timing after serialization. These are separate clocks. The new validator accepts the observed 0.1 second difference and enforces a 30 second maximum checkpoint-save interval with 0.11 seconds for independent tenths rounding. It leaves both original timing fields and training artifacts unchanged.

The prediction command uses `predict_run.py` with `--fit-subset-manifest` and FP32 inference. Its `FeatureOnlySequences` reader requests `targets=False` and `mask=False`; only `valid.parquet` is mounted, and the worker checks that file's byte identity before loading model tensors. It verifies both checkpoint/provenance/history sets, exact source and subset hashes, identical saved sampler and normalization identities, finite tensors, the original PLE knot digest, and final 468-group output sidecars. The root grader owns all score computation.

Both fits use the same already frozen E540 recipe and seed 540; only the numeric map differs. V8 control and V9 PLE source fits were independent and both completed. The old V9 receipt remains immutable and reports “fit completed; prediction export absent” rather than treating the consumer error as a scientific failure.

No provider assets or kernel are uploaded by this packet builder. The generated packet remains pending root review and export authorization until the exact source, artifact binding, private assets and bundle hashes are independently checked.
