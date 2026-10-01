# Conditional native whole-fold preparation — not allocated

This is source/protocol preparation only. No native OOF production, model export
or quality query occurs before all four corrected core gates pass and root freezes
and allocates an exact plan under amendment-02. The current training sources and
execution-plan v2 remain unchanged.

The proposed producer uses the actual scorer CPython3.11 environment and the
already frozen v3 extension c37161f890dae279873cb0a931c34542a0436d82886303776ce57d0f87fb32d6.
That environment has pyarrow and numpy available, confirmed by module discovery
without data access. It can therefore use the exact existing SequenceFile reader
directly, without a different Python ABI or a temporary raw-feature disk cache.

For each fold separately, export exactly these fixed epoch1 checkpoints, in order:

1. on_ctxmemory_s3_f{fold}
2. on_ctxmemory_s4_f{fold}
3. on_ctlc500_f{fold}

Use equal thirds and the unchanged archived solution.py callback wrapper. Export
every state array verbatim as float32 NPZ, including the fixed FP32 projection and
the learned extra matrix in the first two members. Check all keys/shapes and source
checkpoint hashes against the completed fit-integrity receipts before export, then
reload and check every array bitexact. Ordinary native load-time FP16 conversion
of GRU/head/E remains unchanged; P remains FP32. The third member has no context
metadata, and all its source arrays remain unchanged.

An immutable composite manifest should bind all three source checkpoint hashes,
their arguments/folds/epochs, all exported NPZ/config/wrapper/extension hashes and
the exact fixed coefficient/order list. The prediction receipt must bind that
entire manifest, not imply that one memory checkpoint alone identifies the
three-member composite. Root's guarded grader can check the manifest lineage in
addition to its existing prediction-file hash checks.

The scorer-environment worker reads only raw feature columns with
targets=False,mask=False and row_group/split/fold metadata. It follows the same
full valid-cache row order and writes only its fold with NaN elsewhere. Reset the
complete composite at every independent20000-row sequence, matching GPU OOF.
Use the exact wrapper callback for every row with need_prediction=True, including
warmup, without reading the hidden/scoring mask. This predetermined return flag
does not alter native state updates. No alternative producer path will be selected
based on quality outcomes.

Hash full valid.parquet, valid_targets.npz, chain_folds.npz and the reader before
and after production. Use one CPU thread per worker, no Torch/CUDA import in the
producer, finite shape checks, exact source/package/native import guards, immutable
partial-file handling and an external resource/timing bound. On roughly half of
the1873×20000 validation rows, two fold workers are about20–25 CPU-minutes total
at the measured native rate, excluding input decoding/hashing. This is an estimate,
not a new runtime gate or a measurement from trained models. The root should choose
two free physical CPUs with idle siblings if parallelizing the folds.

Amendment-02 requires exactly two further comparisons, after core success:

- Native composite versus the same GPU memory composition: pooled and each-fold
  delta at least-.0001.
- Native composite versus the existing plain GPU H2 anchor
  on_ctl128s3+on_ctl128s4+on_ctlc500: pooled at least+.0003 and each fold nonnegative.

These are native-versus-GPU audit comparisons, not a native-versus-native superiority
claim. No precision, projection, clock, half-life, member or blend tuning may follow
the results. Final full-data head training and ZIP export remain conditional, and
final all-ISA correctness plus the absolute50-minute complete callback gate remain
separate requirements.
