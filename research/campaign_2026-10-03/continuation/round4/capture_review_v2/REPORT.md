Independent v2 review: BLOCKED by remaining exact RNG and moment type gaps
=====================================================================

V1 FAIL_AS_FROZEN and all earlier artifacts remain unchanged. V2 fixes the three
measured v1 mechanisms for optimizer IDs, Torch discrete clocks and calibrated
floating comparison. Its pristine CPU reproduction passes. A newly measured
NumPy prestate/comparison coercion gap and warm-moment dtype coercion nevertheless
violate exact captured-state identity. No controller receipt for v2; a separate
correction candidate is needed.
Candidate hashes below are an identity inventory, not authorization.

What actually ran
-----------------
One synthetic CPU actual-code probe, not a37-test rerun: actual RNN/Muon/OneCycle/
FastEMA/scaler recipe and four warmed chunks, six independently restored original
normal baselines and six independently restored original deliberate-skip baselines.
Normal, repeat, skip and uninstrumented-original maximum deltas are all0; both
empirical floors are0. Twelve original artifacts and their hashes are preserved.
Raw prediction, lagged EMA prediction, stopped blended target and actual clip norm
are now saved. CPU wall1.97s inside Python, peakRSS791432KiB; external resource.log
records process/import overhead. No CUDA, datasets, scoring or receipt creation.

Nineteen targeted tests rejected their corruption/unsupported condition, with one
within-floor positive control passing. Both same-shaped optimizer-ID swaps reject
in validate and restore, INCLUDING simultaneously forged optimizer_id_names. The
validator derives canonical names/IDs from bound source under fork_rng; restore
checks again before positional optimizer state loading. Tensor counter100000+1,
bool changes, bool/int scalar alias, int/float scalar alias and changed Torch RNG
bytes reject. Naked floating tolerance is forbidden. Signed-zero bytes reject
when tested without a prospective tolerance. EMA decay is now recipe-bound.

Calibration retest and source independence
------------------------------------------
Our independent AST erasure recovers the exact original chunk from BOTH online
instrumentation and the separately extracted original-baseline observer. Baseline
hooks observe source RHS terms/gradients/returned clip norm; they do not alter
addition order. The six-original calibration receives PRESTATE ONLY, not online
post-state; first reconstructed execution is included, and floor files/hashes
are written before differences are examined. Sharing the source-bound restore
and observation container is disclosed; extraction differs from online code.
Six identical scalar originals derive floor0 and reject a one-ULP change. Six
synthetic nonzero scalar originals derive floor5.00679016e-6, limit1.00135803e-5
(2*floor): within-limit control passes, outside-limit rejects. CPU nonzero floors,
over-fixed-ceiling floors, changed NaN masks, changed inf sign and only5 originals
all reject. Nonzero-policy fixtures use CPU arrays with a 'cuda' POLICY LABEL;
they are NOT CUDA measurements. No GPU floor has been observed. Globally cold
process/cache equivalence remains unsupported, as documented; no first outlier
may be dropped to improve the gate.

Remaining blocker: measured NumPy identity/coercion
--------------------------------------------------
live_capture.py:805–808 uses np.array_equal without dtype/byte metadata checks.
It accepts equal-shaped bool versus uint32, uint32 versus float64, and float32
+0 versus -0. Values alone are insufficient for the frozen typed/byte contract.
For the actual captured MT19937 words (uint32 shape624), equal-valued uint64 AND
float64 replacements pass capture.validate, capture.restore AND full RNG compare.
set_rng:157 calls np.random.set_state, which silently casts BOTH replacements
back to uint32. get_rng after restore confirms original numeric words but changed
declared dtype. validate checks RNG key names/topology, not this NumPy grammar.
This is a full prestate/restoration witness, not merely an artificial post-state
tensor. Exact scalar comparisons do not close array numeric aliasing.

Restoring the same numerical MT19937 state does not prove contamination or a
changed gradient, and no such claim is made. It DOES mean the validator silently
normalizes a different serialized state and its comparator certifies non-exact
identity. Output checksums and an external dtype post-check do not repair that
scientific acceptance gate. Our standalone check_numpy_rng.py was diagnostic
only; it passed current pristine artifacts, but is expressly NOT an approval
workaround. Controller's exact-RNG requirement remains unchanged.

See V3_MINIMAL_SPEC.md: NumPy metadata+byte equality, canonical source RNG schema
before restore, no coerced types, and exact set/get roundtrip with alias/endian/
signed-zero regressions. Do not edit frozen v2 or relax baselines/tolerances.

Additional decisive blocker checked before final freeze
------------------------------------------------------
The same implicit-cast problem remains in optimizer-moment validation: each of
Muon momentum_buffer, Adam exp_avg and Adam exp_avg_sq was replaced by an
equal-valued float64 tensor of the SAME shape. All three pass validate and restore;
optimizer.load_state_dict silently converts each back to float32. The restored
values equal the pristine moments, and no changed gradient or GPU corruption is
claimed. Unlike ndarray comparison, Torch compare would reject those dtype
differences directly; the gap is specifically the prestate validator/loader boundary.
V2 checks moment fields/shapes and native int step, but not moment dtype/storage.
V3 must reject incompatible moment dtype/device/attachedness before loading and
preserve the source-defined finite detached CPU FP32 capture grammar. See
moment_cast_results.json and moment_cast.log. This cheap extra test did NOT rerun
the calibration or37-test suite. Other source/floor checks above remain passing.

Runtime and hashes
------------------
The controller-applied wrapper selects v2 and TERM585/KILL5. Actual SHA256 is
335233c719cea03a9cef689a12ad89150682dc1caa0f0bc49038913b266af8ad,
matching the actor's anticipated patched hash. Frozen runtime_manifest.json's
unapplied/current-old-payload entries are historical, not current authorization.
Other current runtime hashes match; final bindings are in authorization_inputs.json
with explicit BLOCKED/NOT_APPROVAL status. No queue_finished assertion is written.
The controller reported successful terminal queue/score units and healthy watch;
this reviewer did not independently attest live units and issues no receipt.

Reviewed chain remains campaign_job --kind gpu --timeout600 --threads8 -> run_gpu.sh
-> controller wrapper -> runner,8CPU/24GiB GPU family, allocator4.8decimalGB and
observed5.5decimalGB caps. Exclusive admission plus terminal controller attestation
remain mandatory. TERM585/KILL5 gives nominal child<=590s with600s outer execution;
605s remaining guard protects the23:00Kyiv deadline. CPU evidence does not establish
CUDA/cuDNN/fp16 parity, real-panel success, intervention benefit or today's full-fit
incoming state. Fresh panel-normalized trajectory remains a limited measurement.

Verified20 protected v2 entries and53 v2 artifact hashes; v117 protected/54 artifact
entries,17 blind artifacts,12 lineage artifacts and15 v1-review artifacts preserved.
Nine candidate files and9 source bindings match. Full expected/actual checks are
in protected_verification.json; candidate/source/final-runtime hash inventory is
authorization_inputs.json, unusable as a receipt. Prior state/provenance verdicts
were not rewritten. No actor/runtime edits, GPU launch or approvals occurred.

Artifacts
---------
results.json and cpu_reproduction/ preserve pristine/six-original tests;
followup_results.json and rng_prestate_results.json preserve the exact remaining
counterexample. V3_MINIMAL_SPEC.md is the minimal builder handoff. Scripts, logs,
resource.log, protected_verification.json, STATE.json and HASHES.json reproduce
this bounded review. A positional-argument invocation error is retained in
numpy_guard_attempt1.log; corrected diagnostic succeeded without policy changes.
