V4 minimal correction frozen for independent review. No selfapproval or GPU receipt.

Only executable change: validate_torch_rng now compares the private engine's
get_state() metadata and exact bytes immediately after set_state(), translating
normalization into a validation error. The same function serves CPU and supported
private CUDA engines. No draws, byte masking, input canonicalization or tolerance
changes. Capture schema is4. NumPy grammar, moment/name-ID association, gradients,
clip/update order, source adapter, six-repeat floor policy and candidate math stay
unchanged. Necessary source/tests/panel bindings are local to v4; v1/v2/v3 protected
and artifact hashes plus all-file preservation inventories are checked in LINEAGE.
The runner is byte-identical to v3; its output manifest's historical "v3" lineage
annotation is inherited, while HERE, schema4, candidate/generated hashes and the
runtime patch bind this v4 directory. No semantic runner changes were needed.

Final critic read: capture_review_v3 REPORT/STATE, MINIMAL_V4_SPEC.md and
SUPPLEMENT_HASHES.json. Verdict BLOCKED_TORCH_RNG_ACCEPTED_STATE_NORMALIZATION;
NumPy/moment fixes independently PASS. Its134 assertions, zero CPU floors, legacy
E385 recovery and2050source/310numerics/privatecontext results are prior independent
evidence, not new v4 measurements. The additive supplement and frozen critic hashes
are verified during v4 freeze. No critic or earlier-version artifact was edited.

New measurements: logs/verify.log records30 targeted tests passing in1.80s and Ruff
passing. A genuine warm actual-source prestate rejects each XOR1 mutation at bytes
20/28/36/44/5048 at validate_rng/full validate/restore/set_rng; the complete global
Python/NumPy/Torch RNG stays exact after each rejection. Actual private get_state
snapshots for seeds0/19/734019 in initial,17-uniform-draw and odd-Gaussian/cache
states pass private and global exact roundtrips; validation consumes no global
randomness. No invented opaque byte-layout assumptions are added to implementation.

cpu_smoke/results.json:1.907s, original/default, repeat, normal and deliberate AMPskip
max deltas all0; six independent original unchanged repeats per normal/skip control
produce0 floors. All four losses/gradients, output/laggedoutput/stoppedtarget,
clip/step, student/laggedEMA, both incoming hidden states, warm moments, RNG/scaler/
scheduler and pre/post clocks are retained. Prestate tensors20584bytes, component
replay tensors6528bytes. No failed v4 computation; complete logs/captures are kept.
The targeted suite is new v4 evidence; prior66tests are not claimed as a v4 rerun.

Limits: this is tiny CPU FP32 actual-source code, not GPU fp16/cuDNN parity. Private
CUDA engine roundtrip is implemented but unmeasured. Actual configure_torch flags
are inherited and checked; no blanket nondeterministic-backend claim. Warm process
backend/descriptor replay only; cold process/cache is unsupported. Hidden dtype is
captured exactly with no invented conversions. Unsupported optimizer/precision/
architecture states still fail closed. Fresh small-panel trajectory and panel
normalization do NOT reconstruct today's full-fit student/laggedEMA/hidden/moments;
EMA exports cannot supply them. No real fit rows/cache predictions loaded or scored
in this pass. Real-mode proposal remains only the frozen permitted28TRAIN+4fitVALID
mixture and mature teacher bindings, no evaluation score/model selection.

controller_payload.patch is UNAPPLIED, only v3->v4 scope. Expected resulting payload
SHA256:4d54fc09d9c29646155d73d7293b7759b18671cfd48fc69c975e7db07384e929.
Runtime chain stays campaign_job->run_gpu.sh->controller payload; TERM585/KILL5,
nominal child<=590s, outer600s,605s entry margin and absolute23:00 Kyiv cutguard stay.
Proposed synthetic actual architecture is batch32/chunk1000/GRU256x2 then bounded
fixed-fit-panel capture, no interventions. GPU cost is unmeasured and hard bounded
by590s (~0.164 GPU-hours), not a performance prediction. Full real prestate estimate
~600MB exceeds old512MiB part bound; controller retains streamed backup/captures.
Missing historical full-fit state bytes are unknown/irrecoverable, not zero.
Remaining conditions: independent final-v4 review, controller scope/runtime/hash
binding with exclusive admission/queue completion, then separately authorized
unchanged CUDA original-floor/default/skip gates. No GPU launch, receipt, queue/live
source/controller edit, calibration relaxation or aligned/permuted observation.
