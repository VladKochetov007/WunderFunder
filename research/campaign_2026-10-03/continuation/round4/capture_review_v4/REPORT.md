V4 independent review — PASS for bounded unchanged/default CUDA measurement
===========================================================================

No remaining blocker was observed in this narrowly targeted correction review.
This is permission-scope advice to the controller, NOT a launch receipt or CUDA
parity claim. No GPU, scientist arms, dataset scoring, candidate/runtime edits or
queue changes occurred.

Actual correction and independent measurements
----------------------------------------------
The v3->v4 executable diff adds exact compare(state, private_engine.get_state())
immediately after private set_state, translating failure into ValueError. Schema
changes from3 to4. The same validator serves CPU and private CUDA generators.
There is no masking, input normalization, random draw, floor change or altered
training math. Validation precedes reconstruction/global setters/forward.

Our retest.py uses a NEW tiny actual-source CPU capture and32 targeted assertions,
not the builder's30 tests or a repeated broad suite. At offsets28/20/5048, each
malformed Torch state rejects in validate_rng, complete validate, restore and
set_rng. Complete Python/NumPy/Torch global state stays exact after every rejected
attempt. Two independently selected seeds23/817 with uniform draws, odd Gaussian
draws and Python/NumPy caches pass private validation and global set/get byte
roundtrip. Validation consumes no global randomness. Pristine named moments and
full RNG restore exactly.

Earlier cheap regressions also pass: NumPy uint64/float64/opposite-endian aliases
reject on restore AND exact compare; all three warm moment float64 aliases reject;
same-shaped optimizer IDs plus simultaneously forged association reject; native
bool/int and tensor clock differences, NumPy bool/int aliases and signed-zero
reject. Pending review documents and entry outside the GPU chain reject.

One actual-code CPU normal/default/repeat/forced-overflow-skip probe retains zero
maximum deltas and zero normal/skip floors, with six independently restored
original baselines per control. Skip leaves student weights unchanged while
attempt/skip and source-defined EMA/scheduler advancement are checked. Saved raw
and EMA predictions, stopped target, clipping, loss/gradients and full states
remain in cpu_reproduction/. Inside-Python cost1.968s; peakRSS791728KiB. We reuse
known falsifiers and shared source-bound reconstruction deliberately; this is
independent execution of the fix, not independent discovery or CUDA evidence.

Preservation, source and runtime
--------------------------------
Every checked identity matches: v1/v2/v3/v4 author/artifact inventories; blind,
lineage and all prior critic final manifests; v3 additive spec; exact E385 archived
originals; nine v4 candidates, nine sources, three generated files and four actual
runtime files. Source trainer remains1d6b059d... . Runner, baseline math, panel,
source bindings, generated adapter and patches are byte-identical to v3.
The inherited output manifest's literal 'v3' lineage string is historical;
directory, capture schema4 and candidate/runtime hashes establish the version.
Do not identify the executable version from that prose annotation alone.

Actual applied controller wrapper SHA256:
4d54fc09d9c29646155d73d7293b7759b18671cfd48fc69c975e7db07384e929.
Shell syntax passes; reversing ONLY its v4->v3 scope line recovers the expected
previous wrapper hash. Actor runtime_manifest remains a historical unapplied
record, not the current runtime identity. candidate_runtime_hashes.json binds
the actual selected v4 chain and full candidate/source inventories, NOT a receipt.

Mandatory gates and limits
-------------------------
The controller must independently attest CURRENT terminal queue/exclusive
admission and issue a receipt binding this approved version and current runtime.
Use campaign_job(kind GPU,8threads,timeout600) -> run_gpu.sh -> applied wrapper.
TERM585/KILL5 gives nominal child<=590s;605s remaining deadline entry guard and
23:00Kyiv cutoff remain. One GPU family,8CPU/24GiB, allocator<=4.8decimalGB and
observed-device<=5.5decimalGB guards remain mandatory.

Only unchanged/default synthetic actual-architecture then fixed-panel trajectory
measurement is within scope. Six ORIGINAL baselines must establish stable bounded
floors before online comparisons; online/default/repeat full-state replay and
deliberate AMP-skip gates must pass. RNG/discrete metadata remains exact; floating
states use the unchanged zero-floor bitwise or empirically bounded backend policy.
Private CUDA roundtrip, GRUfast/fp16/cuDNN parity, runtime cost and real-panel replay
remain unmeasured and must fail closed if unsupported. CPU success does not prove
CUDA parity. Fresh panel normalization/trajectory does not recover today's mature
full-fit state. No intervention, scientist arm or forecast-quality claim is approved.

Review complete; final STATE.json and verified HASHES.json are the terminal critic
bindings. Prior reports/specs/logs and E385/E389 evidence remain unchanged.
