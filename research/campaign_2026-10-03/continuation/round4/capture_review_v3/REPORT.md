V3 independent review — BLOCKED: accepted Torch RNG state is normalized
====================================================================

The NumPy and warm-optimizer corrections PASS, but the unchanged exact-state
contract still has one concrete blocker. No GPU launch, receipt, scientific arm,
dataset scoring or candidate/runtime edit occurred.

Measured correction tests
-------------------------
retest.py ran one pristine tiny actual-code CPU trajectory, normal/default/repeat
and forced-overflow skip controls, six independently restored original baselines
per control, plus targeted adversaries: 134 assertions passed. All four maximum
deltas and both empirical floors are zero. Cost inside Python: 2.182s,
peak RSS792948KiB; wrapper also accounts for startup. These are independent
measurements, not the builder's reported66 tests and not CUDA parity.

Equal-valued NumPy uint64/float64/opposite-endian uint32 states reject during
validate, restore, set_rng and compare. Tuple/scalar types, native-int position
bounds/cache flags, cached-float type/finiteness, words shape/layout, Python word
and version types, Torch byte dtype and CPU/CUDA topology reject before global
RNG mutation. Numeric/bool aliases, signed-zero and changed words reject. Two
independently seeded, advanced states with Gaussian caches pass exact set/get;
pristine serialized RNG and named moments restore exactly.

All three warm optimizer moment fields reject float64, attached, wrong-shape,
meta-device and nonfinite fixtures before loading. Same-shaped ID swaps reject
even with forged association; discrete clock aliases and wrong backend reject.
Removing observers from both online and independent-baseline ASTs recovers the
exact original CUDA-source chunk. Math, baseline policy, panel and generated
adapter bytes are unchanged from v2. CPU uses its documented FP32 adaptation.

Remaining falsifier and cheapest correction
-------------------------------------------
torch_engine_boundary.py tests13 selected bytes of the opaque CPU RNG state
using private engines. Five alterations pass validate_rng but fail exact private
set/get roundtrip. At offset28, the same uint8 shape/storage prestate changes0
to1: complete capture.validate and capture.restore BOTH accept; actual restored
get_rng reports0, and capture.compare rejects the declared-versus-restored RNG.
The pristine and several changed canonical states roundtrip exactly, providing
positive controls. The original global RNG is restored and checked afterward.

validate_torch_rng checks metadata and private engine acceptance, but not the
private engine's resulting state. Thus supported metadata does not make silent
normalization impossible. The direct comparator correctly detects the difference;
the defect is the pre-forward validation/restoration boundary. No changed draws,
gradients, contamination or GPU corruption are inferred from these ignored bytes.

REQUIRED: in a separate preserved version, after private engine.set_state(state),
require its get_state() to match the intended state exactly before any global
setter, replay construction or forward. Apply the same rule to CPU and each
CUDA engine without consuming draws. Preserve all supported pristine states and
existing typed checks. This uses an already-created private engine, requires no
new RNG layout assumptions, tolerance or refit. Regress offsets20/28/36/44/5048;
pristine/private accepted canonical state remains a positive control. NumPy and
Python private setter roundtrip assertions would make the analogous guarantee
explicit. No external postcheck or v3 receipt workaround is sufficient.

Integrity and permitted future scope
------------------------------------
All protected checks pass: v1/v2/v3 author+artifact inventories; blind, lineage,
prior reviews and final v2 amendment; nine candidate/nine source hashes; generated
files and actual runtime chain. candidate_runtime_hashes.json is an identity
inventory, NOT a receipt. Actual controller payload now selects v3 and hashes
d841cee9e6816384d49e5023a66972063d6aa9f51189c0343d40f11d22606917.
Actor runtime_manifest is a historical unapplied record. TERM585/KILL5 remains
nominal<=590s inside600s admission;605s remaining deadline guard and exclusive
campaign_job -> run_gpu.sh chain remain necessary. Shell syntax passes.

E385 original report/spec were recovered using the exact inverse amendment in
THIS thread's recorded file edit, then verified against ledger hashes BEFORE
writing capture_review_v2/versions/E385/. Hashes de15ebb7... and4428b173... match.
Final amended v2 report/spec/STATE/logs and ledger were unchanged; historical
STATE was not reconstructed. Controller independently recovered the same bytes
and records the final amendment separately as E389. legacy_recovery.json proves
our exact recovery; no earlier verdict was rolled back.

After correction and independent retest, only bounded unchanged/default CUDA
measurement could be considered, with mandatory six-original floor/online replay/
skip gates and current controller queue/runtime attestation. CUDA engine validation
and fp16/cuDNN parity remain unmeasured. Fresh small-panel normalization/trajectory
does not reconstruct today's full fit. No scientist arms are admissible here.
