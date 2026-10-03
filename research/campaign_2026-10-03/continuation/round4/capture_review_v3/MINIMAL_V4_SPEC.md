Minimal v4 correction — exact private RNG engine roundtrip
=========================================================

Preserve frozen v3. Its independent verdict remains
BLOCKED_TORCH_RNG_ACCEPTED_STATE_NORMALIZATION. This supplement does not amend
REPORT.md, STATE.json or HASHES.json and is not a GPU receipt.

Required implementation
-----------------------
Inside validate_torch_rng, retain canonical Tensor/dtype/shape/CPU storage/layout
checks and private-engine acceptance. Immediately after engine.set_state(state),
require engine.get_state() to have exactly the intended tensor metadata and bytes.
Reject normalization before any global setter, replay construction or forward.
Use this same function for the private CPU engine and each private CUDA engine.
Do not consume random draws, canonicalize the declared input, mask ignored bytes,
relax compare, change scales/floors or add scientific arms. Engine acceptance alone
is insufficient. The existing exact comparator can implement the roundtrip check;
translate its failure to an explicit validation error if needed.

Cheapest decisive verification
------------------------------
Reuse a genuine warm prestate and the existing pristine CPU control. The offset28
uint8 perturbation must reject in validate_rng, full validate, restore and set_rng;
each rejected attempt must leave the complete global RNG unchanged. Existing
private-engine witnesses at offsets20/36/44/5048 provide cheap optional checks.
Pristine and legitimate advanced/cached source states must remain byte-exact.
Keep existing NumPy typed grammar, warm-moment and canonical optimizer-ID fixes.
No broad test sweep or calibration refit is needed. CUDA private-engine behavior
remains unmeasured until a separately reviewed bounded default CUDA measurement.

Evidence and admission
----------------------
torch_engine_results.json records full prestate validate/restore acceptance and
declared byte1 versus restored byte0 at offset28; exact comparison rejects the
result. This establishes an integrity gap, not changed gradients or GPU corruption.
Source/backend-dependent opaque RNG bytes require installed-engine roundtripping,
not an invented padding layout.

Refreeze v4 candidate/generated/source hashes. The controller must separately
bind the selected v4 runtime and current exclusive queue admission. Existing v3
runtime d841cee9... and v3 review cannot authorize v4. Only unchanged/default
CUDA parity may later be considered, with original six-run floors and mandatory
online/default/skip gates. No receipt, GPU launch or intervention is authorized here.
