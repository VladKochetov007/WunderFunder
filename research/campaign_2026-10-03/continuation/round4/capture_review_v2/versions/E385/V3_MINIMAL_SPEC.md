Minimal separate-version correction — exact RNG/type identity
===========================================================

Do not modify frozen v2. Its numeric RNG dtype alias is blocking, even though the
aliased fixture restores the same numeric MT19937 words. Exact serialized state
identity is the frozen contract; mathematical equivalence after coercion is not
an allowed replacement. No controller receipt for v2.

1. In compare's NumPy-array branch, require a NumPy ndarray on both sides, exact
   dtype INCLUDING byte order, exact shape and equal canonical contiguous bytes.
   Do not use np.array_equal alone: it accepts bool/integer/float aliases and
   signed-zero changes. Compare via bounded contiguous byte views/tobytes after
   metadata checks, with no cast to a common dtype. The baseline policy delegates
   nonfloating/NumPy state to this exact branch; neither floors nor ceilings may
   relax RNG or discrete metadata. Preserve Torch and scalar exactness.

2. Validate the supported RNG grammar BEFORE constructing/restoring a replay.
   For this source get_rng: NumPy state is exactly the 5-tuple MT19937 grammar;
   words are a uint32 ndarray of shape(624,), with canonical byte order/layout;
   index and has_gauss use exact native int types/ranges, not bool/float aliases;
   cached Gaussian uses its native float type. Validate Python/Torch/device RNG
   state types/storage similarly, preserving source backend/version/topology.
   Do not accept a dtype solely because np.random.set_state can cast it.

3. After set_rng, verify get_rng's full typed/byte state matches the intended
   prestate WITHOUT consuming random draws. If a setter normalizes types/values,
   reject. This roundtrip may be a test rather than repeated online overhead,
   provided type validation makes coercion impossible on the supported path.

Cheapest regressions (retain own v2 tests)
- From a genuine CPU pre.pt, replace the MT19937 uint32 words by equal-valued
  uint64, float64 and opposite-endian uint32 arrays of identical shape. validate
  and restore must reject BEFORE forward; full RNG compare must reject each.
- Compare equal-valued bool versus integer arrays; float32 +0 versus -0; dtype
  mismatch and one-word change. Reject every change without a numeric envelope.
- Genuine prestate validates, serialized roundtrip remains typed/byte-identical,
  set/get RNG matches, and the same raw/EMA/Muon/OneCycle normal+skip CPU probe
  retains zero floors. Six original baselines must remain independent; no online
  deltas enter calibration and no ceiling/tolerance changes are needed.
- Keep same-shape optimizer ID + forged-association rejection and existing exact
  tensor/scalar clock controls. No37-test sweep is needed to establish this fix.

Refreeze candidate/source/generated/runtime bindings with new version hashes.
Only after independent correction retest may the controller attest terminal live
queue state and consider a receipt for bounded default CUDA parity. No estimator
intervention, mature full-fit reconstruction or source/runtime edit is authorized
by this specification. The already-applied controller wrapper may remain as-is
until a reviewed separate candidate path is explicitly selected by the controller.
