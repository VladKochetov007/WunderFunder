# E489 independent source review

Reviewed the frozen E489 trainer and callable GPU probe for admission of the
actual-shape probe. No CUDA probe, full trajectory, test-role read, or score was
run by this reviewer.

## Frozen identities

- `fit_fp32_duration.py`: SHA-256
  `af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc`
- `gpu_probe_fp32_duration.py`: SHA-256
  `2f0bd08aa0e760956a233ec144cb41c82613cc5ce98aec1cc5ed7b4400effd6f`
- `tests_fp32_duration.py`: SHA-256
  `fa5cd3da2afd9860b0a34d2a929ae1526747650d1f9dcd71cfbb34e553d66ec6`
- E489 design: `0859e6ac528b850452183afc46ab28358cac13c97ace5cb0350e413c2b9a7090`
- E489 provenance addendum: `9a1803680163d556037cabecfbc5cdfb923b30039140e64cda783493bfe2ae68`
- E487 verifier v4 and review: `f717d0d30d1c6e1456b80bdd31d76896633a588ea22e5b9243652af5f92391e6` /
  `0685ce65335c956a94003700827798c8d3080d746bb876362074cc19545f9132`
- E487 saved-delta reproduction receipt: `4fced259c38beae9a74652cf486ccd952d62c830e5927e5316be161aba6c6cac`
- Independently hard-limited fit-only preflight manifest:
  `cpu_preflight_root_limited_v2/fit_manifest.json`, SHA-256
  `3bcd0909c800f9c6f133ad55a23b9490cf85ea64ed8b4deec8151e73702bed6a`

## Findings

The candidate keeps the registered 42-input, two-layer H128 native model
topology and adds no trainable parameters. Alpha zero uses the native reset,
update, and candidate equations; alpha 0.1 changes only the bounded recurrent
candidate. Both variants load identical native initialization state. The
trainer uses the pinned E487 FP32 diagnostic optimizer, checks the four
registered recurrent matrices are the only Muon group (three Newton–Schulz
blocks), and validates the shared AdamW settings. Every batch resets hidden
state, chunks carry detached state, optimizer moments continue across the four
epochs, and the fixed seed-plus-epoch shuffles cover the 24 fit-role groups
exactly once per epoch. The 79 chunks per batch produce 474 updates per epoch;
epochs 1, 2, and 4 produce the registered 18 snapshots.

Before any training, the code checks immutable E470/E471/E474/E487 sources,
dataset/cohort, candidate, design, provenance, and exposure receipts. The E487
gate requires the independent strict FP32 arithmetic pass, finite saved
deltas, the v4 verifier/review, and verified teardown. The probe repeats that
gate before CUDA setup, sets the 2 GiB allocator limit before configuring
CUDA, requires one visible device, and stops above the 3 GB physical-device
limit. Its native alpha-zero parity check covers outputs, recurrent state,
input/parameter gradients and clipped optimizer deltas for all three seeds,
using the preregistered tolerances. The six seed/alpha cells each perform two
successive B4×256 updates with detached carried state and persistent optimizer
state; raw gradients, loss, forward tensors, parameters and optimizer state are
checked for finiteness. Strict update-delta gates retain the registered
`0.02` relative-RMS and `2e-5` maximum-absolute limits. Probe timing includes
input transfer and synchronization, and it records initialization hashes,
active IEEE settings, per-cell memory, and its complete body time.

The controller must use the actual provider-charged probe duration for the
7200-second admission calculation. The probe report also supplies the
conservative `max(update time) × 11,376 × 1.5 + 120 seconds + probe` estimate;
all gates, the 600-second probe cap, hard memory stops, teardown, and source
identity checks remain required before admitting full fits. This review does
not admit full fits by itself.

The independent test run passed all four synthetic tests, and Ruff plus
`py_compile` passed under `scripts/limited` with a 1 CPU / 2 GiB cgroup. Root
also reran the real-data one-update-per-arm preflight under the same hard
limits; its manifest has six finite, changed updates, all 24 fit-role inputs,
and `test_rows_loaded: false`. The earlier `SOURCE_NOTE.md` example commands
set resource environment variables without invoking `scripts/limited`; those
examples alone do not demonstrate hard cgroup limits. The separate root
limited run supplies that evidence.

## Disposition

**GO for the sealed E489 actual-shape CUDA probe only.** Bind the exact source
hashes above and the root-limited preflight receipt. Full-fit admission still
depends on a passing three-seed native parity probe, the original strict
per-family update gates, bounded timing projection using actual provider
duration, and verified resource/teardown receipts. No tolerance or resource
gate is relaxed by this review.
