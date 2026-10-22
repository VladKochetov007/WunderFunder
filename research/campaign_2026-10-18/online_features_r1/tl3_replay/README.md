# TL3 exact-ZIP F0 replay

This packet replays the unchanged official-team artifact
`submissions/team_tl3_original_20261018.zip` (SHA256
`d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`) through
its embedded Python 3.11 `wnn_cpp` callback. It reuses all three fixed members
and coefficients from the package. No model fitting or blend-coefficient
search occurs here.

## Evidence boundary

The independent [data audit](../DATA_AUDIT.md) records TL3's actual
training, teacher, initializer, and sequence membership as unknown. The ZIP
docstring calls its parent teachers all-data. Its F0 predictions and every
local score or complementarity result must therefore be labeled potentially
fitted diagnostics, not clean cross-validation or a held-out gain. F0 is also
previously exposed and has been used for alpha and geometry selection.

The exact callback receives only `seq_ix`, `step_in_seq`,
`need_prediction`, and the current 112-float state. The replay processes all
99 warmup rows in each sequence and verifies row-group/sequence identities
against the frozen F0 split. It does not read target or score-mask payloads;
labels remain available only to a separate offline evaluator. Each prediction
cache sidecar preserves the split hash, original row-group ordering, sequence
IDs, ZIP hash, and ancestry caveat so existing fixed-blend graders can align it.

## Implementation and execution

[score_tl3_f0.py](score_tl3_f0.py) composes source validation
([f0_source.py](f0_source.py)), bounded callback replay
([f0_replay.py](f0_replay.py)), receipt creation
([f0_receipt.py](f0_receipt.py)), and safe package/native loading
([native_runtime.py](native_runtime.py)). Ruff format/check passed. It uses
the cached offline Python 3.11.16 image with PyArrow 19.0.1, NumPy 2.2.6, and
the exact archive's CPython 3.11 native extension. No Torch or network access
is required.

The [eight-group profile](profile_first8.json) processed 160,000 rows with all
792 warmup rows suppressed, finite required outputs, and an exact A–B–A reset
replay. It selected native v3 dispatch for the three-member dynamic-int8
ensemble. The profile's 35.61 second wall time and 344,356 KiB process RSS
were measured at 0.5 CPU and a 1.5 GiB hard cap.

The [full-F0 receipt](full_f0_serving.json) records 468 source-qualified groups,
9,360,000 rows, 46,332 warmup rows processed and suppressed, 9,313,668 finite
predictions, archive/split/producer identity, exact native-library and replay
source hashes, and runtime/resource data. The first-eight reset check is the
reset evidence for this package; the full replay did not add an A–B–A pass.
The [prediction cache](../../../../cache/pred_tl3_unknown_ancestry_f0_valid.npy)
is 468×20,000×2 float32, SHA256
`3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3`; its
[sidecar](../../../../cache/pred_tl3_unknown_ancestry_f0_valid.json) preserves
the exact F0 row-group order and sequence IDs for aligned grading.

Full-F0 callback cost was 828.95 seconds, 88.56 μs per row. Extrapolating by
row count to 39.4M callback rows gives 3,489.39 seconds (58.16 minutes); this
is a callback-only projection, not a measured full competition runtime. The
process RSS peak was 400,876 KiB, while the external Docker cgroup recorded a
2,049,712,128-byte total-memory peak under its 2 GiB hard cap. Docker's host
cgroup is outside `wnn-campaign.slice`; the separate
[admission receipt](../EXTERNAL_DOCKER_RESOURCE_SPLIT.json) and
[completion/restoration receipt](../EXTERNAL_DOCKER_RESOURCE_RESTORE.json)
record that scope and the parent-slice measurements. Do not infer the cgroup
total from process RSS alone.

The ZIP metadata calls its parent models all-data, while exact training,
teacher, initializer, and sequence membership remains unknown. F0 was
previously exposed; every local score or complementarity result must therefore
remain labeled potentially fitted diagnostics, not clean cross-validation.
This task did not score outputs, fit blend coefficients, add test-time
adaptation, or upload the archive.
