# Transformer optimizer and gradient-horizon screen

This packet executes exactly six new B300 fits authorized by the 8 November
instruction: three recipes (`C0` Muon256/A4, `C1` AdamW256/A4, `C2`
Muon512/A2) at paired seeds203101/203102, each fixed4E and global sequence
batch256. The models use Transformer768x24, BF16/BF16x9 forward with FP32
master/optimizer arithmetic, and the cache-free `standalone_labels_wmse_v2`
objective. Full recipe identity is in [the six frozen configs](configs) and
[profile-plan.json](profile-plan.json).

The official primary remains `global_wp_clip2_v1` on all 1,873 validation
sequences / 3,528,273 scored rows. Train has 10,325 fitting sequences and no
validation fitting ancestry. Validation is repeatedly exposed development
evidence; this screen cannot establish significance or a public-score gain.
Train-eval is a deterministic, seed-independent hash-ranked 128-sequence fit-only
cohort, clearly separated from the full valid population. Its gap is not claimed
comparable to valid.

The fixed endpoint is configured E4. All four epoch snapshots and same-forward
raw-z monitoring are retained for diagnosis; they do not select or replace the
configured final endpoint. Per-update diagnostics sample update1 and every50th
successful optimizer step. They add no inference pass. Their large-model memory
and transfer overhead is not yet measured.

Root owns source/data/native-profile receipts, resource/time authority, all
transport and GPU admission. The six exact-config50-update profiles are
engineering gates, not scientific fits; no profile checkpoint may be reused as
a fit. No research fit or grade is claimed by this packet until ROOT's receipts
and complete fixed endpoints are linked in [EXECUTION_R1.md](EXECUTION_R1.md).

## Packet map

- [CONTRACT_R1.md](CONTRACT_R1.md): authority, frozen protocol and acceptance gates.
- [PR.md](PR.md): required eleven-section experiment draft.
- [EXECUTION_R1.md](EXECUTION_R1.md): exact profile/receiver commands and finite lifecycle.
- [receiving](receiving): bounded plan builder, exact-profile launcher, cache-free
  phase adapter, six-cell receiver, score validation and CPU-only plots.
- [source capsule](../../training-repair-20261108/POINT5_B300.md): implemented
  optimizer telemetry, LR controls and context contract.

The AdamW cell is an optimizer recipe contrast, not a pure algorithm-only
intervention. It applies global weight decay to all parameters; C0's Muon groups
exclude bias/normalization no-decay parameters. This coverage difference remains
part of the declared contrast and must be reported with outcomes.

## Native execution

[NATIVE_EXECUTION_R1.json](NATIVE_EXECUTION_R1.json) records six qualified
50-update profiles and the first four actual full fits on all eight B300s. Two
remaining cells are queued under the frozen four-plus-two waves. The source is
5d73bc2b1, including the actual-loop cadence fix. No new validation/public gain
is established. The local dashboard is http://127.0.0.1:8769/. Raw launch,
qualification, profile, runtime and live progress receipts remain in the linked
local artifact root. Coarse controller status can lag actual producer updates.

[The live handoff](MONITORING_HANDOFF_R1.md) records verified epoch plots,
checkpoint availability, detached monitor/publisher owners and the first
automatic PR72 update. E1/E2 monitors remain provisional; final E4 grades and
independent endpoint byte bindings are pending.

[The R2 postprocessing repair](POSTPROCESS_REPAIR_R2.md) handles the observed
WMSE reduction roundoff after the frozen receiver exits, preserving its source
and receipts. It changes no scientific fit, metric or endpoint policy.
