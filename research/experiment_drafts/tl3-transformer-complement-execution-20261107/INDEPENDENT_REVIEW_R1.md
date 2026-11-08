# Independent receiving-source review

Reviewer: compact-model execution agent,7November2026. This reviewer did not
implement the B300 receiving changes and performed no SSH, data access or GPU
execution. Scope is the current receiving amendment, canonical integration,
reference export/capsule path, monitor, comparator, configs and run.sh commands.

Disposition: no remaining source blocker found after the two concrete issues
below were fixed by the B300 actor. This admits source preparation to native
receiving checks; it does not approve an unmeasured profile or scientific fit.

1. The initial run.sh capsule invocation used `python -m` while the module had
   only a verifier function, so the command could succeed without verification.
   The actor added argparse/main/__main__. The final CLI help was independently
   executed, and existing mutation/extra-Python tests exercise the verifier.
2. The initial full-valid monitor checked only1,873sequences. The actor added the
   exact3,528,273selected-row gate alongside sequence coverage, closing mask drift.

The reviewed scientific path keeps the exact train-only primary split and
all-fit input statistics. Frozen TL3 cache producer/member/17-ancestor identities
are validated before use. The label arm receives reference data only in its
reporting monitor; the complement arm receives the fitting reference explicitly.
The reference's historical consumed-cache ancestry limitation remains declared.

Both report endpoints compose0.9reference+0.1candidate inFP32 before official
clipping. Complement loss retains original-label absolute clipped-target weights
and the declared alpha^-2 curvature scaling. Every full sequence advances carried
causal state, with row99warmup handling and official scored-mask evaluation.
Report-only evaluation restores model modes, RNG streams and checks BatchNorm
buffers. Safe-epoch state binds monitor offsets and preserves orphan report tails.
No monitored metric selects, stops or changes the fixed final epoch30endpoint.

Two ranks times batch4 times four accumulated256-rowchunks preserve the original
eight-rank/batch1 mass:8,192rows/window,25,820updates/epoch and774,600updates/fit.
Both arms/seeds have the same non-objective policy. The comparator's opt-in
homogeneous-device mode removes only UUID from comparison, retains actual UUIDs
in evidence, and explicitly requires two ranks. It does not claim trajectory or
bitwise equivalence to the original eight-rank implementation.

Independent local command, using the existing Torch2.14.1+cpu environment and
PYTHONPATH=src:

```text
python -m pytest tests/test_pr67_receiving.py tests/test_sequence_blend_objective.py tests/test_sequence_reference_cache.py tests/test_b300_sequence_models.py tests/test_b300_sequence_contracts.py -q
53 passed in8.30s
python scripts/check_training_structure.py
105files,0errors
git diff --check
clean
python -m wnn.training.sequence_source_capsule --help
CLI arguments exercised successfully
```

The actual receiver must still pass the unchanged baseline replay tolerance,
complete-source/data admission, native compiledBF16/BF16x9/DDP/Muon checks and
independently approved one/two-rank/peak profiles bound to the actual devices.
No CUDA compatibility, real-data timing, memory headroom, final score, serving
latency or public/private gain is established by this review.

Exact reviewed-file digests are in REVIEWED_SOURCE_R1.json. Later actor changes
to those files require targeted re-review before source freeze/admission.
