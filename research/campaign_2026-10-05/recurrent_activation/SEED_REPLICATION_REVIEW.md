# Independent review: E475 seed wrapper

## Decision

**PASS for wrapper seed/order behavior and source-reuse design.** The CPU
fixture mocks the fit; it did not train, score, or launch CUDA work. Do not
start E475 until each run has its own content seal/resource spec and the
controller preserves the one-worker-at-a-time, verified-teardown rule.

## Seed and recipe checks

`fit_seed_replication.py` sets the imported trainer's module-level `SEED` and
replaces its `set_seed` function with a new seed-bound callable. Both are
needed: changing `SEED` alone would not alter the old function's captured
default argument. The trainer uses the module global for the NumPy fit-order
shuffle/result seed and calls `set_seed()` before constructing each model.
The wrapper therefore changes each intended training seed while preserving
the unchanged candidate model, optimizer construction, data, loss, number and
order of six arms, and 474-update recipe. It creates a new output path and
refuses to overwrite existing outputs through the underlying trainer.

I reran the mocked CPU wrapper fixture under one CPU and 2 GiB. It confirmed
that seeds 271829 and 271830 each produce a different initialization
fingerprint from 271828, that all three alpha initializations in a seed share
the intended seeded bytes, and that each seeded fit order matches NumPy's
order from that seed. Receipt hashes were `1044916d…802e81d` for 271829 and
`16de256f…1479e7c` for 271830, compared with the original seed's
`ce46099d…74459ffe`. The fixture source is
`replication_cpu_check.py` SHA `c17ef74d…afb74b135`; the final wrapper is SHA
`cbf82953…6173546`.

## Precision and admission notes

The wrapper sets the same three CUDA flags as the already-admitted E471
`fp32_policy_run.py`: matmul TF32 false, cuDNN TF32 false, and cuDNN enabled.
It directly invokes the same unchanged candidate trainer. My initial concern
that this bypassed E471's accepted wrapper was incorrect: both paths apply the
same flags and call the same trainer. This does not change E471's amended
policy. The E475 receipt records those three flags but should also preserve
E471's `float32_matmul_precision` value (`highest` in the current runtime) and
the worker spec should bind the wrapper and frozen trainer/support files.

The wrapper receipt now records and asserts `float32_matmul_precision="highest"`
in addition to the three CUDA flags, matching E471's recorded runtime policy.
The CPU fixture was rerun after this metadata-only change and retains the same
initialization/order checks. The final wrapper SHA is
`cbf82953164f7ca7541afcec1f68e2b25ddc71395994dfe559e5793be6173546`; the
fixture receipt SHA is `5ed276cf07ec19679fd76a2681d04c110fb8517eb20b2cd06c5fcb9838244804`.

The CPU fixture cannot establish CUDA precision, parity, resource use,
throughput, or fit completion. E471's existing common-policy production-shape
probe is the matching shape/model gate; full replication workers still need
the sealed 2-CPU/6-GiB-RAM/2-GiB-allocator/3-GB-observed-device limits and
separate teardown verification. All six fixed arms must run at each new seed;
retain all scores/contrasts, do not select a recipe from the first replicate,
and do not pool conditional component intervals into a training-seed CI.

Evidence: `SEED_REPLICATION_PREREG.md`, `fit_seed_replication.py`,
`replication_cpu_check.py`, `REPLICATION_CPU_CHECK.json`,
`versions/replication_cpu_fixture_pre_metadata_receipt.json`,
`fp32_policy_run.py`, and the sealed E471 fit receipt.
