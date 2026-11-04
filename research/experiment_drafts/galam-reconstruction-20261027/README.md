# Source-traced GALAM reconstruction

Status: implemented and CPU-checked; **not exact numerical replication**.
No scientific fit, GPU profile, remote execution or public score was produced.
Branch: `experiment/galam-reconstruction-20261027`, based on `19383f2c`.
H/E IDs: `H-GALAM-20261027`, `E-GALAM-20261027`.

The user requires Denis Tomin's architecture to be restored exactly. The supplied
diagram resolves the visible block graph but omits original sizes and several
equations. Public primary-source search found no original code or full report.
Known source details are implemented; missing details are explicitly declared
assumptions. The four-model FiLM ensemble and historical training recipe cannot
be restored from these screenshots and are not fabricated.

## Implemented artifacts

- [Contract](CONTRACT.json): scope, chronology decision, source identities,
  deterministic verification tier and the reference-identity limitation.
- [Source evidence and fidelity map](../../unified/sources/galam-20261027/README.md):
  preserved diagram, screenshot hashes, public search, known graph and unknowns.
- [Model package](../../../src/wnn/galam/__init__.py): local LSTM memory and global
  attention/ConvMixer memory; QK RMSNorm; relative-time additive bias;
  LayerNorm in global residuals; projected h/c; channelwise sigmoid gate and
  post-gate product; decoder RMSNorm/FFN residual and final linear readout.
- [Sequence adapter](../../../src/wnn/training/galam_adapter.py): reuses the existing
  sequence training/checkpoint/scorer path, external train-only normalization,
  empty hints and explicit state detachment. Only FP32/AdamW is admitted in this
  unprofiled adaptation; this is not a claim about Denis's optimizer.
- [Three example configurations](configs/README.md): historical32→32 graph,
  Connectome112→2 adaptation, and a sequence-runtime recipe. Example dimensions,
  optimization and context are assumptions, never labelled original values.
- [Chronological manifest producer](../../../scripts/build_chronological_primary_split.py)
  and [loader](../../../src/wnn/chronological_splits.py): train-only fitting,
  full official valid evaluation, source pins and feature-component overlap
  exclusion. It does not infer chronology from seq_ix or fabricate timestamps.
- [Validation receipt](VALIDATION.json): exact checks, environment and no-fit status.
- [Independent review](INDEPENDENT_REVIEW.json): source/layout/dataflow findings,
  corrections and remaining exact-reference gaps.
- [Draft handoff](PR.md): eleven-section scientific comparison proposal. No new
  fit is dispatched by this implementation.

## Local checks actually supported

From the repository root, with the package importable and CPU Torch installed:

```bash
PYTHONPATH=src python scripts/check_galam.py --config research/experiment_drafts/galam-reconstruction-20261027/configs/historical32.example.json
PYTHONPATH=src python scripts/check_galam.py --config research/experiment_drafts/galam-reconstruction-20261027/configs/connectome112.example.json
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_galam_components.py tests/test_galam_runtime_adapter.py tests/test_chronological_primary_splits.py tests/test_campaign_splits.py tests/test_sequence_inner_splits.py tests/test_b300_sequence_contracts.py
python scripts/check_training_structure.py
```

The inspector builds weights for shape/parameter inspection only; no data, forward
pass or optimizer step is needed. CPU tests use tiny synthetic sequences and
producer fixtures. They establish implementation behavior, not historical parity,
scientific improvement, B300 throughput or actual dataset chronology.

## Receiving-host preparation; not executed

With the approved data root containing official `datasets/train.parquet`,
`datasets/valid.parquet`, `cache/chains.npz`, `cache/train_targets.npz` and
`cache/valid_targets.npz`, run the implemented producer:

```bash
PYTHONPATH=src python scripts/build_chronological_primary_split.py --root DATA_ROOT --output OUTPUT_PRIMARY_JSON
```

`DATA_ROOT` and `OUTPUT_PRIMARY_JSON` are explicitly unresolved host paths, not
commands claimed to have run. The producer validates actual metadata, complete
source coverage and train/valid component disjointness before returning a frozen
manifest. Target indexes are read without target payload arrays; checksums use
bounded streaming reads. Existing campaign caches are required and have not been
audited on the real dataset in this task.

The architecture is selectable as `model.architecture="galam"` through the
existing `scripts/train_sequence.py`; **do not launch** until a separate fit
instruction, source/data audit, exact receiver checks and real-shape profile.
Training uses per-time BatchNorm and requires at least two members per local batch,
no singleton tail and no zero-weight DDP padding. Admission rejects incompatible
membership/world-size/batch combinations before model construction. No data is
silently dropped and no dummy samples enter BatchNorm. Eight-GPU suitability and
elapsed time are unmeasured.

The adopted primary protocol and historical source limits are in
[ADR0044](../../../specs/adr/0044-chronological-primary-validation.md). Old random
fold results remain history; new chronological evidence is not retroactively
assigned to them. No stopped process is resumed.
