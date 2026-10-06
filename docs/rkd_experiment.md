# Relational distillation experiment

The opt-in RKD component compares sampled within-sequence distances between the two source-cache teacher outputs and the student's existing first two outputs. Each sequence normalizes its sampled distances by its own mean before Smooth-L1. The term adds one scalar to the existing training objective; it reads no labels or label weights and adds no output head. The source target is the registered weighted teacher sum with coefficient total `0.8125`.

`src/wnn/rkd_component.py` uses at most 256 disjoint sampled pairs per sequence by default. It samples with a local `torch.Generator` seeded by `args.seed + step_count`, detaches the teacher, filters pairs through the existing boolean `valid` mask, and checks non-finite values at eligible rows. Masked values may be non-finite without affecting the loss. Zero-distance teacher pairs remain in the relation mean. A sequence contributes only when at least two eligible pairs remain and its sampled teacher geometry is not fully collapsed. No pairwise `T × T` tensor is constructed.

The trainer flag defaults to exact disablement:

```text
--distill-rkd-weight 0
```

When disabled, the existing loss object is returned unchanged by the helper and the training loop does not slice or inspect relation tensors. When enabled, it requires `--distill-cache-root` with `--distill-cache-variant source`; specialist architecture, quote auxiliary mode, reference calibration, controlled microbatch loss modes, and `ce_joint` are rejected before data loading. The integration passes raw `teach[:, sl]`, `out[..., :2]`, and `valid[:, sl]` after the existing label and auxiliary losses. Existing pointwise target blending and weights remain in place.

The treatment coefficient must be selected and recorded in the frozen campaign input manifest before a fit. Use a paired control and treatment with the same data, parent checkpoint, seed, batch order, epochs, schedule, checkpoint policy, and every other argument. Their only fit-argument difference is `--distill-rkd-weight 0` versus the single predeclared positive coefficient. Keep `--distill-rkd-pairs 256` fixed in both arms. Do not search over coefficients, seeds, epochs, or checkpoint choices. A paired interval crossing zero is inconclusive; the external fixed evaluator owns the endpoint and any decision.

For an already registered recipe, the training invocation includes:

```text
--distill-cache-root <registered-source-cache> \
--distill-cache-variant source \
--distill-rkd-weight <predeclared-positive-coefficient> \
--distill-rkd-pairs 256
```

The source cache is supplied read-only. `scripts/run_rkd_experiment.py` offers a CPU-only diagnostic over a source cache, matching `(N, 20000, 2)` float32 student predictions, and an optional `(N, 20000)` boolean validity mask. It verifies the source cache's `metadata.json` format, its recorded `0.8125` coefficient total, and the teacher NPY's declared bytes, shape, dtype, and hash. It uses NumPy memory maps and one copied chunk at a time, hashes all inputs and code, and reports only relation diagnostics. It never fits, reads labels, invokes the grader, or reports a forecast score. Its synthetic-only check can be run without research arrays:

```bash
PYTHONPATH=src WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
  WNN_MEM_MAX=1500000000 scripts/limited .venv/bin/python \
  scripts/run_rkd_experiment.py --smoke-test
```

For a registered source-cache array, use a new output path for each diagnostic:

```text
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 \
  scripts/limited .venv/bin/python scripts/run_rkd_experiment.py \
  --teacher-cache <cache-root>/main_source_train.npy \
  --student-predictions <read-only-student-predictions.npy> \
  --valid-mask <optional-validity-mask.npy> \
  --output-json <new-diagnostic.json> \
  --output-markdown <new-diagnostic.md>
```

Omit `--valid-mask` when every row is eligible.

The launch packet builder is a separate preflight step. The controller must provide the review receipt path on every invocation; this code never creates or infers an approval. The receipt may live outside the project root, while the reviewed code, protocol, input manifest, profile receipt, and command file must be regular project files. Their exact SHA-256 bytes are bound into the review and the receipt supplies the bounded resource fields. It writes a new packet atomically and refuses to replace an existing path. Packet creation does not launch a fit; controller review of the returned packet remains required.

```text
scripts/build_rkd_trial_packet.py \
  --project-root . \
  --actor-code src/wnn/rkd_component.py \
  --actor-code scripts/train_rnn.py \
  --actor-code scripts/run_rkd_experiment.py \
  --actor-code scripts/build_rkd_trial_packet.py \
  --actor-code tests/test_rkd_component.py \
  --actor-code tests/test_run_rkd_experiment.py \
  --protocol <frozen-protocol> \
  --input-manifest <registered-input-manifest> \
  --profile-receipt <profile-only-receipt> \
  --approved-commands <frozen-command-file> \
  --review-receipt <controller-review-receipt> \
  --output <new-packet-path>
```

`--actor-code` may be repeated for all code files covered by review. Its SHA-256 is framed from sorted project-relative paths, file sizes, and exact file bytes. The builder accepts only regular, nonempty project files and fails closed on a missing receipt, wrong status/version, hash mismatch, invalid resource fields, symlinked assets, or a pre-existing output. Tests use clearly synthetic receipt data and do not represent controller approval.
