# E463 TPU v4 loss-reduction diagnostic candidate

This is a source-audited, offline-assembled candidate for the next bounded TPU
diagnostic. The packet under `packet/` has not received independent transport
review or fresh quota admission; it was not pushed or launched and carries no
fit or scoring authorization. A controller must select it, review the worker
implementation and transport map, and set fresh quota before provider work.

## Why this stage follows V3

V3 passed an actual eight-chip TPU v5e systems check with a synthetic
batch-16/chunk-2000 update and exact checkpoint export/reload. It used the same
synthetic local batch on every replica and called
`rnn.weighted_fixed_target_mse`; it did not exercise the frozen four-component
combo objective or unequal shard masses. Its `lax.pmean` of replica gradients
therefore does not establish that independently normalized components reduce
to the global-batch objective.

The frozen own E503 checkpoint has no incoming EMA checkpoint state. The
verified `wunder-funder` mount observed in V3 contained only `train.parquet`,
`valid.parquet`, and `valid_mask.parquet`; it did not contain fold membership,
teacher prediction caches, or propensity caches required by
`load_sequence_supervision`. This diagnostic consequently uses deterministic
synthetic inputs, labels, all three structural targets, next-change targets,
weights, and fixed EMA predictions. It tests mechanics only and must not be
reported as a real-data profile, recipe replication, fit, or score.

## Frozen equation and source binding

The reference implementation in `combo_loss_reference.py` independently
computes raw sums for the exact source equation in
`src/wnn_jax/bridge.py::combo_sbd_loss`:

```text
L = N_primary / max(D_primary, 1e-8)
  + N_consistency / max(D_row, 1e-8)
  + 0.3 * N_structural / max(D_row, 1e-8)
  + 0.3 * N_next / max(D_next, 1e-8)
```

`D_primary` is the sum of `abs(clip(labels, -2, 2)) * main_weights` over both
primary channels. `D_row` is the sum of `main_weights`; the EMA term sums two
channels and the structural term sums all three heads against this same row
mass, matching the frozen implementation. `D_next` is the sum of
`next_weights`. A distributed implementation must sum each raw component
numerator and its matching denominator over all eight replicas before
normalizing. Averaging replica-local normalized losses is the deliberate
negative control.

The packet source must verify the following pinned direct-tree identity before
importing JAX:

| Source/input | SHA-256 |
|---|---|
| E463 source manifest | `851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb` |
| `src/wnn_jax/rnn.py` | `294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc` |
| `src/wnn_jax/bridge.py` | `788e83eb24759e63c1dc877252ae4ad4a400f1f7f64af2dce8611e844d1990bb` |
| `src/wnn/nn.py` | `388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3` |
| own E503 `model_ep3.pt` | `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e` |
| bound `model_ep3.jax.npz` | `677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a` |
| NPZ sidecar | `3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668` |

The direct source dataset is private and offline, with TPU enabled and GPU
disabled. The packet metadata explicitly sets `enable_internet` to false and
mounts only that pinned source dataset; it omits the unrelated raw-data mount.
The source manifest plus every member's size/hash and exact member set are the
transport identity; the declared archive digest is not treated as an observed
file.

## Device-stage contract in the assembled candidate

1. Reuse the V3 host, direct-tree, checkpoint, eight-device, and HBM admission
   guards without changing their pinned identities. The source manifest and
   all 17 direct-tree files must still pass identity checks. Do not require the
   raw dataset mount because this synthetic diagnostic does not consume it.
2. Load the exact E503 NPZ and run `rnn.forward_chunk` at global batch 16,
   local batch 2 per chip, sequence chunk 2,000, input width 211, hidden width
   256, two layers, and 23 outputs.
3. Give each replica distinct seeded synthetic inputs and supervision. Scale
   row masses differently by replica, make one replica's `next_weights` all
   zero while other replicas have positive next mass, and verify that total
   denominators remain positive. Keep the incoming two-channel EMA predictions
   fixed and explicitly synthetic.
4. Independently compute the concatenated full-batch loss from the model
   outputs with `combo_loss_reference.py`. Compare that scalar with
   `bridge.combo_sbd_loss` at `rtol=1e-4, atol=1e-5`.
5. Compute a full-batch reference gradient with the frozen source loss. In the
   distributed path, differentiate each replica's local raw numerators divided
   by the already-global denominators, then `lax.psum` the resulting gradient
   trees. Compare every trainable leaf against the full-batch gradient at
   `rtol=3e-4, atol=3e-5`; record max absolute and relative L2 errors.
6. Run the negative control that `pmean`s gradients of replica-locally
   normalized `combo_sbd_loss`. It must differ from the full-batch gradient by
   relative L2 at least `1e-3`; otherwise the deliberately unequal-mass fixture
   did not detect the invalid reduction.
7. After one compile/warmup update, time three synchronized full-shape
   synthetic updates. Require finite outputs, recurrent states, component sums,
   gradients, and updated parameters. Record per-replica denominators, all four
   loss components, update times, actual JAX/backend/device inventory, and
   pre/post HBM usage. Keep 20% free HBM before execution and peak HBM at or
   below 80% of capacity.
8. Do not save or export diagnostic-updated weights, run validation, produce a
   metric, or transition into a fit. These checks establish equation/reducer
   mechanics and a full-shape synthetic profile only.

The full-batch gradient pass uses all 16 sequences on one TPU chip. The
independent packet reviewer must check its HBM cost, the pmap collective
semantics, and all tolerance choices before a fresh quota admission. The
attempt quota and provider timeout remain unset until the controller selects
this stage and refreshes Kaggle status.

## Checks already run

`tests/test_combo_loss_reference.py` checks that summing raw component
sufficient statistics across deliberately unequal shards matches the
concatenated NumPy objective and that the mean of locally normalized shard
losses produces a detectable error. It passed with `2 passed`; Ruff passed for
both packet code files and tests, and `py_compile` passed for the assembled
worker and host oracle. Offline packet self-validation is recorded alongside
the build manifest. This host check does not import JAX or claim TPU execution.
