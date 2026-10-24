# Receiving-host acceptance

These checks are prepared, **unrun locally**. Execute on the allocated B300
host before admitting scientific fits. Record command, source SHA, output,
versions and actual device identity. A written checklist is not a passed check.

1. Run `PYTHONPATH=src uv run --no-sync pytest -q
   tests/test_b300_sequence_models.py tests/test_b300_sequence_contracts.py`.
   These cover causal prefix invariance, chunk/row equivalence, state/reset/
   detach, canonical GRU equality, checkpointed gradients, padding, objective
   scaling, cache ancestry and malformed config rejection.
2. Run the multi-rank synthetic acceptance command documented by
   `scripts/check_sequence_ddp.py` on two allocated GPUs. Compare the globally
   normalized accumulated gradient with a single-model reference, including
   an empty rank and tail window. Save its JSON receipt.

   ```bash
   PYTHONPATH=src uv run --no-sync torchrun --standalone --nproc-per-node=2 \
     scripts/check_sequence_ddp.py \
     --output-receipt "$WNN_RUN_ROOT/ddp_acceptance.json"
   ```

3. On source-qualified real data, profile each intended config/world size with
   eight updates. Require all updates finite, FP32 model/gradient/AdamW state,
   TF32/autocast off, correct memory-growth plateau and sufficient device/host
   headroom. Confirm train/validation component disjointness and source hashes.
4. Interrupt a bounded disposable fit after one completed epoch; resume on the
   same topology and compare final tensors/predictions with an uninterrupted
   equivalent run. Prove config/data/cache/source/world-size mismatch rejection.
   Do not count this resource-only exercise as a quality replication.
5. Export one completed disposable endpoint using one and two ranks into
   separate directories. Compare all arrays and exact membership; partial
   shard failure must leave no complete manifest. Reset/A–B–A replay and
   row-vs-chunk outputs must agree within frozen numerical tolerance.
6. Recompute held-out score from immutable exported arrays with the canonical
   official scorer. Verify no row duplication, first99 exclusion, clipping,
   weighting and target order. Outer score is reporting only.

Only then approve the exact production shape receipt. Start the six-fit
transfer panel first; hints and streaming panels depend on its completed,
hash-bound teachers/caches. Do not run optional follow-ups by default.

The profile measures actual hardware, not all numerical invariants. An
`approved` flag without the linked acceptance evidence is not a scientific or
deployment validation claim.
