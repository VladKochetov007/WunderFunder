I'll read the workspace source only and derive a runnable TCN teacher specification from the actual model, stream, and training paths. No fits, edits, GPU, or network.The launchable train CLI is blocked. The model math is fully determined in `src/wnn/nn.py` and `src/wnn/tcn_ref.py`; the trainer chunk loop, optimizer, AMP, split manifest, and `event_content` import are not in this workspace.

## 1. Source-bound assumptions and correctness risks

Candidate that the supplied factory actually builds: `TCNModel(n_in, hidden=128, layers=12, n_out=2, mix=True)` via `TRAINER_SECTIONS.txt:16-24` and `:66-72`. Constraints enforced there: `dropout==0`, `enc=="linear"`. Head is hard-coded `Linear → GELU → Linear` (`nn.py:188-190`). `--readout-activation` does not touch it.

Derived from the ops, not the docstrings:

- Block `i` (`nn.py:146-155`): `a = GELU(LayerNorm(x))` over channels only, then `y = Linear([a(t−d), a(t)])` with `d = 2^i`, optional `GELU+Linear` mix, residual `x+y`. Returned state is `ext[:, t:].clone()` of length `d`.
- Dilations: `1,2,4,…,2048`.
- Receptive field: `1 + Σ d_i = 2^12 = 4096` rows. Output at `t` depends on input rows `[t−4095, t]`.
- State: `Σ d_i = 4095` post-GELU-LN vectors, `4095×128 = 524160` elements. fp32 `2096640` bytes per sequence; batch 16 fp32 state `33546240` bytes. This is the ring contract in `tcn_ref.py:57-72` (`slot = t % d`, zeros while `t < d`).
- Parameters, mix on: `3 C^2 + 4 C = 49664` per block, `12× = 595968`, plus TCN norm `256`, head `33154`, input `128(n_in+1)`. Total `629506 + 128 n_in`. `n_in` is not in the workspace.
- Chunk `2000` is shorter than the last tap `2048` by 48 rows. Exact full-sequence equality holds only when tails are passed in order. At global `t=4000` the past tap reads `a(1952)`, which was produced in chunk 0 and only survives inside the length-2048 tail.

`LayerNorm` does not see the tail. History longer than a chunk changes `y`, not `a`, and not the current row's norm statistics. Left context is a zero `a`-tail (`nn.py:149-150`), which matches `TCNStream` and does not match prepending raw zero rows once any LayerNorm bias is nonzero. At init, bias is 0 so those two pads coincide; after training they do not.

`tail.to(a.dtype)` (`nn.py:151`) recasts carried history at every chunk boundary. Under fp16 that can make a stateful chunked forward differ from one full-sequence forward. `clone()` drops the extended storage (`nn.py:155`); it does not detach. No detach exists on this path in the supplied source. `PooledCorrMemory` detach is a different loss (`nn.py:348-349`). The recipe loss is per-chunk `weighted_mse_loss` (`nn.py:353-356`): targets clipped to ±2, predictions not clipped, weight `|clip(y)|`, denominator is the chunk weight sum. Summing those chunk losses is not the full-sequence WMSE. Official scoring clips both sides (`gmetric.py:15`, `METRIC_CLIP=2`, `METRIC_ID=global_wp_clip2_v1`). Leave both clips alone.

`valid = steps >= WARMUP` (`TRAINER_SECTIONS.txt:270`) still runs those rows through the conv. `gmetric.py:16` sets `WARMUP=99`. The trainer import of that name is not in the excerpts. Masking to the receptive field would change the loss. `main_row_weights` is not in the workspace; the objective uses `main_valid[:, sl]`, not raw `valid`.

Missing source that blocks a fit:

- `scripts/train_rnn.py` forward between the supplied regions (excerpt ends in the batch loop, resumes at the loss). State pass, detach, AMP, AdamW, OneCycle, and `n_out` are there. `args.seed` is applied to `EpochOrderSampler` / `WeightedRandomSampler` only in the excerpts; parameter-init seeding is not shown.
- Argparse for lr, weight decay, OneCycle, fp16, `--seed`, `--chunk`, `--batch`, `--epochs`, `--loss`, `--raw-only`, `--no-eval-during-fit`, `--max-batches`, split path. Attributes `args.chunk`, `args.epochs`, `args.stop_after_epoch`, `args.loss=="wmse"`, `args.raw_only`, `args.no_eval_during_fit`, `args.max_batches`, `args.batch_order=="seeded"` are used. Flag spellings for those are not in the supplied `add_argument` block.
- `research/campaign_2026-09-29/artifacts/splits/fold0.json`. Fit membership, the corrupt train id, and `10606/10607` are not verifiable here. Do not treat train+valid as a fresh holdout.
- `src/wnn/event_content.py`. `nn.py:9` imports `ENDPOINT` at module import, so `import wnn.nn` fails in this tree even though `TCNModel` does not use it.
- `build_inputs`, `RAW_ONLY`, `N_FEATURES`, padding, and sequence lengths. `token_groups` (`nn.py:274-277`) documents 112 raw + 99 event columns and stops there. If every fit sequence has length ≤2048, block 11's past tap is identically zero and `conv.weight[:, :C]` for that block gets no past-tap gradient.
- Distill cache writer, serve/combination module, `four_family_weighted_mse`.

`stop_after_epoch` keeps the LR schedule on `--epochs` (`TRAINER_SECTIONS.txt:221-222`). `--epochs 4` and a longer run stopped after zero-based epoch 3 are different OneCycle trajectories. The scheduler source is absent, so those two flags cannot be equated.

## 2. Candidate CLI

No launch command. The only exact, source-backed selector is:

```text
--arch tcn --hidden 128 --layers 12 --tcn-mix 1 --dropout 0
```

Omit `--enc` (default linear). Omit `--freeze`. Leave the shown aux defaults at 0 and `--aux` empty. Do not pass `--propensity`. `TCN` provenance hashes `src/wnn/nn.py` (`TRAINER_SECTIONS.txt:181-182`).

Recipe values with no flag definition in the supplied sections, so they are not a CLI: seed `7`, chunk `2000`, batch `16`, loss value `wmse`, AdamW `lr=0.001`, `wd=0.01`, OneCycle, fp16, raw features, 4 zero-based epochs, no eval, no checkpoint selection. `--no-eval-during-fit` only skips the valid loader (`TRAINER_SECTIONS.txt:126`). Checkpoint ranking is not shown.

When the missing trainer section is read, bind those recipe values only to flags that exist there. Keep per-chunk `objective(out[..., :2], y[:, sl], main_valid[:, sl])` (`TRAINER_SECTIONS.txt:356`). Fit rows are the manifest's train+valid allowed groups, ceiling 10607 sequences, one corrupt train excluded, excluding exposed evaluated fold0 components.

## 3. Smallest tests, then the real-shape profile

Import barrier: tests need `wnn.event_content` on the path. They do not need the manifest, a fit, or the GPU. This task envelope is 1 CPU, 1.5 GB, 10 minutes: `WNN_THREADS=1`. Do not start them beside a GPU fit.

Construct `TCNModel(n_in, 128, 12, mix=True)` with a fixed tensor seed. `n_in` can be 112 for these tests only; that does not establish the train width. Assert `[b.dilation for b in model.tcn.blocks] == [2**i for i in range(12)]` and `sum == 4095`.

Use `B=1`, `T=6000`, chunk `2000`, fp32, one sequence, three chunks. That crosses the 2048-tap twice (`a(1952)` lands in chunk 2's tail).

- Forward carry: stateful chunks with no detach match one full forward on every row.
- Prefix: full `[:, :t]` matches a separate forward of the prefix for `t ∈ {1, 99, 2000, 2047, 2048, 2049, 4095, 4096}`.
- Reset: `TCNStream` in float64 matches `TCNModel` row-wise; `reset()` then `step(row0)` matches a fresh stream.
- Norm vs history: two different tails, same `x` → identical `a`, different block output.
- Pad: after setting a block `norm.bias` nonzero, `h=None` differs from prepending raw zeros. The zero `a`-tail is the implementation.
- Gradients, same concatenated predictions, one `weighted_mse_loss`: no-detach chunked backward matches full backward. Detach-between-chunks keeps the forward match and cuts the graph into `a(1952)`; `conv.weight[:, :C]` can still be nonzero because the detached tail is a feature. Past half is `weight[:, :C]`, now half `weight[:, C:]` (`tcn_ref.py:45-48`).
- Loss identity: sum of per-chunk `weighted_mse_loss` differs from one full-sequence call when chunk weight sums differ. That difference is the trainer objective. Do not retarget the loss to remove it.

Profile gate, only after those pass, only on a free GPU. Healthy fits stay untouched. `configure_torch` (`resources.py:46-66`) plus `WNN_GPU_MEMORY_BYTES` as the incumbent integer byte ceiling (`allocator_fraction` rejects a `GB` suffix and does not cap the byte path by `GPU_FRACTION`). Default fraction is `min(0.75, 0.70)` and is not 4.3 GB. The allocator cap is per process; it does not reserve the device.

One real step: real `n_in`, batch 16, chunk 2000, the trainer's fp16 and AdamW if the missing section defines them, forward, per-chunk WMSE, backward, `optimizer.step`. `torch.cuda.reset_peak_memory_stats` before the step; `synchronize` before and after; record `max_memory_allocated`, RSS, and rows/s = `batch * chunk / dt`. If a `DutyCycle` is on, `synchronize` then `tick` (`resources.py:51-52`). Also log min, median, and max fit-sequence length. If max length ≤2048, the longest tap is structurally zero on this data.

If that step OOMs, the only fallback is batch 8, same chunk, width, depth, and mix. Do not drop blocks to make the profile pass.

## 4. Conditional value vs a declared control

Standalone WP is not the keep/drop test. No score and no public forecast from this pass.

Declared control, both branches present in the loss section: same student, same seed, same chunk, same fit rows, `teach is None`, so `loss = objective(...)` with `weighted_mse_loss`.

Treatment: same student with this TCN frozen, its causal per-sequence predictions written in row order into the existing teach channels `y[..., 2:4]` (`TRAINER_SECTIONS.txt:305-332`). Blend stays inside that branch, including `.clamp(-2, 2)`. Reset state on every sequence (`h=None` or `TCNStream.reset()`). Warmup rows stay in the forward. The cache writer is not in the workspace, so this comparison cannot be run from the supplied tree. Four-family layout, when used, is 12 channels: labels, teach, eight family targets (`TRAINER_SECTIONS.txt:263-268`). A weak standalone TCN still belongs in that heterogeneous slot; nothing in source supports dropping the family.

What would count as conditional value: the student's parameter update on the same batches moves relative to the control. What would not: a new `global_wp` on the exposed fold0 eval components.

## 5. Transferable component

The dyadic two-tap ring, independent of whether the head is weak.

For each scale `τ_i = 2^i`, memory is a ring of `τ_i` values of `a = GELU(LayerNorm(x))`, updated as a linear function of `(ring[t mod τ_i], a_t)` plus an optional pointwise mix, then a residual add (`tcn_ref.py:66-73`). Reset zeros the rings and `t`. Total state `2^L−1` rows, field `2^L` rows. Norms do not cross time or scales. The head then mixes every scale into one vector (`nn.py:194-195`), so the 2-d prediction drops the timescale identity.

The existing distill path can carry the 2-d head only. It cannot carry the twelve pre-residual `y` streams. Those `y` values are already computed inside `TCNStream.step` before `x = x + y`. That per-scale two-tap is the piece to keep for a native port or a later student input even if `TCNModel` itself loses to the recurrent models. No new module: the contract is `CausalConvBlock` plus `TCNStream`.
