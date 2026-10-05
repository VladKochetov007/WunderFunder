# Independent implementation review: E478

**Decision: GO for the sealed actual-data CUDA probe only.** The six full fits remain gated on a passing probe and separate controller admission. I ran CPU tests and static checks only; no CUDA command or fit was launched.

The custom model keeps the real registered two-layer `nn.GRU` parameter container and overrides only forward recurrence. Its input embedding and GELU readout reuse E474 unchanged. For each layer it supplies the registered input/recurrent matrices and both bias vectors to the already reviewed E471 single-layer scan, feeds layer 0 outputs into layer 1, and returns stacked final states with shape `[2,B,64]`. A single alpha is applied to both candidate maps. The alpha-zero CPU comparisons verify exact state-dict keys/values/buffers and close full-model output, final state, input gradients, parameter gradients, and matched clipped MuonAdamW step against ordinary `RNNModel`.

The GPU parity path is cuDNN-safe: both fresh models remain in PyTorch's default training mode when they perform the forward and backward passes, and the probe does not call `.eval()` before gradient parity. This is necessary for cuDNN's RNN backward reserve space. Dropout is configured as zero, so training mode preserves the same GELU baseline function. The CPU tolerances, GPU probe tolerances, and optimizer-step tolerances match the frozen design; the implementation does not widen them. The CUDA probe requests the actual B4×256 shape, performs alpha-zero output/state/input-and-parameter-gradient and clipped Muon-step parity, then runs one finite, changed-parameter update for each of six seed/alpha cells.

The optimizer partition is explicit and checked as an exact, disjoint cover of all trainable tensors: Muon receives only both layers' input and recurrent matrices (`weight_ih_l0`, `weight_hh_l0`, `weight_ih_l1`, `weight_hh_l1`) with three gate blocks per matrix; AdamW receives all remaining input, head, and bias tensors. The total remains 61,058 parameters. The fit-only path loads 24 fit groups, uses the same seeded order per alpha, resets the two-layer state at each B4 batch, carries/detaches it over 79 chunks, applies the unchanged E470 `need_prediction`-weighted objective, and records the fixed E474 telemetry. CPU tests cover bounds, causal future isolation, chunk carry/reset, and finite recurrent states. The CPU preflight has six finite parameter-changing one-update cells.

The bounded CPU invariant suite passed under one CPU and a 2 GiB limit. Ruff format/lint passed, and the probe's clean `--help` import works. The root-owned CPU preflight receipt is `cpu_preflight_v1/fit_manifest.json`; it reports all six seed/alpha updates and the expected 61,058 count. A separate clean actual-data CUDA probe remains required.

Exact source identities reviewed:

- `fit_recurrent_activation_scale.py`: `3ccc2fcd0601f6a2d239bd47799561834ffd6e42517b2b2af603e03cd3b08d20`
- `gpu_probe.py`: `e11b1dce6f21e24beb32174758af950e7642d7b1ae448fa60ec3e6763da1d75a`
- `tests.py`: `1a20253e28df94cf7debca215bc593989a4ea6c5b456c2d8a584b72d1f298d91`
- `E478_DESIGN.md`: `da6b300401a9e3d78bc2b29410ec9f142ebb1b0f6da70c00b3bedf8ec05b8cad`
- Blind `SPEC_CRITIC.md`: `9a28fa18fc513997f107da17365c669cf39beef974a9832fbfd6a46efef27284`

The evaluation cohort has prior exposure. Keep all six fits and paired within-seed comparisons; results remain conditional on each initialization and those exposed components. They do not establish long-horizon gradient stability, H256 transfer, or a public forecast.
