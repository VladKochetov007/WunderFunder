# Structured-state actor artifacts

Card B model plus its single-half-life initialization control. Width96, two
blocks, 64 scalar/32 rotation-pair modes per block; flat carried state tuple,
1024FP32 bytes per sequence. Both arms have identical parameter shapes, phase
buffers and compute; only initial raw half-life values differ. Multi/single
comparison tests initialization. The architecture is exploratory; no competition
data, fit, score, promotion or native implementation is claimed here.

`preregistration.md` commits the implementation. `arithmetic_amendment_v4.md`
records the independent critic's long-horizon failure and the coefficient fix.
`model.py` provides FFT training, independent real recurrence, true `stream_step`
and explicit `reset`. `smoke.py` checks causality/stream/chunk/gradient/stability
and CPU performance. Original code/checkpoints/results remain preserved.

## CPU reproduction

From repository root:

```bash
research/campaign_2026-09-30/structured_state/run_smoke.sh fresh_label
```

The runner sets CUDA_VISIBLE_DEVICES empty and campaign slice/CPUQuota100%/
MemoryMax2G/no swap/one thread/runtime300s, writes command/log/exit/results and
synthetic-only checkpoint. Fresh labels preserve previous evidence.

`integration_cpu_check.py --label fresh_label` checks the proposed factory from
AST without editing live trainer files, strict checkpoint reconstruction, actual
trainer FastEMA/forward_chunk/detach_state, inherited initialization metadata,
and exact head-only freeze behavior. Run it through the same limited CPU scope.

## Integration and prediction

`integration.patch` is **unapplied** and its source/base/patch hashes are in
`integration_manifest.json`. It adds `src/wnn/structured_state.py` and small
trainer factory/parser/inheritance/provenance changes. `src/wnn/nn.py`, scoring,
launcher, native and prediction code were not edited. Generate a fresh patch with
`build_integration_patch.py` if the active trainer changes; review its new base
hash before application, after clean controls and independent grading finish.

Training additions: `--arch structured --hidden96 --layers2 --dropout0
--enc linear --decay-init multi|single` (CLI actual syntax uses spaces, e.g.
`--hidden 96`). Do not initialize from incompatible GRU checkpoints.
Fixed4epochs means epoch3, using `--checkpoint-policy fixed --checkpoint-epoch 3
--no-eval-during-fit`, empty freeze, same clean fold0/data/seed/aux/EMA recipe.
Both arms must complete all updates within1080GPU seconds each or the comparison
is inconclusive. Controller owns exact commands, GPU lease and launch.

Head-only future FT uses `--freeze inp,blocks` (`--head-freeze inp,blocks` through
ft_gclp.sh). Legacy `inp,gru` is rejected for this architecture. Unfreezing the
second FT stage remains explicit. No head/auxiliary slicing during fitting.

Checkpoint reconstruction uses stored `args.arch=structured`, hidden96/layers2,
`args.decay_init`, `n_in`, and `head.2.weight.shape[0]` for n_out, then strict
state-dict load. `mean/std`, frozen phase grids and learned raw half-lives all
serialize. Provenance adds model_source_sha256; retain trainer/feature/split/
propensity hashes and checkpoint bytes in the root durable job record.

Existing predict_run.py imports trainer.make_model and already reconstructs
n_out from head.2.weight, so the factory patch supplies compatibility. Existing
predict builds full causal inputs once, resets to None at each independent batch,
passes carried flat tuple across chunk4000, and returns only the first2 outputs.
The model overrides the caller's outer AMP to keep all state arithmetic FP32.
EMA teacher has a separate carried state and resets with student at each batch.
State never receives sequence IDs, targets, scoring masks or future features.

Prediction sidecars currently record checkpoint/prediction hashes, row groups,
epoch and split hash. For this screen also retain architecture, decay_init,
model source hash, input feature/preprocessing flags, output-head layout and
prediction chunk length; use checkpoint args/provenance as the authoritative
reconstruction record. Keep final deployment coefficients/activation arithmetic
explicit, with sequence-boundary reset of both blocks and causal feature state.

## Root-owned GPU profile, prepared only

`gpu_smoke.py` and `gpu_smoke_preregistration.md` are ready for root to launch
after controls and independent grade. Two sequential durable runs:

```bash
.venv/bin/python research/campaign_2026-09-30/structured_state/gpu_smoke.py \
  --decay-init multi \
  --output research/campaign_2026-09-30/structured_state/gpu_smoke_multi.json
```

Repeat with single and a distinct output. This is batch32/full20k/chunk2000,
input211/output23, FP16 outer AMP+GradScaler with model FP32, actual FastEMA
teacher and losses matching shape/weight structure. JSON and synthetic checkpoint
are exclusive writes; root captures log and runs continuous VRAM monitoring.
Stage samples miss transient peaks, and the synthetic core profile excludes
data-loader/feature-construction intermediates. GPU fit requires observed total
VRAM <=5,000,000,000bytes; a smaller batch is not a permitted workaround.

No torch.compile is needed: FFT and dense operations already use optimized
kernels, with O(T log T) work rather than an unrolled Python recurrent training
graph or T² attention. One-vCPU native runtime and GPU memory remain unmeasured.
Native scalar/2x2 inference is required only after score-gated promotion.
