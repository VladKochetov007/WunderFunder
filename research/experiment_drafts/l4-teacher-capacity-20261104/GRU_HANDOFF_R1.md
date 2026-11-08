# L4 teacher capacity: GRU execution handoff R1

This packet freezes the six ready native-GRU teacher recipes from
[`CONTRACT_R1.json`](CONTRACT_R1.json). It does not start SSH, GPU work, or fits.
The independent causal-Transformer adapter remains a separate family and does
not gate the six GRUs.

## Frozen GRU cells

| GPU slot | Configuration | Seed | Fixed checkpoint |
| --- | --- | ---: | --- |
| 0 | GRU 256 × 2 control | 1001 | epoch 3 |
| 1 | GRU 256 × 2 control | 1002 | epoch 3 |
| 2 | GRU 384 × 2 | 1001 | epoch 3 |
| 3 | GRU 384 × 2 | 1002 | epoch 3 |
| 4 | GRU 512 × 1 | 1001 | epoch 3 |
| 5 | GRU 512 × 1 | 1002 | epoch 3 |

Physical UUIDs are copied from the completed PR60 L4 panel in the same slot
order. GPU 4's preexisting 1,148 MiB service allocation remains untouched. The
exact profile and fit argv vectors, paths, limits, and UUIDs are in
[`GRU_TEACHER_PANEL_R1.json`](../../../scripts/l4_teacher_capacity_20261104/GRU_TEACHER_PANEL_R1.json).

Run the six profiles with the existing source/data admission receipt:

```bash
/data2/lc-epanov_s/tabdata/.venv/bin/python \
  /data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1/source-r1/scripts/l4_teacher_capacity_20261104/runner.py \
  --panel /data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1/source-r1/scripts/l4_teacher_capacity_20261104/GRU_TEACHER_PANEL_R1.json \
  --phase profile \
  --admission-receipt /path/to/S_GRU_admission.json \
  --admission-sha256 <sha256>
```

`runner.py` uses the repository's `run_owned_command`, PID/birth/PGID receipts,
per-GPU slot locks, and the fixed 12-hour deadline with a 45-minute collection
reserve. The profile commands use 65 warmup batches and 3 timed batches, with no
validation pass. Fits require six successful profile receipts plus an independent
profile-review receipt and digest. Each cell launches at most once. The runner
does not use the legacy helper's progress callback; resource and finite-loss
review must consume the actual logs and receipts before fit admission.

## Exact numerical recipe

- `scripts/train_rnn.py`, plain GRU, native event211 inputs (raw112 plus the
  existing causal event-feature block), trained from scratch with no initializer.
- Official chronological manifest SHA256
  `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`; all
  10,325 official train sequences fit, zero valid sequences fit, and all 1,873
  official valid sequences (3,528,273 scored rows) are reported every epoch.
- `wmse`, `aux-next=0.3`, `aux-struct=0.3`, `aux-weight=0.5`, batch 32,
  chunk 1,000, four epochs, fixed epoch 3, seeded batch order, and train-eval
  over the fixed first 32 train sequences each epoch.
- Strict FP32, OneCycle schedule with 5% warmup, peak LR `1e-3`, weight decay
  `0.01`, global clip `1`, and `MuonAdamW` on the GRU gate matrices with AdamW
  on the input projection, output heads, and biases.
- Every `model_ep0.pt` through `model_ep3.pt`, `model.pt`, history, and
  provenance are retained. Full optimizer/training state is not saved.

The legacy `MuonAdamW` CLI has no exposed Newton–Schulz step or epsilon flags;
the implementation defaults are 5 steps and `1e-7`. `--muon-lr-mult` applies
to the separate `tl3-muon` optimizer, so it is intentionally omitted here.
For plain `--optim muon`, OneCycle drives both the Muon momentum and AdamW
beta1 from `0.9`; the AdamW beta2 is `0.999`. `training_dynamics.build_optimizer`
now passes `strict_fp32=True` whenever the CLI precision is FP32, so CUDA
Newton–Schulz intermediates are checked and retained as float32. The source
snapshot used for GRU profiling must include that narrow fix.

The six-cell budget is 48 CPU threads, 288 GiB aggregate owned RAM allocation,
6.5 GB maximum new output, and a 3 GB free-space floor. The enclosing eight-fit
contract still reserves two slots and profile attempts for the independent
Transformer family. Do not remove older results or unrelated files.

## Frozen student continuation design

After both teacher families finish and their fixed epoch-3 predictions pass
train-only ancestry/causality review, run four students: two target families ×
seeds 1001/1002. Keep the existing FBC seed-5 initializer, two-epoch/fixed-epoch-1
student recipe, FP32 AdamW at `1e-4`, batch 16, chunk 2,000, and the PR60 control
command unchanged apart from the declared teacher target.

For each family, construct its main cache from the existing preweighted main
pool and the newly clipped family prediction mean:

```text
T_new = 0.8 * T_old + 0.2 * 0.8125 * mean(clip(new_family_member, -2, 2))
```

`0.8125` preserves the old pool's preweight coefficient sum. The wide-GRU family
contains the four epoch-3 GRU384×2/GRU512×1 teacher predictions; the
small-Transformer family contains its two epoch-3 Transformer predictions.
Generate one main-only aggregate stream per family. Reuse the old four-family
auxiliary arrays read-only with their original `distill-alpha=0.7` and
`distill-family-weight=0.1`, and retain the FBC seed-5 initializer. Do not
materialize per-teacher full train banks or a second copy of the old auxiliary
arrays. Bind source hashes, input ancestry, split membership, cache hashes and
student argv before starting those four fits.

The frozen mixture remains a hypothesis allocation, not a promised gain. The
two seeds share exposed development validation; full-valid results must be
described accordingly.
