# Throughput panel: training pipeline (2026-09-27)

Lens: performance engineer. Goal: more hypotheses per hour from `scripts/train_rnn.py` on one RTX 3060 Laptop
(sm_86, 30 SMs, ~5.7 GB usable) + 10 CPU cores. Recipe under test: REC of q22 (train+valid fold 0, prop_v1^2+0.03,
valid-true-mask 0.03, aux-struct 0.3, aux-next 0.3, wmse, GRU 128x2, B32 x C2000). Design only: no GPU was used and no
repo file other than this report was written. CPU probes ran under `scripts/limited` (WNN_CPU_PCT=6, 1 thread, 3G) on
three row groups (train 0, train 5000, valid 0). Probe scripts are in the session scratchpad.

## 0. Evidence
| id | finding | source |
|---|---|---|
| F1 | The same recipe runs 43.6-45.2 seq/s as a single job at WNN_GPU_FRACTION=0.9 with 2 workers (ab46_ref_f0, ab46_n25/n50), and 12.9-15.5 seq/s as one of two jobs at 0.48 (ab61_*). The two-job schedule gives ~30 seq/s in aggregate against ~44 for one job | artifacts/logs |
| F2 | The running jobs have WNN_CPU_PCT=30, so 4 threads and **2** loader workers (`cpu_threads()//2` caps WNN_LOADER_WORKERS=3), and a 2.9 GB memory fraction | /proc/pid/environ, resources.py:63 |
| F3 | Parquet uses ZSTD + BYTE_STREAM_SPLIT with no dictionary pages: 3.9 MB compressed / 9.3 MB raw per row group. Train is 10,607 groups (29.2 GB), valid 1,873 (5.6 GB) | metadata |
| F4 | `SequenceFile.read` takes 49-65 ms/seq on one core: `read_row_group` 21-28 ms, then the 112-column strided fill (data.py:101-102) 13.5-20 ms. `np.stack(cols, 0)` (1.4 ms) plus `np.ascontiguousarray(.T)` (3.2 ms) gives **bit-identical** features | probe |
| F5 | No NaN, inf or -0.0. Distinct values per column range 14..19,998 (volumes 6-15k, a-columns up to 20k), and the union of 3 groups reaches 60k. Per-sequence uint16 dictionary codes are lossless (bit-exact round trip) but take 5.3-7.3 MB against 8.96 MB, and numpy decode costs 16-25 ms/seq | probe |
| F6 | A chunked twin of `event_features_torch` carries state per sequence: the last event rows `(B,3)` and the held i0 price deltas `(B,22)`, where held = where(last_px>=s, gather(d_chunk, last_px-s), carry). It is **bit-identical** to the full-sequence features for C in {500, 1999, 2000, 4000} | prototype, CPU |
| F7 | Syncs per B32 batch: 10x found_inf `.item()` (grad_scaler.py:371), 10x `torch.cuda.synchronize()` (train_rnn.py:514), `scored.to(device)` blocking (:432), `has_mask.any()` (:434), and `torch.tensor(QUANTA, device=)` in structure_labels (outcomes.py:100). EMA runs add 20x `n_averaged` checks (swa_utils.py:310,318). Each sync drains the queue, so the GPU idles while Python enqueues the next chunk. The waiting thread spins on a core of the cgroup quota | code |
| F8 | `DutyCycle.tick` counts all wall time since the last tick as busy, including loader waits, the pageable 287 MB H2D and label building, then sleeps (1-f)/f of it | resources.py:35 |
| F9 | Torch 2.14/cuDNN 9.24 selects `CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H` for GRU/LSTM with hidden <=192 forward and **<=128 backward** (ATen cudnn/RNN.cpp `use_rnn_persist_small_h`, since PR #73211). Our h128 GRU and the h64 propensity GRU therefore run persistent kernels. **h>128 falls back to the per-timestep algorithm in backward**, so wider arms train several times slower | pytorch source |

Reading: py-spy puts 69% of active samples in `.item()`, yet sampled GPU utilization is ~6%. The two readings can agree
only if the GPU is mostly idle between short bursts of persistent kernels, with time going to the loader, syncs, the
pageable H2D and duty sleeps. F1 fits a loader bound at 2 workers (~22 seq/s per worker at 40-50 ms/seq). F1 under a
time-slicing model implies ~0.35-0.45 s of GPU work per B32 batch. Step 0 below settles this in 15 minutes.

## 1. Common protocol
- **Refactor first (T0 vs HEAD):** move train_rnn.py:417-452 into `prepare_batch()` and :454-515 into `train_chunk()`.
  Every item then lands as a flag on those functions, default off, and flips to on only after its parity and bench pass.
  This allows A/B inside one process.
- **Parity harness** `scripts/parity_step.py`: CUBLAS_WORKSPACE_CONFIG=:4096:8,
  `use_deterministic_algorithms(True, warn_only=True)`, cudnn.deterministic, benchmark off, seed 3, REC args, fold-0
  order, 0 workers, 4 batches (40 steps). Per batch it records sha256 of inputs (fp32 bytes), of every label tensor and
  of main_valid. Per step it records the loss (float64 repr), the pre-clip grad norm and the scaler scale. At the end
  it records sha256 of the model, optimizer and EMA state. Run the reference twice to get the floor (expected exactly 0).
  **T0** = the candidate differs from the reference by no more than the floor. **T1** = the stated tolerances.
  **T2** = a recipe change: quality A/B on chain folds 0 and 1 with 2 seeds via fold_compare.py, logged in the ledger.
- **Bench:** add `--max-batches N --no-eval` to train_rnn.py and have it print a `BENCH` line with seq/s, loader-wait
  ms/batch, GPU-busy ms/batch (CUDA events), peak GPU MB and worker ms/seq (timed in `__getitem__`). Run REC, N=45,
  dropping the first 5 batches, with WNN_GPU_FRACTION=0.9 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=60 WNN_MEM_MAX=15G, no other
  GPU compute apps, and `nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,temperature.gpu,power.draw -lms 500`
  logged. Interleave ABABAB. Accept when the median ratio clears the item's lower bound and all 3 pairs agree in sign.
  Discard a pair when the median SM clock differs by >10% between its arms. Once per item, run
  `torch.cuda.set_sync_debug_mode("warn")` over 3 batches to count syncs.
- **Fix `bench_step.py` before trusting it.** Its docstring promises aux losses, but it computes only wmse (no
  structure/next-change labels, no valid-true-mask path). `e[4]` is recorded before the last chunk's optimizer step, so
  "chunks" includes 9 optimizer steps and "opt" only 1. It has no loader and no duty cycle. Make it call
  `prepare_batch`/`train_chunk`.

## 2. Ranked plan (speedups are per job on the REC recipe; cumulative in section 4)
**S0. Measure (15 min, gating).** Run the fixed bench_step (GPU ms per stage, B in {32,64,128} x C in {2000,500}), one
`torch.profiler` trace of a batch (kernels per chunk, kernel-time/wall inside a chunk, top-10 kernels), and the BENCH
run at 2 vs 4 workers. This decides whether REC is loader-bound, host-bound or GPU-bound, and gates E, H and J.

**A. One job at a time, 4 workers (config only). x1.3-1.6 per job over 44 seq/s, x3.5-4.5 over 15.** Two time-sliced
contexts at f=0.48 lose to one at 0.9 (F1). The best single-job run still had 2 workers (F2), so it is likely
loader-bound. Change: queue scripts run jobs serially with the env of section 1. Parity: none needed (no numerics
touched). Bench: BENCH 2 vs 4 workers, then aggregate seq/s against today's two-job mode over 10 minutes.

**B. Sync-free step. x1.15-1.35.** It removes ~24 host round trips and a blocking 50-90 ms pageable copy per batch,
stops the duty cycle charging host time as GPU time, and frees the spinning core.
- (1) `AdamW(..., fused=True)` at :382: GradScaler hands found_inf to the fused kernel (grad_scaler.py:418-469), which skips the step and rolls back `step` on-device, with no `.item()`.
- (2) New `GpuDutyCycle(f, depth=2)` in resources.py, replacing :514-515: `torch.cuda.Event(enable_timing=True, blocking=True)` at step start/end, <=2 steps in flight (`synchronize()` on the oldest yields instead of spinning), busy = sum of `elapsed_time`, sleep only while busy/f > wall.
- (3) Cache QUANTA and class-weight tensors per device once; compute `has_mask = scored.any(1)` on the CPU; move `scored` with `non_blocking=True`.
- (4) Replace `ema_model.update_parameters` with a Python counter plus `torch._foreach_lerp_(ema, cur, 1-decay)`, the same op as `get_ema_multi_avg_fn` (bitwise equal; the first call copies).
- (5) `pin_memory=True, persistent_workers=True` on both loaders (:336, :338).
- Parity: two commits. (2)-(5) is T0 (harness equal) with zero sync warnings in the chunk loop. (1) is T1: one step from an identical state gives max|dtheta| <= 1e-7; over 40 steps param and loss-trace rel diff <= 1e-5; a forced-overflow test (init_scale=2**40) gives identical skip pattern, scale trace and `state["step"]`.
- Bench: ABAB, plus the sync count.

**C. Loader transpose. Worker CPU -25-35%, so +30-45% seq/s per worker** (F4). This matters whenever S0 shows loader
waits. Change data.py:100-102 to
`features = np.ascontiguousarray(np.stack([table.column(n).to_numpy() for n in FEATURE_COLUMNS], 0).T)`.
Parity: T0 on the CPU via `np.array_equal(new.view(np.uint32), old.view(np.uint32))` over 20 row groups of both splits
(already equal on train rg 0). Bench: a single-thread timing loop over 20 groups old vs new, then BENCH worker ms/seq.

**D. Propensity cache. x1.08-1.2** (it drops a 20k-step GRU-64 forward and a 540 MB normalization pass per batch).
- New `scripts/cache_propensity.py --prop prop_v1` writes float32 memmaps `cache/prop_v1_{train,valid}.npy` (groups, 20000; 849 + 150 MB) via the live path (build_inputs -> propensity, B=32, chunk 4000, last batch padded to 32), keyed by sha256 of model.pt. `Sequences` returns the p row; :426 becomes `p.to(dev, non_blocking=True) ** power + eps`. Refuse the cache under `--raw-only` (live propensity sees zeroed event features there).
- Parity: T0 expected (rows are independent at an identical batch shape): `torch.equal(cache[idx], live)` on 4 batches of different composition. Fallback T1: max|dp| <= 1e-3 and weighted main-loss rel diff <= 1e-4, against the floor live(batch) vs live(permuted batch).

**F. Chunked build_inputs (enabler for B>=64). x1.0 at B32; required for G.** Full-batch inputs at B128 need 1.15 GB raw
+ 2.16 GB inputs + ~2 GB temporaries, which OOMs at 5.7 GB. The chunked path needs 1.15 GB + 54 MB per chunk.
- Change: `event_features_chunk(x, s, e, carry)` per F6, inputs built inside the chunk loop. Propensity must come from D (the valid-true-mask rescale needs whole-sequence sums before the loop). ONLINE_NORM stays on the full path (a cumsum with carry is not bitwise).
- Parity: T0 via `torch.equal(cat(chunks), build_inputs(x))` on 2 GPU batches, then the harness at B32 x C2000.

**G. B128 x C500 (ml_lead X5; recipe change, T2). x1.8-2.5.** It keeps the same 7,500 optimizer steps and 64k rows per
step. Per-step cost of the persistent GRU is near-flat from B32 to B128 at h128, so the GRU cost per sequence falls
about 4x. Row-wise work per chunk is unchanged.
- Change: `--batch 128 --chunk 500`, lr {1e-3, 2e-3}, 5 workers (raise the `max_workers=4` cap in resources.py:58), prefetch 1; ~7 GB of pinned + shm batches in flight. `drop_last` drops 107 sequences per epoch instead of 11; fit_normalization sees 256 sequences instead of 64. Needs D and F.
- Parity: T2 (folds 0 and 1, 2 seeds, full vfold). Falsifier: speedup < 1.5x or delta < 0 on both folds. Bench: the S0 sweep, fitting t_chunk = a*C + b*B*C + c, then BENCH.

**E. EMA teacher on a side stream (EMA recipes only). x1.0-1.3.** ab56 (EMA) ran 30 seq/s against 44 for ab46, so the
teacher forward is ~30%. The student and teacher forwards of a chunk are independent. Run the teacher on
`torch.cuda.Stream()` after `wait_stream(main)`, then join before the consistency loss, with `record_stream` on shared
tensors. Two persistent kernels overlap only if their CTAs co-reside, so S0's profile decides. Parity: T0 (torch.equal
on out_ema and loss).

**H. MPS for 2 concurrent B32 arms (recipe unchanged). Aggregate x1.0-1.7.** Use
`nvidia-cuda-mps-control -d` with CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY under ~/.cache/mps (user-level;
the binary exists). It pays only if S0 shows low SM occupancy. The 90% rule would then need a device-level duty (read
`utilization.gpu` via an `nvidia-smi -lms 500` reader thread), because per-process event time double-counts overlap.
**User decision:** define the cap as device busy time. Parity: T0 (the harness under MPS equals the harness without
it). Bench: 2 jobs concurrent vs serial, aggregate seq/s. Risks: MPS on GeForce is not officially supported, and a
fault in one client can kill the other.

**I. torch.compile of per-chunk losses/labels and the mlp/conv/tokens encoders. x1.03-1.08 at B32, x1.1-1.15 at
B128 x C500** (row-wise share ~20%), more with `enc=tokens`. The GRU graph-breaks (`_dynamo.config.allow_rnn=False`),
so compile the pure function (out, label slices, weights) -> loss with static shapes. Parity: T1, loss rel <= 1e-6 and
grad max rel <= 1e-4.

**J. CUDA graphs for the chunk step (gated). x1.0-1.3.** Pursue only if S0 shows kernel-time/wall < 0.7 inside a
chunk. It needs fused+capturable AdamW, a tensor lr, OneCycleLR lr/beta written in place, and static input and hidden
buffers, and cuDNN persistent-RNN capture has to be proven. Parity: T1 as in B(1).

## 3. Rejected
- **bf16 autocast.** sm_86 gives bf16 the same tensor throughput as fp16, so no compute gain; its one benefit (no GradScaler sync) B(1) delivers without new numerics. It puts 8 mantissa bits into the hidden state carried across 10 chunks: a T2 risk for zero speed.
- **JAX rewrite.** XLA has no persistent RNN (`lax.scan` launches per-step kernels, slower than cuDNN's small-h path, F9). It means rewriting the loader, losses and exporter parity; JAX is not installed; XLA preallocation fights the 5.7 GB cap.
- **Custom persistent GRU (Triton/CUDA).** cuDNN already runs one for h<=128 (F9).
- **Custom event-feature kernel.** build_inputs is ~20 ms per 32 sequences (profiling-2026-09-15), ~3-5% of the step; ~10% at B128, which I covers.
- **On-disk raw cache.** Lossless uint16 codes save only 1.2-1.7x (F5): 66-91 GB for 12.5k sequences against 77 GB free, and their decode (16-25 ms) is no faster than parquet's (21-28 ms). fp16 is lossy (ml_lead M6). Precomputed event features add 7.9 MB/seq (~100 GB) to save ~20 ms of GPU work per batch. (datasets/wunder.zip, 34 GB, appears to duplicate the parquet files; deleting it is the user's call.)
- **Two jobs time-sliced at 0.48 each (today).** ~30 aggregate against ~44 for one job (F1).

## 4. Expected trajectory and ceiling (estimates until S0; jobs/h includes ~1.5 min of eval, predict and startup)
| configuration | seq/s per job | 2-epoch fold job (24k seq) | jobs/h |
|---|---:|---:|---:|
| today: 2 jobs x 0.48, 2 workers | 13-15 (agg 28-30) | 27 min | ~4 |
| A: 1 job x 0.9, 4 workers | 50-65 | 6-8 min | ~7-8 |
| + B, C, D (same recipe; T0/T1) | 70-90 | 4.5-6 min | ~9-11 |
| + F, G (B128 x C500; T2) | 130-160 | 2.5-3 min | ~14-17 |

**Ceiling.** Exact recipe B32 x C2000: ~70-90 seq/s, bounded by persistent-GRU sequence latency (~0.3-0.35 s of GPU work per 32 sequences once propensity is cached; x1/0.9 duty). B128 x C500: the GPU reaches ~180 seq/s (0.3 s GRU + 0.24 s losses/optimizer + 0.08 s features per 128 sequences), but the loader binds first: zstd+BSS decode costs 21-28 ms/seq per core, C gives ~30-35 seq/s per worker, and 5 workers give **~130-160 seq/s end to end: 9-10x per job and ~5x in aggregate over today.** Going beyond needs faster decode than parquet zstd, which 77 GB of free disk rules out. EMA-4 recipes double the epochs; E and I recover part of the teacher cost.

## 5. Ledger entries for the controller
- E-166 (analysis): the throughput plan, F1-F9, and the chunked-features bit-exact prototype. Researcher note: GRU/LSTM hidden >128 leaves cuDNN's persistent backward (F9); budget the slowdown for wider arms.
