# Future synthetic GPU profile, root launch only

Prepared before any GPU launch. Controls must finish, independent parity grading
must pass, and root owns the single GPU lease and durable job/VRAM monitor.
`gpu_smoke.py` is a profile runner, not a scoring/training replacement.

Fixed shape: batch32, length20000, chunk2000, input211; output23 from primary2 +
spread12/branch3/class4 + next-row2 auxiliary outputs. Student and actual trainer
FastEMA teacher both have these shapes, independent carried `(s1,s2)` tensors,
and reset together only at sequence start. Full raw112 + input211 + target/loss
arrays are resident to represent the real trainer's input/loss memory. No data,
targets, or masks are read from disk; all losses/labels/weights are synthetic.

Policy: same seed3, AdamW lr1e-3/wd1e-2/fused=True, outer CUDA FP16 autocast and
GradScaler (the existing trainer default). Model explicitly disables autocast
internally so recurrence **and dense layers** use FP32; no hidden FP16 slow mode.
Clip gradient norm1, EMA decay0.999, consistency1.0, structural0.3, next0.3.
Student/teacher process all10 chunks; detach each carried state at TBPTT boundary,
EMA update after each optimizer update. Actual label values are intentionally
arbitrary: this tests allocation/gradient shape, not model usefulness.

Root runs each init separately, with identical shapes and policies. Filenames:
`gpu_smoke_multi.json/.pt/.log` and `gpu_smoke_single.json/.pt/.log` under this
directory (or root-owned durable artifact directory). No overwriting old evidence.
CUDA allocator peak/reserved and sampled total-device usage in bytes are recorded.
Runner aborts at sampled total device >5,000,000,000 bytes; root's faster monitor
remains required because stage samples can miss a transient peak. Failure/OOM is
preserved and blocks fitting; do not change batch size or accumulation silently.
Resource configuration continues using configure_torch/WNN_GPU_FRACTION<=0.70.

No torch.compile invocation: training uses optimized FFT/matmul kernels with
O(T log T) work and O(B*T*state_width) space. Compiler startup/unrolled recurrence
is unnecessary for this profile. This is not a native latency measurement.

Future paired fit: batch32/chunk2000, fixed4 epochs, selected zero-based epoch3,
--checkpoint-policy fixed --checkpoint-epoch3 --no-eval-during-fit. Freeze empty,
same fold0 clean chain-excluded recipe, auxiliaries/propensity/since weights/seed.
Maximum1080GPU seconds per arm. If **either** arm misses the fixed4 epochs under
the cap, the paired comparison is inconclusive regardless of available checkpoints.
No unequal-update comparison, evaluation checkpoint choice or batch workaround.
Multi versus single tests only initialization/optimization; the architecture
family remains exploratory. No native implementation before score promotion.
