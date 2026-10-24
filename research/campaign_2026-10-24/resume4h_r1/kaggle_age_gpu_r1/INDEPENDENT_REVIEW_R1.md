# Independent static review: Kaggle age GPU profile R1

## Findings

### Blocker — runtime allocator cap exceeds the frozen recipe

`KERNEL_RECIPE_R2.json` declares both `gpu_allocator_limit_bytes` and
`gpu_memory_allocator_ceiling_bytes` as **4,000,000,000**. The pinned
`src/wnn/training/age_readout_runtime.py` instead sets
`CUDA_ALLOCATOR_LIMIT_BYTES = 4 * 1024**3`, or **4,294,967,296** bytes. The
profile can therefore allocate 294,967,296 bytes more than the declared cap.
The admission template contains no separate VRAM limit that can tighten this
setting. Before admission, either make the runtime enforce 4,000,000,000 bytes
and re-pin the source closure, or update the recipe to the intended 4 GiB cap
and re-pin the frozen packet. This review did not modify either.

### Admission gate — full-cache rebuild cost is unknown

The driver reconstructs and SHA-validates the full F0 fit/evaluation cache in
the same Kaggle kernel. The fit cache alone is pinned at 16,024,320,384 bytes;
the R3 CPU profile deletes its cache and does not provide reusable bytes. The
R2 recipe labels the rebuild time unknown. Keep GPU admission closed until the
R3 CPU cache-parity receipt establishes the complete reconstruction/open cost
and root confirms it fits the 600-second wall limit with the profile and
finalization reserve. This is an execution-feasibility gate, not a scientific
result.

### Residual time-bound limitation

The driver hard-bounds hashing/extraction/profile phases against the admitted
deadline and kills the profile process group on timeout. The 45-second
finalization reserve is budgeted but cleanup and receipt writing in `finally`
do not have a separate hard deadline. Thus the 600-second total is protected
through the profile child, while finishing cleanup within the reserve remains
an assumption. Root should treat the 600-second value as an intended cap, not a
proven hard bound on every exit path.

## Static contract review

- **Seed and recipe:** R2 binds profile F0/seed1001. R1's seed1201 label is
  retained as the documented erratum. The base machine recipe is separately
  pinned to SHA-256
  `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd` and
  its fold 0 is seed1001.
- **Profile scope:** 16 fit sequences × 20,000 rows, 3 epochs, 8
  target-free evaluation sequences, arms A/G/F. At batch size 8,192, the fit
  subset yields 40 batches per epoch; three epochs yield 120 updates per arm,
  360 optimizer updates total. It reads fitting labels to run the profile but
  records zero scientific fits and performs zero official metric calls or
  quality scoring.
- **Target exposure:** the portable raw reader requests `targets=False` and
  `mask=False`. Evaluation cache reconstruction fills the target/mask columns
  with zeros, and the fit evaluator indexes only raw features and parent
  outputs. No evaluation targets, scoring masks, metric, or quality output are
  read or computed by this profile path.
- **Source delta:** the frozen source map has 66 module pins. Against the R3
  checkout, only `age_readout_cli.py` and `age_readout_runner.py` differ. The
  CLI change allows portable CUDA only for `profile` and requires exactly one
  visible CUDA device; the runner removes the CPU-only check only on portable
  profile cache reconstruction. Portable `fit-fold` remains CPU-only. The
  canonical fitting/math modules are unchanged.
- **Precision:** the pinned fitting path requires FP32 default dtype, enables
  deterministic algorithms, disables CUDA matmul and cuDNN TF32, and wraps fit
  and prediction in disabled autocast. The path does not use gradient scaling.
- **Resource guards:** CPU affinity is reduced to one core, thread pools are
  configured to one, admission caps process-tree RSS at 4,000,000,000 bytes and
  requires zero swap, and the child watchdog uses the admitted wall deadline.
  The CLI rejects portable CUDA unless exactly one device is visible. The
  allocator guard has the mismatch above; it is also a PyTorch allocator cap,
  not a device-wide cgroup limit.
- **No hidden cache reuse:** the driver hashes the 1.143GB base archive,
  extracts the pinned bundle, reconstructs and full-file-hashes the F0 cache,
  runs the profile, then deletes the generated cache. The R2 recipe explicitly
  marks reuse from the R3 CPU run false. The unknown full-cache work is not
  omitted from the child deadline; it is included in `run_profile`'s elapsed
  time.

## Pins and review scope

- R2 recipe: SHA-256
  `ad424a874d326336442ffd1a8417dfb7f70a1e9e4aacc0242c38280dd41e1878`.
- Source manifest: SHA-256
  `bf034ffc8ac0212f5a52d7d5b9c57a11bb54750275147aa70d601c0a7b9795fc`.
- Source overlay: 389,120 bytes, SHA-256
  `8d5c2567589876855256e7009a5be23a9f4b56ff366956da072daf21a49b2716`.
- Overlay manifest: SHA-256
  `c977ca865e2e40da4fe7f38d7ab53c533edd8f85dfd4df6266ea2c1a581e3fef`.
- Kernel driver template: SHA-256
  `8e653edc7370e5be046bae8068cf53dea854e6fc8f19921adba6c5c1ce73473b`.
- The actor's payload audit verified all 42 overlay members against the actual
  tar bytes and all 66 source pins (64 unchanged checkout files plus the two
  expected replacements). This review recomputed the R2 recipe, source map,
  overlay, driver-template hashes and inspected the two-file source diff.
- The 1,143,552,000-byte base archive was not available locally for an
  independent byte rehash. Its SHA is an existing export/dataset pin; root must
  validate the exact mounted Kaggle artifact. No Kaggle API, upload, raw-data or
  cache-payload read, Torch/project import, test, profile, or fit was performed.

**Disposition:** static review has one actionable source/recipe blocker and
one known cost-based admission gate. The remaining path checks and scope match
the R2 packet as frozen. No GPU admission is recommended until the allocator
cap is reconciled and the CPU cache-parity timing is available.
