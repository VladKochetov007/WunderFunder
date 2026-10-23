# Portable age-screen source packet

This folder holds the isolated source for the six-head cached-parent age screen.
`SOURCE_SHA256_R1.json` preserves the earlier source snapshot;
`SOURCE_SHA256_R2.json` pins the portable Kaggle CPU actor and its canonical
dependency closure. The frozen machine recipe remains at the repository draft
path and its SHA is unchanged.

The R2 adapter streams raw112 plus the transported fit suffix5/evaluation
parent2 into a one-fold 117-column cache. It preserves the original NPY header,
checks every transported sidecar and the root metadata attestations, then
requires the reconstructed full-cache SHA to equal the original local payload
pin before fitting. The original local source paths and mtimes remain in the
logical map; Kaggle Parquet paths/mtimes, cache/output/resume locations and the
read-only extracted bundle are recorded as physical paths. It never fabricates
local mtimes or calls the strict local cache loader on relocated files.

The profile runner rebuilds all of F0, samples 16 fit and 8 target-free eval
sequences for three heads, closes both memmaps and removes the derived cache.
The fit runner handles one complete fold, publishes three checkpoints and
complete prediction arrays, records full live disk usage, closes mappings and
removes the cache before a subsequent fold. Root admission and transport/source
manifest hashes are mandatory. The work roots are fixed to
`/kaggle/working/disk_scratch/age_actor_bundle` and
`/kaggle/working/disk_scratch/age_actor_work`; raw data remains mounted below
`/kaggle/input`.

Ruff, Python compilation, the training structure checker and synthetic delivery
invariants passed. The synthetic suite checks checkpoint continuation, fit tail
order/page release, duplicate and incomplete prediction coverage, eval
target/weight-column invariance, exact fit/eval cache assembly and eval zero
columns/parent identity. It used one CPU with CUDA hidden and disk scratch under
`b300_scout/tmp`. No real cache arrays, labels or Kaggle payloads were read and
no profile, scientific fit, grading or provider mutation was performed. Root
owns integration, profile admission and execution.
