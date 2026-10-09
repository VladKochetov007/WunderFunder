# H230 all-data student pair recipe and launch contract V2

V2 supersedes [H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V1_20261008.json](./H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V1_20261008.json) (SHA256 404688c536eb3bd7c44987bd0e6c3b6efbe1d326b28187ad75f31b49d2674ef4) because V1 overstated per-epoch sequence exposure. The scientific worker argv, source files, parent/cache dependencies, resource proposal, seeds, and checkpoint policy remain unchanged. The complete pool is included in the all-data validator and loader, but the loader drops the last incomplete batch.

The two students are GRU256×2, seeds3 and4, two epochs, fixed epoch1. They share the own seed3 all-data parent at ep3 and the same teacher cache. This follows [ADR0019](../../../../specs/adr/0019-common-parent-final-student-pair.md) (SHA256 15466c26e37a6fb9a7106f47db3811e795338ad77f98a609cd01a33dc42f3ba8); the pair members are not independent initializations. The fold-specific teacher-fit history remains a declared limitation, with a separate own all-data teacher cohort queued after this endpoint.

## Pool membership and per-epoch exposure

The training pool contains all 10,606 train and1,873 valid groups, including 468 previously exposed validation groups. The source uses the default RandomSampler with DataLoader shuffle=True and drop_last=True. Each epoch therefore has 779 full batches ×16 =12,464 unique sampled sequences from a 12,479-group pool; 15 groups are omitted by that epoch’s seed/order. The omitted identities vary and are not listed by the trainer; provenance captures the sampler RNG identity. Complete pool membership must not be reported as complete per-epoch exposure. This fit cannot support a heldout claim.

The cache itself still must contain every supplied train/valid target row exactly once in target-index order. Its expected eight arrays, family metadata, group hashes and builder command remain bound to the cache contract SHA recorded in the JSON.

## Dependencies and commands

The parent fit must terminate successfully and root must bind the fixed ep3 checkpoint SHA. The all-data cache must be separately admitted, built and validated. Fresh data identity and per-fit resource admissions are required; this V2 contract is still not launch authority.

The JSON contains unchanged scientific worker argv and V2-bound controller commands for seed3 then seed4, with one GPU worker at a time. Closest matched two-epoch H230 fits took2,000.28s and2,004.93s for1,492 batches; scaling from746 to779 batches per epoch gives planning estimates of2,088.8s and2,093.6s, or4,182.4s sequentially. Each fit has a3,000-second cap. These are estimates, not measured all-data student throughput.
