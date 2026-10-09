# H230 all-data student pair recipe and launch contract V3

V3 supersedes [student-pair contract V2](./H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V2_20261008.json) (SHA256 `5ffacae67cf87abe21a7f6b26f24bebe9010ab5b8e8bf87766908010acb21baf`). It preserves V2's science, commands, source/cache recipe and corrected shuffled drop-last exposure. It adds the root's source-compatibility receipt and accurately describes the model/head ancestry. V3 is **not launch authorization**.

## Parent checkpoint and new family head

The parent uses source V2 with `family_head=false` and 15 main/auxiliary output rows. Each source-V3 student has 23 output rows, including an added 8-row family head that is initialized fresh under that student's seed. The two students share the parent's recurrent trunk and compatible main/auxiliary parameters, the same fixed ep3 checkpoint, and the same teacher cache. Their seed-specific new family heads and training RNG do not make the pair independent whole-model draws.

Root's static compatibility receipt (`2b2856eb2f9ed41acab4fb18a8def3dc00a90bca4855f8140bf33c95e967905f`) confirms the listed shared model, feature, propensity, data, training-state and optimizer modules are byte-identical across the source snapshots and describes the remaining trainer/cache differences. It explicitly says no actual checkpoint load was performed. Before either fit, a separate bounded check must load the exact successful parent `model_ep3.pt` through the source-V3 student path, verify the permitted key/shape mapping and finite tensors, and persist the load receipt. This static receipt is not a substitute.

## Membership and exposure

The pool contains 10,606 train and 1,873 valid groups, including 468 previously exposed validation groups. With RandomSampler, batch size 16 and `drop_last=True`, each epoch has 779 full batches and consumes 12,464 unique sampled sequences from 12,479 groups; 15 are omitted for that epoch. Omitted identities vary with each student's seed/order and are not materialized by the trainer. Complete pool membership is not complete per-epoch or two-epoch exposure, and this fit cannot support a held-out claim.

The frozen recipe remains GRU256×2, seeds 3 and 4, 2 epochs, fixed epoch1; batch16/chunk2000; WMSE chunk loss; AdamW/OneCycle; FP16; alpha0.7; raw four-family mean targets with family weight0.1; propensity weighting and structural/next-event auxiliary weights0.3 each. The exact source, checkpoint, cache, sampler and worker commands are bound in the JSON.

## Launch gates and timing

The parent must terminate successfully and root must bind the fixed ep3 checkpoint hash. The V2 cache request must be separately approved and admitted after the parent fit; all eight full-membership arrays and realized metadata must validate. Then run the exact checkpoint-load check, obtain a fresh root approval and current resource admission for seed3, followed by a separate admission for seed4. One GPU worker at a time.

Closest matched seed3 fits took 2,000.28 and 2,004.93 seconds for 1,492 batches. Scaling from 746 to 779 batches per epoch estimates 2,088.8 and 2,093.6 seconds per all-data student, or 4,182.4 seconds sequentially; these are planning estimates, not measured all-data throughput. Contract SHA256: `5827127486542036a00c82b1957ed1788fff48ab84f36ca454beea2817d256d0`.
