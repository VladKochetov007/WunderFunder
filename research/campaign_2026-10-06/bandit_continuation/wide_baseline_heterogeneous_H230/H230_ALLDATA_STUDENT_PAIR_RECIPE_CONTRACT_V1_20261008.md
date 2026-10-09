# H230 all-data student pair recipe and launch contract

This packet freezes a two-member all-data student recipe and exact worker/controller command lines. It does not authorize launch. The live seed3 all-data parent must finish, the full-membership cache must be built and independently checked, and root must bind both artifacts plus fresh controller admission in a reviewed successor.

The students are GRU256×2, train for two epochs and select fixed epoch1. They use alpha0.7 against the unnormalized teacher source sum (coefficient sum0.8125), raw family targets with family weight0.1, the audited propensity weighting, and structural/next-event auxiliaries0.3 each. Both students reuse the same own seed3 all-data parent ep3; seeds3/4 change training randomness and do not support an independence claim between pair members.

## Membership and ancestry

Each student trains every supplied train and valid row group exactly once per epoch: 10,606 train groups (658498ef9fb3ad7f4dab9dea249da809168be595171099406795f72b63554fbe) and 1,873 valid groups (008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7). The 468 exposed validation groups are included; these fits cannot support a heldout score claim.

The eight teacher sources in [H230_ALLDATA_TEACHER_CACHE_BUILD_CONTRACT_V1_20261008.json](./H230_ALLDATA_TEACHER_CACHE_BUILD_CONTRACT_V1_20261008.json) were trained on clean fold0 membership (10,535 train, 1,405 valid) and have predictions over the complete target-index order. Their family-history difference from full-data teachers is explicit. A separate own all-data teacher cohort is queued after this endpoint.

## Frozen implementation and cache

The student uses the independently written H230 source-v3 all-data snapshot, manifest SHA 9938cf880addd22467dd71fc81fc550ea3ab9c7ba792c84cbd73f7073e7de15d, with 55 members rehashed and verified. Trainer SHA256 is 09902e4b4d886f863d5090ff58ba9ec4b7c677f995bb0192f0538c819a693bbb. The cache builder contract SHA256 is 53bfbdaebf6e3990f4c0df43c067de66703914fadd0045b551650ed6b3513430; cache output is cache/bd26_h230_fold0_teacher_family_all_rows_v1, estimated at 13,976,480,000 bytes. It contains eight float32 arrays: main source/normalized/homogeneous (two columns each) and four raw family means (eight columns), each split across complete train/valid membership. The student consumes the source main variant.

Expected student teacher target arrays and group identities are recorded in the JSON contract. Cache build resource admission is still open; the previous clean-fold chunk2 build is not a profile for this all-data chunk1 packet.

## Commands and resource request

The JSON contains exact worker argv and controller launch commands for both unique run IDs. Fit order is seed3 then seed4, one GPU worker at a time. Proposed reservation is 9GB RAM, 2 worker CPUs, zero loader workers, and the current corrected 18GB/20GB/zero-swap parent bounds. Each student has a 3,000-second runtime cap. Fresh resource admission and root approval are required for each fit.

The duration estimate uses completed H230 matched family and family-off fits (2,000.28s and 2,004.93s, each 1,492 batches), scaled from746 clean-fold to779 all-data batches per epoch: approximately2,089s and2,094s, or4,182s for the sequential pair. This is a planning estimate, not measured all-data student throughput.
