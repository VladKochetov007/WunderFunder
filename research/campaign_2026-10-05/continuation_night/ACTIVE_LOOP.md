# Active continuation loop

User goal remains active: run reviews/proposed experiments until October 6 10:00 Kyiv locally, then Kaggle GPU/TPU only until explicitly stopped. Latest aggregate limits: 5 CPU cores, 3 decimal GB VRAM, 40 GB RAM. Campaign slice now enforces CPUQuota=500%.

Admitted: independent E492 raw-event recount (1 CPU/2 GiB) and E499 non-scoring clipping diagnostic (1 CPU/3 decimal GB RAM, 2 GiB allocator, 3 decimal GB observed device stop). Both run under the existing carried local budget.

Actors prepare matched full-recipe common shapes and cloud packets for the new VRAM cap; source-only GO from older larger profiles cannot authorize new-cap fits.

Actual outcomes: E499 diagnostic finished (26.174 s, 2.577 GB RAM, 0.422 GB VRAM); its seed-level AdamW discrepancy keeps E489 full-fit admission blocked. E492 reproduction v1 failed a comparator serialization bug, preserved; independent v2 fixes are under cross-review. E496 seed3/fold0 calibration is being source/input sealed for bounded CPU admission.

E496 actual fit-only calibration admitted as night-e496-calibration-s3-f0-smallcap-v1 (1CPU/6GiB/zeroGPU), on the existing reviewed runner. Original18GB draft rejected before launch, preserved.

E492 v2 independently reproduced all raw counts in10.434CPUseconds, peakRAM244MB, zeroGPU. E496 calibration body is running, observed peakRAM2.967GB at first check; no error.

Reviewed E493/E494 full-recipe B16/H256/T1000 control resource probe prepared. Live host headroom rejects the18GB variant; immutable12GB RAM capsule now under independent review and data sealing. CPU calibration continues without OOM; cache pages hit its6GiB cap and are reclaimed.
