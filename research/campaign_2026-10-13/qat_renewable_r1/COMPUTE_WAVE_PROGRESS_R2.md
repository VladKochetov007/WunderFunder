---
kind: operation
status: active
created_utc: 2026-10-13T17:47:00Z
hypothesis_ids: [H-QAT-W4-20261013, H-QAT-LSQ-20261013, H-QAT-A8-20261013]
---

# QAT computation wave, second continuation

The full-data floating continuation finished 746 complete-sequence batches and 14,920 optimizer updates in 1,430.7 seconds, with zero AMP skips and fixed raw epoch-0 checkpoint. It inherits the same clean-T3 CP5 F0 seed-3 parent, teachers, normalization and full fitting membership as the frozen execution packet. The native deployment manifest is `4d0e8fe7345124c8837305568f1fa0ae69aa26bd1f12dd01dc57d9e2a09bc24b`, at `artifacts/qat_renewable_r1/100pct_fp32_deployment_r3/manifest.json`. Native Q8, uniform W4 PTQ and mixed PTQ full-fold evaluation is running. No full-data quality result is claimed at admission.

Both fresh same-source LSQ comparison profiles completed 100 optimizer updates, with finite losses and gradients. Fixed/learned peak observed VRAM was 2.174/2.176 GB; measured host peaks were 3.952/2.938 GB. Those short-profile host peaks do not replace the prior complete-fit bound of approximately 4.95 GB. The paired 10% fits reserve 6 GB each and run sequentially through the existing GPU worker. Learned steps include every selected recurrent-W4 matrix, including the Q8 first-head blocks. The comparison control recomputes nonlearned row absmax scales after updates. Both use fixed raw epoch-0 selection and the same optimizer/sampling budget.

The seed-4 private Kaggle profile packet passed independent review, actual upload-byte verification and version-1 READY checks. At 17:42 UTC the provider reported 34,439.317 GPU seconds and 72,000 TPU seconds available. Root admitted only the 1,800-second GPU profile kernel with a 3,600 aggregate GPU-second upper bound and 300-second finalization allowance. Version 1 was accepted at 17:44:30 UTC. This starts provider delivery; it does not establish hardware allocation or finite updates. Full fits await actual profile evidence and renewed quota. The separate actual-data TPU V8 bridge was still QUEUED at 17:42 UTC; no duplicate push was made.

The recurrent activation prototype passed small CPU/native checks. Its initial compiled profiler uses synthetic inputs and no optimizer update, so it cannot establish actual-data training readiness. An independently reviewed successor must bind the real parent, inputs, objective and update/cache behavior before any activation-aware fit. This limitation is an implementation issue, not evidence against activation-aware QAT quality.

The independent continuation/compression grade is linked from the review index. Native-Q8 continuation has the strongest point score in the first small-screen wave (10%: 0.681818498), but its parent comparison remains uncertain on one exposed fold and fitting seed. Neither that score nor the positive 30% QAT deltas establishes a strongest validated package. Full-data, distinct-seed/fold and fixed-blend comparisons remain open.
