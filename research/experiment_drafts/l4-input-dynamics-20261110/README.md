# Input-dynamics -> shared GRU, 10 November 2026

The exact eleven-section contract and run instructions are in [PR.md](PR.md).

| GPU | Seed | First stage | Conditional next stage | Comparator |
|---:|---:|---|---|---|
| 6 | 203201 | Input innovations, fixed2E | Shared two-target regression, fresh optimizer, fixed4E | PR77 shared6E, seed203201 |
| 7 | 203202 | Input innovations, fixed2E | Shared two-target regression, fresh optimizer, fixed4E | PR77 shared6E, seed203202 |

Four new training stages reuse the two already admitted PR77 controls. Across
the two studies there are ten unique stages. Per-phase real-device profiles and
source/data/parent/reset checks precede scientific fits. The separate source
capsule preserves every live PR77 recipe. Native fitting continues after SSH
disconnect; no historical fit is resumed.

Pretraining uses only train raw features and predicts innovations at ordinal
steps16/64/256. It has no task validation score. Children use the complete
official valid set every epoch; the final4E child is compared with its same-seed
fresh6E control. Validation is exposed development evidence. A lower pretext
loss or an interim score is not a public-score claim.
