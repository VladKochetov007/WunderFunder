# Eight explicit launch recipes

g0 through g7 match [PANEL.json](../PANEL.json) and the user's fixed v3 allocation.
Each starts from fresh weights and has one assigned GPU. Initial batches are
32 for plain GRU and16 for GALAM; these are engineering candidates, not measured
optima. The common controlled GALAM batch and telemetry cadence will be updated
from the admitted profile before science, with old hashes retained in receipts.

Every recipe has maximum20epochs, all-fit train-only normalization, no teacher
or hint loss, all-after-warmup fitting, full-valid epoch monitoring and all-epoch
checkpoint retention. The interval200 diagnostics/telemetry budget is deliberately
bounded under1GiB for the initial batch16 case; it can scale with the chosen batch.
No five/ten-epoch v1 recipe or historical result is overwritten.
