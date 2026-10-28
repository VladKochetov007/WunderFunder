# Eight explicit launch recipes

g0 through g7 match [PANEL.json](../PANEL.json) and the user's fixed v3 allocation.
Each starts from fresh weights and has one assigned GPU. Current actual-data
profile candidates are batch32 for plain GRU and384 for all GALAM arms, following
the strict-FP32 synthetic shape sweep. One prefetched batch is explicit; full
valid reporting uses batch384. The common controlled GALAM batch must pass the
actual-source memory/throughput profile before science. Old candidate hashes are
retained in Git and engineering receipts.

Every recipe has maximum20epochs, all-fit train-only normalization, no teacher
or hint loss, all-after-warmup fitting, full-valid epoch monitoring and all-epoch
checkpoint retention. Diagnostics/telemetry intervals are100 for plain GRU and25
for GALAM, with event-specific and full-run byte bounds enforced at startup.
No five/ten-epoch v1 recipe or historical result is overwritten.
