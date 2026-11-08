# Initial full-valid fixed-prediction comparison

[Comparative table](COMPARISON.md) and [complete report](REPORT.json) score the
available frozen endpoints using [actual input identities](INPUTS.json).
All1873 official-valid sequences and3528273 scored rows are covered with the
unchanged global_wp_clip2_v1 metric. Missing LSTM/student rows stay unavailable;
there is no coefficient fit or implicit baseline substitution.

Canonical TL3 uses34%completed W18 epoch1 +33%tiedlead seed5 epoch3 +33%tiedlead
seed6 epoch3. WP is0.6923774245652725. The fixed90/10 GALAM contribution is
+0.000119022 and the TCN contribution is−0.000033684. These changes are small,
single-fitted-pool, development-exposed observations; no significance, fresh
confirmation, private/public transfer or serving-compliance claim is made.
The final six-fit comparison and paired student differences are still pending.

The GALAM source is the original scored-only raw prediction artifact with its
native epoch/index/producer receipt and independently recomputed scoring-row
ordering. Temporary zeros outside that verified mask exist only inside this
evaluation adapter; they are never training targets or claimed full predictions.
