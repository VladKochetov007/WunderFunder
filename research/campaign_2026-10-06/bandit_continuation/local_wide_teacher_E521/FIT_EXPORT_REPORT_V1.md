# E521 seed3 fit/export completed

The independently scratch-trained raw GRU384 two-layer candidate completed its frozen
four-epoch fit and epoch3 prediction export. No checkpoint selection or score promotion
has occurred. Matched GRU128 seed3 comparison remains pending E520 cloud delivery.

Fit: exit0,5,178.922s; observed total VRAM peak3.994GB, parent RAM17.431GB.
AMP skipped15 of29,840 attempted updates; preserve this numerical execution fact.
Export: exit0,80.652s; observed total VRAM peak1.192GB, parent RAM16.594GB;
scoped RAM reached5.500GB due to accounted cache, without scoped OOM increments.
All-valid predictions are a286MB NPY plus metadata, committed identity in
[FIT_EXPORT_RESULT_V1.json](FIT_EXPORT_RESULT_V1.json). Export covers fitting and
excluded groups; only the declared excludedfold0 may support a held-out comparison.
That fold is previously exposed. No new public score, ZIP or public forecast.

The completed-export cache handoff reclaimed bounded clean filecache under the shared
GPU lock, reducing campaign RAM15.302→9.272GB without changing limits. See
[handoff](EXPORT_CACHE_HANDOFF_V1.json). Seed4 same-shape scratch replication is
prepared with only seed/output-name changed; independent proportional review is pending.
