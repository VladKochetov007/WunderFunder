# Independent current-input endpoint replay

PASS for the stated feature-only marker identity on the accepted supervised
movement subset. `quote_endpoint_independent_replay.json` was produced by a new
independent raw TRAIN parquet replay, without actor module imports. It rebuilds
the pooled interior-transition vote map on the 64 fitting groups and grades
current code/a6 on the 32 test groups. It matches every group/class count.

The endpoint is the TRAIN fitting minimum of a6, independently also matching
`float32(NormalDist.inv_cdf(1e-7-float64epsilon)) = -5.199337482452393`.
Current-value equality is causal. The held-out read opens only 22 price-code
fields and a6; no held-out targets, masks, sequence IDs or future rows enter
marker inference. Retrospective classes come from the sidecar whose raw-label
derivation was independently replayed earlier, and are used only for grading.

All 67,839 accepted nonzero proxy moves coincide with the endpoint. None of
45,649 accepted zero proxy moves does. There are 16,894 unknown moves among
130,382 code events, with 8,973 endpoint events in the unknown subset. Keep
unknowns unknown; the 100% conditional agreement cannot be reported over all
events. Known movement coverage is especially limited in classes0/1.

Of 76,825 endpoint rows, 76,812 are price-code event rows. The remaining13 are
10 sequence origins and3 nonorigin quiet rows, all3 in class1. Initialization
and an observed current event are different cases; reset state at a sequence
origin regardless of whether a6 equals the endpoint. Do not force the three
quiet exceptions to zero movement merely because the representation stayed
unchanged. An endpoint indicator can still be computed there without an oracle.

Movement votes derive from TRAIN target changes under an assumed stable future
reference. This is not independent identification of physical bid/ask movement,
seconds, target horizon or source instrument. The marker differs from the raw
code-refresh clock on known zero events, so it supplies a direct causal event
falsifier beyond the earlier age correlation. The network already sees a6;
adding its threshold may add no useful model information.

Next, freeze a new TRAIN component panel and the detector/float32 handling
before reproduction. Preserve raw feature values before normalization so exact
endpoint equality does not depend on standardization or reduced precision.
A matched clock/auxiliary comparison must hold chronology, parameters, masks,
updates, event insertion timing and inference state fixed, without tuning a
target horizon on this already inspected panel. No model or auxiliary cache was
changed by this reviewer, and no competitive/public gain is approved.

The durable job was `c20260930_independent_quote_endpoint_v1`, one thread, 2GiB
scope in the shared slice, exit0, 5.42 seconds, peak RSS1,039,632KiB. The previous
metric scorer freeze is unchanged.
