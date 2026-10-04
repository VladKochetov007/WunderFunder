# E462: actual fixed INT8 callback scores

The corrected scorer completed all three actual native callbacks over the
same 368 excluded validation sequences (7,360,000 streamed rows and 647,645
scored rows), then used 2,000 paired component-bootstrap replicates across
249 components. It finished in 2,571.36 seconds, with exit zero, verified
teardown, about 2 GiB peak charged RAM, and no worker OOM events.

| Fixed native package | Global WP | Difference from original i16 | 95% paired interval |
| --- | ---: | ---: | --- |
| Public-confirmed i16 | 0.699821638 | — | — |
| q8r: recurrent weights INT8 | 0.699816586 | -0.000005052 | [-0.000043098, +0.000030580] |
| q8x: recurrent and head weights INT8 | 0.699804134 | -0.000017505 | [-0.000069563, +0.000022038] |

Both changes are small and uncertain. These are actual ZIP callback results,
not substituted Torch exports. Composite parent/head checkpoint lineage and
fusion bytes match the archived public-confirmed member. No new teacher, fit,
head, or blend weight was selected. This repeatedly exposed panel provides
descriptive evidence of score preservation, not independent public transfer.

The earlier 400,000-row parity experiment compared each quantized callback
with its own quantized Torch reference and passed the fixed numerical gates.
That implementation-parity threshold is distinct from the change caused by
quantizing the original predictor. In particular, q8r's -0.000005052 score
change above does not violate its own-reference parity gate.

Keep the v3 scorer failure and exact failed source. The v4 repair only added
the repository import roots needed by a fresh callback-scoring child; it did
not change scoring arithmetic or rerun successful fits/parity.

Quiet one-CPU timing is still required before manual-upload readiness. Lower
weight storage alone does not prove a runtime improvement. Neither package
currently establishes a forecasting gain or an expected public score above
the original 0.6655.

Raw score/provenance: `E462_public_q8_score_v4/report/score_report.json`.
Resource record: `runner/attempts/e462-public-q8-score-v4.json`.
Parity evidence: `E462_FULLSTREAM_PARITY.md` and the immutable parity report.
