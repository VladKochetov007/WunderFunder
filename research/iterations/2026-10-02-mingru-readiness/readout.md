# minGRU readiness completed; forecast quality next

Corrected the exact-zero derivative in original block64 log-scan implementation. CPU literal-recurrence/gradient checks and GPU mixed-precision generated checks pass; see numerical-readout.md. No real data or forecasting fit was used in this study.

Native attempt01 failed compilation because our adapter omitted a raw-row argument. Mechanical attempt02 retained all numerical tolerances/kernels and passed55 checks, including literal FP32 reference, same-INT8 scalar oracle, reset/split/prefix/state semantics. Long20k generated INT8-vsFP32 output RMS0.000668,relativeL2~0.00733; trained-checkpoint impact remains unknown.

Five paired full4 generated20k-row runs gave median1.081676s minGRU versus1.042208s operational student4: ratio1.03787. ProcessCPU nearly equals wall time. The isolated member improves from78.58us FP32 to18.56us INT8 perrow; operational shallowGRU13.89us. Additive/ratio transfers to historical44.5369minutes predict45.83/46.22minutes. These exclude Python/scorer and use random weights/streams; they establish plausible budget, not actual submission qualification.

Decision: allocate the separately frozen two-fold minGRU256x2 parent/H2 recipe, same teachers/features/quarter anchors. Current student4 package remains incumbent. If forecast gain appears, trained native OOF and actual full runtime must be checked before a new deliverable. No GitHub/public submission.
