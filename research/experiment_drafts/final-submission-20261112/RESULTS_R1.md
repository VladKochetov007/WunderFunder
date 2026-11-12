# Final package: measured status and release limits

The exact R2 candidate is built, not uploaded. ZIP SHA256:
`a0fb8b1e254dd8be94301f6553f11b9e1b1ae758c9543f38aae3f021c944894f`,
14,804,012bytes. Its four selected endpoints reuse the completed18-fit final DAG;
all12,480 supplied sequences are fitting data. No independent validation score
was calculated or claimed for these weights.

## Correctness and provenance

The17-fit archive and complete W18 evidence archive passed checkpoint/intake
verification. The Source4 exporter repair accepts its exact historical
`valid_rows` training counter and still rejects real evaluation metrics.
Seven focused counter/guard tests passed. The model-set sidecar has the exact
SHA referenced by the ZIP. The original checkpoints, source tree and historical
scores were preserved.

Feature-only Linux/Python3.11 checks passed finite predictions, deterministic
reset/replay, automatic sequence reset, unchanged past predictions after future
input perturbations and exact fixed nested blending. The exported Q8 callback is
not bitwise equivalent to FP32: blend RMS drift is about0.00242 and max absolute
drift0.01410 on two sequences. This is numeric engineering evidence, not a new
quality score. Native NumPy2.4.6 was used; exact official NumPy2.2.6 qualification
remains outstanding.

## One-core deployment measurement

All measurements below use one Xeon6767P core, BLAS threads1 and no GPU.
The current official FAQ specifies1,970 sequences of20,000 rows; the60-minute
limit therefore permits about91.4microseconds per input row before overhead.
Host-to-platform differences and container/startup costs remain unmeasured.

| Variant | Seconds /20,000 rows | Finding |
|---|---:|---|
| Delivered R2 baseline | 3.43 | Approximately112.6minutes projected; fails admission |
| Q8 VNNI256 | 4.89 | Slower than baseline |
| Q8 blocked | 3.81 | Slower than baseline |
| Fused GRU Q8 | 3.38 | No useful gain |
| SIMD tiedlead GELU only | 3.36 | Negligible quick-timing gain |
| GELU + alternating member/cache order | 2.663 |1.2705x faster than paired3.383s baseline;87.4minute projection |

The combined patch passed actual bitwise comparison of all four members and
post-warmup callback outputs on two full20,000-row sequences, plus reset,
automatic-boundary and causal-prefix checks. It preserves weights, temporal
member order and blend arithmetic. It still does not meet the serving budget.
The first parity-checker attempt failed before inference because its parser
field was misnamed; the R2 checker fixes that argument name and preserves the
failed attempt as evidence.

## Decision and remaining work

Keep R2 and the optimized runtime as reproducible engineering artifacts. Do not
upload a known over-budget package or present a projected duration as an actual
platform run. Incumbent public TL3 remains0.66832; the already submitted
all-data W18 singleton scored0.66453. No improvement forecast follows from the
historical exposed-validation90/10 gain.

The user subsequently requested active B300/L4 work. A separate fresh quality
wave tests PR91 input interaction and compact clean-teacher GRU distillation;
its source/clock/fit receipts must be recorded separately. It does not reopen
the closed previous wave or turn all-data teachers into primary-valid teachers.

The signed-in Windows session is disconnected. Normal `tscon` restoration
returned Access denied5; no portal upload occurred. An available authenticated
interactive session remains necessary for actual upload.
