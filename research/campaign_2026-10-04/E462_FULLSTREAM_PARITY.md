# E462: exact public-combo INT8 parity

Both fixed INT8 representations of the public-confirmed combo passed their preregistered full-stream parity checks. This establishes native/reference agreement for each represented model; it does not establish score preservation against the original i16 predictor or a public improvement.

The comparison covers 20 complete 20,000-row sequences, 400,000 rows per representation, plus reset and future-feature perturbation checks. Native portable v2/v3 results are bitwise equal. The independent Torch reference uses the representation's own quantized parameters, including the actual fused parent/head readout.

| Representation | Maximum absolute native/reference difference | Global-WP discrepancy |
|---|---:|---:|
| q8r: recurrent weights INT8 | 0.0001354217529296875 | 0.00000010132613070812369 |
| q8x: recurrent and readout weights INT8 | 0.00019338726997375488 | 0.00000015202746306286485 |

Both pass the unchanged maximum-error gate of 0.0003 and WP-discrepancy gate of 0.000005. Required predictions are finite, warmup returns no prediction, A/B/A sequence reset reproduces A bitwise, and future-feature perturbations leave earlier outputs unchanged.

Worker `e462-public-q8-parity-v2` exited 0 with verified teardown after 2,207.735 seconds, peak RAM 1,058,693,120 bytes, no GPU allocation and zero worker OOM events. This loaded-host parity duration is not competition runtime. See `E462_public_q8_parity_v2/parity_report.json` and its immutable runner record.

The historical 368-sequence i16/q8r/q8x paired score and quiet one-CPU timing remain packaging requirements. The first score attempt, `e462-public-q8-score-v3`, failed before any callback because its fresh child process imported root `utils` before adding the repository to its import path. It exited 1 with verified teardown and zero worker OOM events. Its source, sealed spec and raw traceback are retained. A narrow import-path repair and a fresh attempt are required; the completed scientific parity computation is not rerun.

The built ZIP hashes remain fixed: q8r `7edb26d8903bafa2f3c38bf31b1a5d0dad1fd6f79e1f4d5a0ea13eaf3ca044fd`; q8x `b435c04c0b047c72aeaca93a0be5aa2436a885d3abce5da483bd7fa43bfa87e7`. Neither ZIP is yet advertised as ready for manual upload.
