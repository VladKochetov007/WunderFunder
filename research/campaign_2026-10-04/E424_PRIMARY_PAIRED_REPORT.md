# E424 primary architecture contrasts

The four primary production-objective fits and their full supplied-validation FP32 exports finished. Fixed epoch 3 was used throughout. A generic prediction exporter assumed a conventional head layout and failed on the specialist checkpoint; the isolated repair passed six width/model/chunk/reset checks, then exported all four existing checkpoints without repeating training. Failed attempts remain preserved.

The same 368 historically excluded sequences and 249 components were scored using clipped pooled WP and 2,000 paired component-bootstrap draws. This panel is repeatedly exposed and is not chain-independent of all training ancestors.

| Predictor | WP | Matched architecture difference | 95% component interval |
|---|---:|---:|---|
| H093 learned specialists | 0.696446285 | +0.000353842 versus H093 budget-matched GRU | [−0.001188360, +0.002638582] |
| H093 budget-matched GRU | 0.696092444 | — | — |
| H094 conditional adapter | 0.697667521 | +0.000308580 versus H094 budget-matched GRU | [−0.000614844, +0.001299199] |
| H094 budget-matched GRU | 0.697358941 | — | — |
| Fixed complete seed-3 combo | 0.699822159 | — | — |

Both primary architecture point differences are positive but uncertain; no result establishes an improvement. Class-3 differences against the respective architecture controls are approximately +0.000412 and +0.002100. These are class-specific diagnostics, not a linear contribution to the global metric.

The architecture parents score below the complete public combo predictor. That comparison also changes the final head fine-tuning/merging procedure and cannot isolate architecture. All contrasts use common FP32 exports in `checks/e424_report_core_fp32.json`. The earlier mixed-precision report remains preserved; `E424_COMMON_PRECISION_AMENDMENT.md` records why a separate FP32 report was necessary.

The hash/universal routing and unconditional/diagonal adapter controls remain part of the eight-arm program. Their pending outcomes must be reported regardless of direction. No submission or architecture-wide rejection is inferred from this primary screen.

Raw evidence: `checks/e424_report_core_fp32.json`, `checks/e424_export_repair_review.json`, `checks/e424_screen_core_fp32.json`, `specs/e424_score_core_fp32_v1.sealed.json`, and successful attempt `runner/attempts/e424-score-core-fp32-v1.json`. The live controller is `checks/run_e424_sequence.py`; its current snapshot is `checks/e424_sequence_v2.json`. The ongoing authorization is in `CONTINUATION_AUTHORIZATION_20261004.md`.
