# E426 full-fit completion

Observed 2026-10-04T19:24:08.971434+00:00. All three four-epoch fold-0 fits completed; checkpoint selection remained fixed at epoch3, with evaluation withheld.

| Arm | Seconds | Peak RAM bytes | Observed VRAM bytes |
| --- | ---: | ---: | ---: |
| constant | 2546.450 | 20419321856 | 4488953856 |
| aligned | 2686.032 | 25754972160 | 4488953856 |
| shuffled | 2574.051 | 18489196544 | 4488953856 |

Each arm completed1,492 batches and29,840 optimizer attempts:29,828 parameter updates,12 AMP skips,29,840 teacher/evaluation EMA update calls. Worker exit0 and teardown succeeded, with no new worker OOM events.

This establishes fit execution, not a forecasting gain. Three canonical all-VALID FP32 exports and the preregistered468-sequence fold-0 paired component score are pending. Existing folds and teacher ancestry are exposed; these are not untouched-data results.

Raw evidence: `runner/attempts/e426-f0-{constant,aligned,shuffled}-s3-4ep-v2.json`, their logs, `artifacts/runs/e426_f0_{arm}_s3_4ep_v2/history.json`, and `checks/e426_sequence_v1.json`.
