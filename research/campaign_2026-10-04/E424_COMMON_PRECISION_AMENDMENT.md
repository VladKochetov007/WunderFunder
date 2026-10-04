# E424 common inference precision amendment

The four specialist/adapter predictions are FP32. The initial core comparison correctly used FP32 for each candidate and its parameter-matched GRU control, but its public-combo anchor still used the older FP16 cache names. The public combo point was 0.699872635 rather than its already-recorded FP32 value 0.699822159. Keep `e424_report_core.json` and its sealed successful attempt as evidence of that mixed-precision anchor. It is not the final common-precision anchor comparison.

The revised reporting-only screen uses the existing hash-bound FP32 parent and head exports, with an explicit alias-to-original-checkpoint map. This preserves exact checkpoint identity, initialisation-chain fit exclusion, sidecar and array hash checks. It changes no fitting, weights, rows, masks, or model selection. The paired architecture-to-GRU contrasts are unchanged. Save the result separately in `e424_report_core_fp32.json`.

An additional source-only controller failure rejected absolute paths while sealing the first paired-score draft; it preceded worker launch and was corrected to normalized repository-relative paths under a new draft identity. No successful fits or predictions were repeated.
