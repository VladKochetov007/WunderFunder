# E419 recovered matched QAT screen

Both dual-T4 QAT fits completed their fixed one-epoch schedule and full-validation FP32 prediction exports. The provider reported ERROR afterward, while copying an imported control-checkpoint symlink as an output. The output-path guard correctly rejected a path outside the worker root. All 16 scientific steps on each worker returned zero; the recovered bytes were independently hashed and rescored locally. No fit was repeated to recover these results.

All contrasts below use the same 468 fold-0 sequences, 141 connected components, official clipped pooled WP, fixed seed 3, and 2,000 paired component-bootstrap draws. This is an exposed development fold. Intervals are conditional on fitted models and do not include training-seed or search uncertainty.

| Fixed intervention | WP | Difference against its matched comparator | 95% component interval |
|---|---:|---:|---|
| Parent FP32 | 0.680941207 | — | — |
| Parent, recurrent PTQ | 0.680923257 | −0.000017949 | [−0.000163505, +0.000156527] |
| Parent, recurrent/head PTQ | 0.680926273 | −0.000014934 | [−0.000163045, +0.000164296] |
| One-epoch FP32 continuation | 0.681496158 | — | — |
| FP32 continuation, recurrent PTQ | 0.681502999 | +0.000006841 | [−0.000015659, +0.000033357] |
| FP32 continuation, recurrent/head PTQ | 0.681509427 | +0.000013269 | [−0.000013003, +0.000043630] |
| Recurrent QAT continuation | 0.681500981 | +0.000004823 versus FP32 continuation | [−0.000021980, +0.000032119] |
| Recurrent/head QAT continuation | 0.681489894 | −0.000006264 versus FP32 continuation | [−0.000030186, +0.000019771] |

QAT minus PTQ of the matched floating continuation is −0.000002018 for recurrent weights and −0.000019533 for recurrent/head weights; both intervals include zero. These comparisons change training as well as final represented weights. PTQ of the trained QAT checkpoint was not separately evaluated here. There is no supported QAT improvement, but the tested quantized predictors preserve accuracy closely on this fold. Native parity and deployment runtime remain separate, uncompleted requirements.

The two separately named floating controls have identical tensors and predictions; they are one effective control, not two independent replicates. Each QAT fit completed 14,920 optimizer and EMA updates, with zero AMP skips. Peak device allocation observed was about 2.09 GB per T4; worker peak host memory was 8.13–8.15 GB. Worker elapsed accounting sums to 3.074 device-GPU-hours, including inference/scoring stages. The provider's settled quota was 5.73/30 hours used at 14:01Z; it changed after termination and is not a substitute for per-device accounting.

Raw recovery evidence: `kaggle/E419_qat_dualT4_v4/provider_error_20261004T1401Z/audit_summary.json` and `output_hash_manifest.json`. Transfer bindings: `checks/e419_v4_recovery/transfer_receipt.json`. Paired result: `checks/e419_v4_recovery/e419_recovered_paired_report.json`. Exact local scoring command/input hashes: `specs/e419_recovered_paired_score_v1.sealed.json`; successful worker: `runner/attempts/e419-recovered-paired-score-v1.json`.
