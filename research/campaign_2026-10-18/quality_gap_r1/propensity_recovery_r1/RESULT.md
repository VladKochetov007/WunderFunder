# Propensity recovery completed

The exact F0 propensity model and provenance were restored to
`B/quality-gap-r1/restored-propensity-v1/artifacts/runs/c20260929_prop_f0/`,
where B is `/home/grinchenkoeg/computer-vision-homework-20261018`.
Both files match the archive size/SHA pins, total187,249 bytes. No tensor load,
inference or fit was performed.

- [Prepared helper and inventory sources](README.md), [independent review](REVIEW.json).
- [Admission and exact launch](LAUNCH.json): aggregate1.35 reserved CPU cores,
  support25%CPU/192MB/120s/no swap, CUDA hidden and physical scratch.
- [Exact success receipt](RESTORE_RECEIPT.json): model SHA
  b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7,
  provenance SHA f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812.
- [Execution](EXECUTION.json): exit0 in2.831190s, success receipt captured as raw bytes.
- [Cleanup](STOP.json): only the own exited unit stopped, after capture.

This resolves the missing propensity-file dependency for strong-control export.
The old cache compatibility, full source/runtime binding and resource admission
remain separate requirements. Do not repeat this completed restoration.
