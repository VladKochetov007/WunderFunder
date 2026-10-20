# S4 fit-valid export adapter draft

This folder contains a local S4 fit-valid export adapter derived from the reviewed S3-valid and S4-train admission gates. Root owns final admission and the single remote launch. The body is now bound to the root-captured S4-train terminal acceptance and exact-stop receipts; it remains unlaunched pending fresh preflight and root review.

The selected producer is S4 epoch 2 (`fg19_t3_h384x2_e3_s4_v1`) using its own accepted checkpoint, provenance, and normalizer. The full-shape geometry/profile remains anchored to the accepted S3 profile. Existing S3 train, S3 valid, and S4 train outputs are acceptance-bound and checked by file stat and sidecar SHA only. The S4-train normalizer identity is checked and carried into the admission receipt for downstream S4-valid acceptance. No prediction array payload is loaded, hashed, or scored here. Streamed CPU cache validation must still check shape, alignment, and finiteness.

The new fit-valid output has 1,405 rows and must be absent together with its `.partial`, `.json`, new unit, and evidence directory before admission. The staged teacher-cache contract and three native teacher acceptance JSONs are verified byte-for-byte and not rewritten.

The S4-train predecessor receipts are acceptance SHA-256 `26557d622a2ad96bd4b25000389e3956f9a3283ecdfb33f38afcffad5cc52c5e` and exact-stop SHA-256 `80454742a9edaebda9cf42839c6101dbc8b8f8bf01158552e43eef230e7392b9`. Selected S4 checkpoint SHA is `a59a747176263fc0ecdfcc584f78cf77b13465ec4645ca4bb5d9992380865a36`; accepted S4-train normalizer hashes are mean `dd4147321b23ccf5fa686a6d1c5366681fc6f4b3a8cd8b420986f05a80b5f807` and std `7b40914fb16a1912b2f0219de5cb7d98b59ff7707bdc09d094c8d8d3ca30a6e7`, both shape 211 and torch.float32.

Planned command shape:

```text
/home/grinchenkoeg/WunderFunder/.venv/bin/python -u /home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-88becca8/scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --loader-workers 0 --output-dir /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1 --create-only
```

The proposed unit/evidence names are `computer-vision-homework-fg19-teacher-s4-valid-export-v1.service` and `research_tmp/fg19_teacher_s4_valid_export_v1`. Reused limits: one CPU, 7.5 GB memory high / 8 GB max, no swap, 4 GiB CUDA allocator cap plus 1 GB reserve, physical scratch, 10,800 seconds. This folder has not submitted a command.

See [REMOTE_SUBMISSION_TEMPLATE.py](REMOTE_SUBMISSION_TEMPLATE.py), the rendered [REMOTE_SUBMISSION.py](REMOTE_SUBMISSION.py), [PREPARATION_NOTE.md](PREPARATION_NOTE.md), and [LOCAL_CHECKS.json](LOCAL_CHECKS.json).
