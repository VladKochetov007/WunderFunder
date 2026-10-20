# S5 fit-valid export adapter draft

This folder contains one local adapter for the S5 fit-valid prediction export. The rendered body is now bound to the actual accepted S5-train terminal receipt and exact-stop receipt. Root retains independent final review, fresh admission, preflight, and the single launch; this folder has not issued a remote command.

The selected producer is `fg19_t3_h384x2_e3_s5_v1` at epoch 2, checkpoint SHA `0ec6dcb7fc82e5a8282ef93d460fcac7b4f6dcb4d3cc63726c485fb2f3595f26`, provenance SHA `0cb36af9cb19fb278d0bd500514d014972ccf997c8b91baa5ab43896e6cb287f`. It reuses the S3-anchored geometry profile and S5's own normalization identity read from its contract-pinned `provenance.json`; the accepted S5-train record matches that normalizer. S5 train was accepted for 10,535 sequences and its exact-stop record confirms the unit is not-found/inactive/dead with no populated cgroups.

S3 train/valid, S4 train/valid, and S5 train producer payloads are never loaded or hashed by admission; producer SHA, file stat, and sidecar SHA are checked. Fit-valid membership is 1,405 ordered rows; full-cache finiteness and output alignment remain the responsibility of the existing streamed CPU builder.

The proposed S5-valid output, `.partial`, sidecar, evidence directory `research_tmp/fg19_teacher_s5_valid_export_v1`, and unit `computer-vision-homework-fg19-teacher-s5-valid-export-v1.service` must be absent before admission. The shared prediction directory and four staged metadata JSON files are checked against pinned identities and are not rewritten. Fresh source/data/split/disk/GPU/process/resource gates are reused from the reviewed predecessor adapters.

Planned command after root's final review and fresh admission:

```text
/home/grinchenkoeg/WunderFunder/.venv/bin/python -u /home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-88becca8/scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --loader-workers 0 --output-dir /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1 --create-only
```

Expected output shape is `[1405, 20000, 2]` from fit membership, with 224,800,128 bytes. Limits remain one CPU, 7.5 GB MemoryHigh, 8 GB MemoryMax, no swap, one GPU, zero loader workers, 4 GiB CUDA allocator cap plus 1 GB reserve, physical scratch, one BLAS thread, and 10,800 seconds. No prediction array, model, or training data was loaded locally.

See `LOCAL_CHECKS.json` for the focused receipt-binding checks and `ROOT_PREPARATION.json` for the current root-preparation pins. The prior fail-closed pending snapshot remains under `pending_s5_train_binding_v1/`.
