# S4 train export successor adapter (prepared, unlaunched)

This draft reuses the reviewed S3-valid adapter and frozen source/profile/resource gates. Root owns final admission and the single remote launch. The S3-valid predecessor now has a real accepted terminal receipt and exact-stop receipt bound into the rendered body:

- S3 valid acceptance SHA-256: `676f609681e71c5ad26ad3261527df69024a84c311ab8375a2c07a42eb76a288`.
- S3 valid exact-stop SHA-256: `902549b2c12290e372c8f11d57f189f2f21cd39141fe12f389a402fc2b3c863e`.
- S3 train acceptance SHA-256: `a3e6b023dda48c339daa12f17d721eb0765efe63dead6220976425fc1485d2f4`; exact-stop SHA-256: `5d9e2a956d21ab8a1130323475dcb6cdac49d2f9efee775dd912adc6255c92ba`.

The selected producer is contract teacher entry 1: `fg19_t3_h384x2_e3_s4_v1`, epoch 2, checkpoint SHA `a59a747176263fc0ecdfcc584f78cf77b13465ec4645ca4bb5d9992380865a36`, model SHA `e20ce56668c9633141e461373cf6ea1f05d4ea2eaf50b6d9b3dbd48833221c5a`, and provenance SHA `6a6c5bae6bd2ee943684ddad3b6c888682cd4e13de492cf1cac571e97727b2bf`. The accepted profile remains pinned to S3 / contract entry 0. S4's own checkpoint and provenance bytes are checked before admission; its native teacher acceptance remains in the existing staged contract metadata.

The adapter verifies both completed S3 prediction files with accepted path/stat/sidecar SHA only. It does not load, hash, or score either prediction array. Array header, payload hash, and finiteness checks remain for the streamed CPU cache builder. The new S4 train output must have 10,535 ordered fit groups; its `.npy`, `.partial`, and `.json` and the evidence directory must be absent. The staged contract plus three native teacher acceptance JSONs must match exact bytes and remain the only JSON metadata there.

Planned one-unit command:

```text
/home/grinchenkoeg/WunderFunder/.venv/bin/python -u /home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-88becca8/scripts/predict_run.py fg19_t3_h384x2_e3_s4_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --loader-workers 0 --output-dir /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1 --create-only
```

The unit/evidence names are `computer-vision-homework-fg19-teacher-s4-train-export-v1.service` and `research_tmp/fg19_teacher_s4_train_export_v1`. Reused cap: one CPU, 7.5 GB memory high / 8 GB max, no swap, 4 GiB CUDA allocator cap plus 1 GB reserve, physical scratch, 10,800 seconds. No S4-train command has been submitted from this draft.

Local checks passed: Python bytecode compilation, Ruff, and receipt SHA/invocation binding during generation. No remote command was issued here.
