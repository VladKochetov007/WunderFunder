# S3 fit-valid export successor draft

This is a local, unlaunched adapter for the single S3 valid fit-prediction export. The root operator owns admission and the one remote launch. The body renders exactly from `REMOTE_SUBMISSION_TEMPLATE.py` and its embedded frozen payload; it does not stage source or rewrite teacher-cache metadata.

The adapter reuses the accepted profile gates and checks the 167-file source manifest (`88becca892b3d6b97022cf06076c84a7ab165974`), dataset/split identity, checkpoint and provenance pins, populated-process/GPU ownership and headroom, disk reserve, and absence of the new unit and valid output paths. It binds the accepted S3 train export (`a3e6b023dda48c339daa12f17d721eb0765efe63dead6220976425fc1485d2f4`) and its exact-stop receipt (`5d9e2a956d21ab8a1130323475dcb6cdac49d2f9efee775dd912adc6255c92ba`), verifying the completed train output by stat and sidecar hash only. It does not load, hash, or score the train prediction payload; finite-array validation remains for the streamed CPU cache builder.

Expected valid-fit output has 1,405 rows. The one-unit command is:

```text
/home/grinchenkoeg/WunderFunder/.venv/bin/python -u /home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-88becca8/scripts/predict_run.py fg19_t3_h384x2_e3_s3_v1 valid --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --loader-workers 0 --output-dir /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1 --create-only
```

The unit is `computer-vision-homework-fg19-teacher-s3-valid-export-v1.service`, capped at 10,800 seconds, one CPU, 7.5 GB memory high / 8 GB maximum, no swap, 4 GiB CUDA allocator cap plus 1 GB reserve, and physical scratch. Output/evidence paths are create-only: the valid `.npy`, its `.partial`, and `.json` must all be absent, as must `research_tmp/fg19_teacher_s3_valid_export_v1`.

Local checks: Python bytecode compilation, Ruff, render parity, and embedded train acceptance/stop SHA validation passed. No remote command was issued and no valid prediction output exists from this preparation step.
