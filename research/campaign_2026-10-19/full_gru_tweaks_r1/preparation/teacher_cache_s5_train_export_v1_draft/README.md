# S5 fit-train export adapter draft

This folder holds one local S5 fit-train export admission adapter. Root owns final acceptance binding, review, preflight, and the single remote launch. The rendered body is bound to root-captured S4-valid acceptance SHA-256 `03c951dab22d1ed089bf5118b350186dd5f1c623bed70cba974b49ab4cddb360` and exact-stop SHA-256 `7b2c47e20a942fe372c7630b0a0bd6c0e90995baffce03fdd4942dfe6752ece9` for invocation `4ee46e04e7264a6ea0278d28d6479400`. Root retains final review, fresh preflight, and the single remote launch.

The selected producer is `fg19_t3_h384x2_e3_s5_v1` at epoch 2, using the checkpoint SHA `0ec6dcb7fc82e5a8282ef93d460fcac7b4f6dcb4d3cc63726c485fb2f3595f26` and provenance SHA `0cb36af9cb19fb278d0bd500514d014972ccf997c8b91baa5ab43896e6cb287f`. The same S3-anchored full-shape profile remains in force. S5's own expected normalization identity is read from its contract-pinned `provenance.json` after that file's SHA is checked; the later root terminal acceptance must verify the exported normalizer matches it. Predecessor S3 train/valid and S4 train/valid prediction arrays are never loaded or hashed here: their accepted producer SHA, stat bytes, and sidecar SHA are checked. Finite values, shape, and order remain the streamed CPU cache builder's responsibility.

Before admission, require the new train `.npy`, `.partial`, and `.json`, unit `computer-vision-homework-fg19-teacher-s5-train-export-v1.service`, and evidence directory `research_tmp/fg19_teacher_s5_train_export_v1` to be absent. The shared prediction directory and four staged metadata JSON files must exist with their pinned identities and are never rewritten. Fresh resource/data/source/output gates are reused from the reviewed adapter.

Exact command for the single root-owned launch after final review and fresh preflight:

```text
/home/grinchenkoeg/WunderFunder/.venv/bin/python -u /home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-88becca8/scripts/predict_run.py fg19_t3_h384x2_e3_s5_v1 train --epoch 2 --campaign-split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json --device cuda --inference-precision fp32 --loader-workers 0 --output-dir /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1 --create-only
```

The fit-train output has 10,535 ordered sequences and expected size 1,685,600,128 bytes. The bounded profile, one-worker, one-GPU, physical-scratch, disk-reserve, and memory limits are pinned in the rendered body. No remote command has been issued from this folder.

See [REMOTE_SUBMISSION_TEMPLATE.py](REMOTE_SUBMISSION_TEMPLATE.py), [REMOTE_SUBMISSION.py](REMOTE_SUBMISSION.py), [PREPARATION_NOTE.md](PREPARATION_NOTE.md), [ROOT_PREPARATION.json](ROOT_PREPARATION.json), and [LOCAL_CHECKS.json](LOCAL_CHECKS.json).
