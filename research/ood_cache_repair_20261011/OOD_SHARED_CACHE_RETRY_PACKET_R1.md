# OOD shared-cache repair and retry packet R1

## Failure and repair scope

The four failed fold-1 OOD jobs requested a full four-family cache. The recorded
cache estimate was 13,382,880,000 bytes, while the host had only about
10.51 GB available above its required 20 GiB free-space floor. This is a
storage-admission failure; it does not score or reject the OOD recipes.

The OOD students request only the homogeneous main target and do not use the
family head. The corrected builder therefore preserves the original five
source records, four-family ancestry, 0.8125 coefficient sum, fold membership,
and exact `combine_teacher_chunk` arithmetic, while writing only
`main_homogeneous_train.npy` and `main_homogeneous_valid.npy`. At 11,949
fitting row groups, the float32 payload is exactly 1,911,840,000 bytes. The
pipeline preflight reserves a further 16 MiB and preserves the 20 GiB floor.
The full profile remains the default.

The remote consumer snapshot requested `variant="homogeneous"` and
`require_families=False`; the current validator checks those two arrays and
still requires all four declared source families and the 0.8125 ancestry.
Tests cover that acceptance path, reject source/family requests from the compact
cache, and compare the compact homogeneous arrays bitwise against the full
producer on a bounded synthetic fixture.

## Frozen remote code and inputs

Stage the committed builder and runner together in a new remote child source
directory, leaving `/root/wnn/scripts` unchanged. The runner expects its sibling
`build_distill_cache.py`, uses `WNN_PROJECT_ROOT=/root/wnn`, and imports the
already-deployed `scripts/distill_pipeline.py`, `scripts/train_rnn.py`, and
`scripts/predict_run.py` from that root.

The exact local source files to stage are bound by these SHA-256 values:

- Compact builder: `scripts/build_distill_cache.py`,
  `2db2d905bd6970b271d26fb4291b571c288789b0821a57af76b7ecc5738c1cd8`.
- Shared retry runner: `scripts/ood_student_pipeline_shared_r1.py`,
  `b88b63c5676b9d86efac2fccc0d278d468ee6c2e8b543a2e049d6bc1201e75ac`.

The source snapshots collected read-only from owned instance `55080658` are in
`remote_snapshot/`, bound by `remote_snapshot/source_receipt.json` and
`remote_snapshot/dependency_receipt_r2.json`:

- Original `ood_student_pipeline.py`: SHA-256
  `1a3e680851415685982ac60f46ca1d8fd25cd0545c165154b6d8426193c3d6e0`.
- Deployed `distill_pipeline.py`: SHA-256
  `dd663a27e92ec7c0fb11542935a89e5011e83a32097335185ed14e39c019c449`.
- Deployed `train_rnn.py`: SHA-256
  `8c041bf2a49b43ef5a014a5f7cee59302605d464f5b82fe8c7dc12267c386a40`.
- Deployed `predict_run.py`: SHA-256
  `46279845540e680940c4444124164102d2499e0468aa7767fc5ba64f90854018`.
- Deployed consumer `src/wnn/distillation_cache.py`: SHA-256
  `403fd74db91f08bb255db147fb50f7f6b6b9702280350aaa2479c01a7f972f19`.
- Deployed full cache builder before this repair: SHA-256
  `09f7110f21507195199d1f50fad6d1074fef1cd4d670a91093935be333f9dd26`.

The runner verifies these exact deployed source hashes before building a cache
or starting a fit. The `predict_run.py` hash was re-read from the owned worker
after adding its exact prediction entry point to the dependency receipt.

The runner's cache path is
`/root/wnn/cache/dc_ood_fold1_strong2_ep2_homogeneous_r2`; all four arms use
the same exclusive lock `/root/run/locks/ood_shared_fold1_strong2_ep2_homogeneous_r2.lock`.
The validated metadata and `shared_cache_receipt.json` bind the fold, source
checkpoint and provenance hashes, prediction hashes, target hashes, and
coefficient records. No code path deletes the shared cache, alias tree, or
completed source predictions. If a `.partial` cache or alias path exists, the
runner fails closed for inspection instead of deleting it.

The two OOD teachers are the completed checkpoints
`combo_scratch_ep3s4_onecycle_f1@ep2` and
`combo_scratch_ep3s5_onecycle_f1@ep2`; the warm start remains
`c20261003_combo_s3_f1@ep3`. The pipeline requires all four existing train/valid
prediction arrays and sidecars and never launches a teacher export.

## Four retry arms

Each command creates one new, unique run and reuses the shared cache. The
script fixes seed 3, fold 1, two student epochs with checkpoint epoch 1, the
original AdamW/WMSE training settings, OOD mode `mix`, and alpha 0.7. Student
names are `vt_f1_ood_<arm>_homshare_r2`.

| Runner arm | OOD augmentation | OOD fraction | Student run |
|---|---|---:|---|
| `splice_r2` | `splice` | 0.5 | `vt_f1_ood_splice_r2_homshare_r2` |
| `spmix_r2` | `splice,mixup_paths` | 0.5 | `vt_f1_ood_spmix_r2_homshare_r2` |
| `swn_r2` | `splice,warp,noise` | 0.5 | `vt_f1_ood_swn_r2_homshare_r2` |
| `swn25_r2` | `splice,warp,noise` | 0.25 | `vt_f1_ood_swn25_r2_homshare_r2` |

After independently checking staged source hashes and current worker admission,
run each as its own root-owned bounded attempt. The daemon's per-step timeout
has previously failed to bound process lifetime, so wrap the worker directly:

```sh
WNN_PROJECT_ROOT=/root/wnn timeout --signal=TERM --kill-after=30s 12h \
  /root/wnn/.venv/bin/python -u \
  /root/wnn/research/ood_cache_repair_20261011/frozen_r1/scripts/ood_student_pipeline_shared_r1.py \
  --arm splice_r2
```

Repeat with `spmix_r2`, `swn_r2`, and `swn25_r2`. Preserve each attempt's log,
student provenance, checkpoint and validation prediction export. Do not interpret
this packet as evidence that any retry has started, completed, or improved a
metric; the actor only prepared the bounded code and tests.

## Local verification

Using the main virtual environment under a 1 CPU / 1.2 GB memory / 180-second
limit:

- `tests/test_distillation.py`: 11 passed.
- Ruff check and Python byte-compilation passed for the changed builder,
  runner, and tests.
- Both CLIs render `--help` successfully.

No GPU training, remote cache creation, teacher export, provider mutation, or
OOD score measurement was performed by this code-only repair task.
