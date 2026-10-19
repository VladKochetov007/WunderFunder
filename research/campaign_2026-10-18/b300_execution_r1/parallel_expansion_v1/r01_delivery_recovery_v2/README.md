# R01 delivery recovery v2

This packet validates and delivers existing R01 fits. It has no fit/admission command and never writes into an original `runs/<job>` directory. Fit validation emits a fresh isolated view under `state/r01-delivery-recovery-v2/fit_views/`; the view links the immutable checkpoint artifacts and input data, and contains only recovery receipts plus read-only links.

The known actor issue is a producer source-identity mismatch: the frozen portable source manifest contains 61 files while the trainer records the same code with `src/wnn/training/README.md` omitted (60 files). Validation first verifies every portable file against the frozen 61-file manifest, then derives the 60-file map by removing only that README and compares it byte-for-byte with trainer provenance. It also verifies the actor resource receipt, exact panel command and split, all six withheld-evaluation epoch records and counters, all checkpoint hashes, and semantic equality of `model.pt` with fixed `model_ep5.pt` using the separately pinned checkpoint identity helper. A fit that failed during training, has incomplete outputs, or has any other actor error is rejected.

The isolated fit-view receipt uses `tabdata-l4-fit-producer.v1` because the frozen R01 `export_contract.py` accepts that historical schema, while the frozen R01 trainer emits `tabdata-b300-fit-producer.v1`. This is a schema-only translation in the new recovery view after the fit has been independently revalidated; it does not alter original producer receipts or fit outputs.

Run validation once per existing panel cell:

```bash
/data/epanov_s/tabdata/.venv/bin/python -B \
  /data/epanov_s/tabdata/research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/r01_delivery_recovery_v2/r01_delivery_recovery_v2.py \
  validate --root /data/epanov_s/tabdata \
  --packet /data/epanov_s/tabdata/runtime/b300-r01-v1 \
  --source-root /data/epanov_s/tabdata/runtime/source-a266-lf-subset-v1 \
  --recovery-root /data/epanov_s/tabdata/state/r01-delivery-recovery-v2 \
  --job-id tabdata_b300_r01_v1_wide_f0_s1003
```

Only after validation succeeds, export that cell from its existing fixed epoch 5 checkpoint. The root controller must assign exactly one B300 UUID and its normal cgroup/CPU limits; the CLI refuses export without a single `CUDA_VISIBLE_DEVICES=GPU-...` assignment. The frozen `gpu_entrypoint.py` enforces one visible B300 and a 6 GiB allocator cap, and its resource receipt is checked before the cell is eligible for grading.

```bash
CUDA_VISIBLE_DEVICES=GPU-<assigned-uuid> /data/epanov_s/tabdata/.venv/bin/python -B \
  /data/epanov_s/tabdata/research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/r01_delivery_recovery_v2/r01_delivery_recovery_v2.py \
  export --root /data/epanov_s/tabdata \
  --packet /data/epanov_s/tabdata/runtime/b300-r01-v1 \
  --source-root /data/epanov_s/tabdata/runtime/source-a266-lf-subset-v1 \
  --recovery-root /data/epanov_s/tabdata/state/r01-delivery-recovery-v2 \
  --job-id tabdata_b300_r01_v1_wide_f0_s1003
```

After both same-fold/same-seed arm exports pass, `grade` runs the original CPU-only R4 paired grader with the deep endpoint as candidate and wide as reference, at epoch 5 and `global_wp_clip2_v1`. It clears `CUDA_VISIBLE_DEVICES`, saves the official result and logs in the recovery tree, and never refits or replaces either endpoint.

```bash
/data/epanov_s/tabdata/.venv/bin/python -B \
  /data/epanov_s/tabdata/research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/r01_delivery_recovery_v2/r01_delivery_recovery_v2.py \
  grade --root /data/epanov_s/tabdata \
  --packet /data/epanov_s/tabdata/runtime/b300-r01-v1 \
  --source-root /data/epanov_s/tabdata/runtime/source-a266-lf-subset-v1 \
  --recovery-root /data/epanov_s/tabdata/state/r01-delivery-recovery-v2 \
  --wide-id tabdata_b300_r01_v1_wide_f0_s1003 \
  --deep-id tabdata_b300_r01_v1_deep_f0_s1003
```

Each recovery phase refuses to overwrite an earlier attempt. A failed export or grade leaves its command, logs, and partial files intact; a retry needs a separately named, root-authorized recovery state. The entrypoint is a single-cell delivery utility, not a queue controller. No B300 actor or fit was launched while preparing this packet.

Actual integration: helper revisions were preserved as runtime/r01-delivery-recovery-v2, v2_1, v2_2 and v2_3. The final variant binds the canonical trainer's derived split hash, distinguishes ten optimizer windows from one full-sequence batch, and passes the exporter an absolute bundled actor path. Root validated the four completed wide seed1003 fits without refitting; wideF0 export succeeded in76.06s with1.10GB peak allocated /1.36GB peak reserved GPU memory. This is endpoint delivery, not a completed paired comparison; deep fits and independent paired grades remain necessary. The finite delivery watcher and publication addon are separate components, never a fit retry mechanism.
