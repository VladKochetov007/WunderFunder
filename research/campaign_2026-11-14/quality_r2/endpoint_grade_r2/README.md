# Endpoint grade R2

**Status: helper prepared; production grade not run.** The six fixed `q14s_*` prediction arrays and JSON sidecars are not yet in the chronological cache. The grader fails closed and writes no result until all six, both historical KD50 predictions, the official target/index caches, and the archived TL3C output are present and pass identity checks.

`grade_endpoints.py` follows the frozen [selection contract](../ENDPOINT_SELECTION_R2.json). It reuses `local_r1/grade_submissions.py` for row-group mapping and its unchanged `global_wp_clip2_v1` plus independent reference check. The script also pins the selection-contract, scorer, and shared-grader hashes; checks full-valid row-group membership, sidecar identity, prediction hashes/layout/dtype/finiteness, and the official mask; then writes `ENDPOINT_GRADE_R2.json` atomically.

The fixed comparisons are six singles, equal seed-3/4 pairs for a50/a90/boot8, equal three-member mixtures of each pair with `l14_kd50_s3`, and the historical equal `l14_kd50_s3/s4` pair. It records per-seed paired deltas and applies the frozen a90 and boot8 gates. A final-fit recommendation requires a gate-passing arm and an endpoint that beats both the new a50 pair and historical KD50 pair; a mixed3 must also beat its own pair. The archived native TL3C blend is scored separately by its report row-group mapping. There is no coefficient search.

Run after the exact six inputs arrive:

```bash
WNN_CPU_QUOTA=50% WNN_MEM_MAX=2G WNN_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-11-14/quality_r2/endpoint_grade_r2/grade_endpoints.py
```

This grades the exposed full official valid panel (1,873 sequences, 3,528,273 scored rows). It is not unseen-CV significance, fitted-data ranking, public-score forecasting, or model-seed uncertainty. A train+valid final fit remains conditional on `final_decision.admitted`; its score is not held out. Serving and native parity remain separate checks. The root is preparing an original TL3C fbc-only replacement diagnostic separately; it is not an input to this endpoint grade.

Root receiver review corrected the six run names to the actual sealed `q14s_*` identifiers. The helper now requires `kaggle_r2/COLLECTION_R2.json` with six verified complete workers and strict optimizer groups, binds each prediction checkpoint/provenance hash to that collection, and requires exported-versus-worker score agreement within 1e-5. Feature-only sidecar scope and group digest were checked against the executed fixed-source `scripts/predict_run.py` producer. No completed cloud-grade path has been exercised yet. The delegated preparation ended before these root corrections; no independent runtime approval is claimed.
