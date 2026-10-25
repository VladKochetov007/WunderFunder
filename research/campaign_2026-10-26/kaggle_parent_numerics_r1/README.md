# Kaggle screen: do the production parents' Muon numerics cost quality? — 2026-10-26 R1

**Why.** Both production parents (`vt_all_fin_mu256p0e4_s{3,4}`, `c20261003_combo_s3_f0`) were
trained with Muon under FP16 AMP and bf16 Newton–Schulz ([diagnostics](../diagnostics_r1/README.md)).
If those numerics cost ~0.001+, retraining the parents cold under strict-FP32 Muon is the larger
remaining win; if not, Muon in the distill stage is all that's left.

**Design.** Fast cold recipe (H512×1, 10% memberships, 3 epochs, CP2), cells and seeds matched to
the R3 strict-FP32 Muon controls (`krepro_{f0_s2001,f2_s2002}_muon_r3`, T4): arms
`kparent_*_muon_fp16_r1` (FP16 AMP, fp32 NS) and `kparent_*_muon_fp16bf16ns_r1` (FP16 AMP, bf16 NS
via new diagnostic flag `--muon-ns-dtype bf16`). 4 new fits; reused controls are the R3 Muon fits.
Paired official grade per fold, export FP32 inference.

- Code `280caea06`, [build_specs.py](build_specs.py), [specs](specs/), [bundle manifest](bundle-manifest.json).
- Local GPU preflight: fp16 + bf16-NS Muon ran 90 updates, 1 initial GradScaler skip (the strict
  profile reporter rejects any skip; normal training does not).
- Dataset `vladkochetov07/wnn-parent-numerics-20261026-r1` READY; kernel `…-r1-runner` v1 pushed
  ~23:58 UTC, RUNNING.
