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

## Result (graded 2026-10-26 ~01:00 UTC) — parent numerics cost nothing

| Cell | strict FP32 Muon (R3) | FP16 AMP | FP16 + bf16 NS | AdamW FP32 (R3) |
|---|---:|---:|---:|---:|
| F0 / 2001 ([grade](grade_f0_s2001.json)) | 0.662119 | 0.662538 | 0.662559 | 0.658505 |
| F2 / 2002 ([grade](grade_f2_s2002.json)) | 0.686365 | 0.686404 | 0.686388 | 0.683105 |

Paired: FP16 − FP32 +0.00042 [+0.000005, +0.0012] / +0.00004 [−0.00009, +0.00017];
bf16 NS − fp32 NS +0.00002 / −0.00002 (both intervals ≈ ±0.0001). FP16+bf16 NS Muon − AdamW
+0.0041 / +0.0033 (third replication of the Muon gain).

**Decision:** the production parents' Muon numerics are not a handicap; retraining them cold under
strict-FP32 Muon is not expected to help (deprioritized). Strict FP32 stays project policy for
comparability, not for quality. The production lineage already holds the Muon gain in its parents,
which explains the small stage-6 transfer (+0.00026).
