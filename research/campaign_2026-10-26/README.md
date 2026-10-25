# Campaign 2026-10-26 — Muon submissions (best single GRU and blend)

User (2026-10-26): "Go on with research. form the best submissions with new information … make
best possible single GRU and blend." Resources: up to 20 GB RAM, 5 GB VRAM, 10 CPU cores.

**Basis.** Strict-FP32 Muon beats AdamW on the fast recipe (+0.0038/+0.0043; vs tuned AdamW
+0.0026/+0.0035) and is **replicated** on Kaggle with fresh seeds, T4 and true FP32
(+0.0036/+0.0033; [R3](../campaign_2026-10-25/kaggle_muon_repro_r1/README.md)). Mechanism and full
evidence: [October 25 synthesis](../campaign_2026-10-25/SESSION_SYNTHESIS_R1.md).

**Contract (competition mode, tier B with exposed folds).**
1. Transfer evidence on the production student lineage: Muon stage-6 warm distilled student, F0,
   vs reused strict-FP32 AdamW control 0.6812108 ([runner](../campaign_2026-10-25/optimizer_panel_r1/run_stage6_muon.sh)).
2. Final all-data students: exact H390 S3/S4 chain (parents `vt_all_fin_mu256p0e4_s{3,4}@ep3`,
   0.8125×mean of three Muon-all H384 teachers, α 0.9, 6 epochs, LR 1e-4, since-weight 32) with
   only `--optim muon --muon-scope rnn --precision fp32 --strict-fp32-kernels` changed
   ([final students](final_muon_students_r1/README.md)).
3. Packages: best single GRU = Muon final student(s); best blend = TL3 with the Muon student(s) in
   the student slot at fixed TL3 coefficients (same rule as the Oct-24 quota ZIPs). Native-ZIP
   checks (CPU replay, A–B–A reset, size/runtime) before any upload decision; no automatic upload.
4. Falsifier: Muon stage-6 F0 ≤ AdamW control → do not promote Muon students into the blend.
