# Final all-data Muon students (H390 chain) — 2026-10-26 R1

- [run_teacher_exports_and_cache.sh](run_teacher_exports_and_cache.sh) (`wnn-tcache`): full train
  exports of `vt_f1_muonall_h384_e3{,_s4,_s5}` (valid exports already present; checkpoint FP16
  inference as in the historical exports), then the frozen H390 R2 builder
  (`artifacts/preservation_20261012/tmp/h390_all_data_homogeneous_r2_b2bee94b.tar.gz`, child source)
  → `cache/h390_all_data_homogeneous_r2`.
- [run_final_students.sh](run_final_students.sh) (`wnn-finals`, after `wnn-muon6` and the cache):
  `vt_all_h390m_s{3,4}_r1`, frozen worktree `research_tmp/final-muon-students-20261026` @ `031989b40`.
- Fitted on all supplied data: no held-out claim; promotion relies on the stage-6 transfer test and
  the replicated fast-recipe evidence.
- Log: `pipeline.log`; thermal guards freeze at ≥92 °C.
