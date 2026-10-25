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

## Trainer source (corrected 2026-10-26)

The current tree has no `--final-distill-cache-root`; the H390 students were trained by the
H390 **R3** child (R2 child + startup-guard fix + pinned R2-cache reuse). The finals therefore use a
copy of the **R2 child** (`research_tmp/h390m_child_source_v1`) with the reviewed edits in
[H390M_SOURCE_PATCH_R1.diff](H390M_SOURCE_PATCH_R1.diff): FP32 Newton–Schulz (child used bf16 on
CUDA), the contract admits `(muon, fp32, rnn)` under names `vt_all_h390m_s{seed}_r1`, and the R2
startup bug (`args.fit_subset_manifest`) is removed exactly as R3 did. Training numerics otherwise
equal R2/R3. [strict_fp32_launch.py](strict_fp32_launch.py) disables TF32 (cuBLAS, cuDNN, cuDNN RNN
`ieee`; plus `NVIDIA_TF32_OVERRIDE=0`) and points the child's snapshot verifier at the pristine R2
child tree so packet/cache identity checks still run against frozen bytes. GPU preflight reached
cache resolution (contract and source verification passed).

The cache is rebuilt by the pristine R2 builder from teacher exports: valid exports are the
historical files; train exports are new (current `predict_run.py`, checkpoint FP16 inference as
historically). It is a numerical reproduction, not byte-identical to the historical R2 cache
(digest `0692a2b1…` train) — agreement to be measured.
