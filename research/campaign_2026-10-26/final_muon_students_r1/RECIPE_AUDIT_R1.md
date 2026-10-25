# Independent recipe audit — h390m final students (2026-10-26)

Read-only audit by a separate Claude subagent (CPU only), after launch preparation, before any fit.

| Check | Verdict |
|---|---|
| Args vs `vt_all_h390_ds_homogeneous_s3_r3` provenance (except name/optim/precision; same parser defaults R2=R3) | PASS |
| R2-copy vs R3 training numerics (diffs only in contract/cache-reuse wiring) | PASS |
| Child Muon == validated Muon (Nesterov lerp, per-gate NS ×3, 0.2·√max(r,c), decoupled WD, OneCycle β1 cycling, unscale→clip→step, fp32 disables autocast/GradScaler; NS fp32 after patch) | PASS |
| Launcher monkeypatch effective before runpy; no child code resets TF32 | PASS |
| Cache arrays bitwise equal to historical R2 pins | PASS |
| Provenance records the pristine R2 snapshot hashes, not the patched files | RISK → recorded here |
| FP32 activation memory ~2× historical FP16 (unmeasured vs 4.5 GB guard) | RISK → watch first epoch |

Patched-source identity (supplements run provenance):
- `H390M_SOURCE_PATCH_R1.diff` sha256 `f0c861f166a5061c9e2ecc0d78ee5f083d64bb822f190ff4a93fd9d7875ac604`
- `strict_fp32_launch.py` sha256 `4729586a6caae64e787009b3b9b96e876e900b493e89029d3c62e7bcff80ed48`
