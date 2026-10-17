# Resolve native versus TPU full-FP32 parity

Status: engineering diagnostic. Lineage `engineering`; `H-DRAFT-TPU-PARITY-20261017` / `E-DRAFT-TPU-PARITY-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. OneTPU update/export ran; reset passed, native parity failed max0.00777 versus1e-4. Cause unresolved.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Engineering diagnostic only: identical operands, explicit backend precision, bounded complete-sequence reset checks. No30-fold quality claim and noGPU/TPU launch in this delivery.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_feature_forward_r4.py](../../../research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_feature_forward_r4.py)
- [research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_native_compat_r2.py](../../../research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/check_native_compat_r2.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
