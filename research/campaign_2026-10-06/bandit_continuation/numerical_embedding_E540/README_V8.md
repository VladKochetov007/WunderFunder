# E540 V8 FP32 paired screen

V8 is a fresh paired rerun of the frozen V7 FP32 treatment. It keeps 512 selected complete components, the exact 468 previously exposed evaluation groups, seed 540, clip2-target WMSE with chunk normalization, the same model shapes and numeric maps, and one fixed epoch-0 checkpoint. Both arms remain FP32 with GradScaler disabled; all profile and fit finiteness and exact-update gates remain unchanged.

V7's T4 profile gate passed for both arms. The linear trainer completed epoch 0 in 414 seconds with 72,996/72,996 optimizer updates, zero skips, 924 batches, 72,996 GradStats observations and zero nonfinite steps. Its checkpoint and provenance hashes are preserved in the V7 terminal collection receipt. The worker then rejected that completed checkpoint because it selected the nested `epoch` count mapping without the sibling `grad_stats` field. PLE fit and both prediction exports did not run; V7 produced no paired comparison or grade.

V8 changes only the worker's receipt assembly: it combines the nested epoch counters with the sibling GradStats mapping before applying the same strict FP32 update contract. A regression fixture is the actual V7 `update_diagnostics[0]` row, extracted from provenance SHA-256 `8082735f26a9989f5f87f8e51b7499e8653b201d60bcfba64d6116e5e0c80975`. No finite-update, count, membership, objective or resource gate is weakened.

This preparation is code-only. It has not been uploaded or pushed, and has no V8 fit, prediction or grade result. Root owns independent source/recipe review, fresh quota admission, provider launch and grading.
