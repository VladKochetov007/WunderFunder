# Independent packer review R2 — 2026-11-12

## Decision

**Static review passes for the built artifact’s fixed recipe and member bindings.** The R2 overlay corrects the Source4 history false rejection narrowly; it preserves the other evaluation guards. The ZIP is built, but this review does not clear the separate full-mix CPU runtime/serving gate or make any quality claim.

I reviewed source and archive metadata only. I did not load checkpoint tensors, run tests, construct a model, run inference, or profile the CPU callback. ROOT reports that the seven focused overlay tests passed and that all four strict member loads and W18 array parity passed in the actual export; the repair receipt and README have not yet been updated to record that test run.

## Reviewed identities

- R2 packer: [build_fullmix_from_archives.py](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/package-source/build_fullmix_from_archives.py), SHA-256 `1f07aa2fb0f898d3b3c67cb484bf55f16cce2051cf2d6ab3acdb142d5df20808`.
- Base helper: [package_final_tl3.py](C:/Users/admin/Documents/Codex/reviews/WunderFunder-final-submit-20261031/scripts/package_final_tl3.py), commit `6493556e853e02538d50baed049073eeb717e9d7`, SHA-256 `4ff9d3975285e3c41fa8fa82ceeffa724258b6bbc069c2f540a4f65f7dfda101`.
- Compatibility overlay: [package_final_tl3.py](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/package-source/source-overlay-r1/scripts/package_final_tl3.py), SHA-256 `cb094e932b7f04b6e3eb12621a3f28879bbd5a2a1a9f3f2271e7f6b72ded75b9`.
- Built ZIP: [tabdata-final-fullmix-20261112-r2.zip](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/build/tabdata-final-fullmix-20261112-r2.zip), 14,804,012 bytes, SHA-256 `a0fb8b1e254dd8be94301f6553f11b9e1b1ae758c9543f38aae3f021c944894f`. The on-disk hash matches [its build receipt](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/build/tabdata-final-fullmix-20261112-r2_build_receipt.json).

## Findings

### Source4 history repair

The base validator rejected every history field beginning with `valid_`. The overlay permits only exact integer `valid_rows=248,364,480`, and requires eight ordered complete epochs with exact counters: 980 updates and 249,600,000 forwarded/real rows each. It still rejects all other `valid_*` fields and every `train_eval_*` field. The counter is derived from 12,480 fit sequences × 19,901 post-warmup objective rows; it is not a validation-set metric. The overlay diff changes only this history validation.

The packer pins the base commit/file hashes, overlay hash, and repair receipt hash before loading the overlay. It then uses the overlay’s unchanged strict checkpoint loader for all four roles.

### Checkpoint, fit, and output binding

The ZIP manifest records the frozen W18, tiedlead-s5, tiedlead-s6, and Source4 LSTM checkpoint hashes; all four expected input widths; per-member exported NPZ hashes; configuration/provenance hashes; and the LSTM data-provenance hash. The selected checkpoints are the fixed endpoints (W18 epoch 1/2, tiedlead epochs 3/4, LSTM epoch 7/8). The fit identity binds all 10,607 train plus 1,873 valid row groups, zero independent evaluation members, and forbids independent score claims. The blend is exactly 0.34/0.33/0.33 for TL3, then 0.9 TL3 + 0.1 LSTM, with the declared FP64-to-FP32 casts, Q8/VNNI fast activations, and clipping after the callback.

The packer re-exports W18 tensors and compares the tensor key set and every array to the pinned W18 NPZ reference using exact array equality. It need not require identical NPZ container bytes: the newly exported NPZ has its own SHA bound in the final manifest and build receipt. The final ZIP inventory contains exactly the solution, precision marker, member manifest, four weight archives, and the three pinned native libraries; each file hash is in the build receipt.

### Ancestry evidence boundary

The packer pins the 17-checkpoint archive and manifest plus the authenticated W18 ZIP/evidence archive, W18 checkpoint, model receipt, and source-validation/membership artifacts. The separately retained W18 evidence has a 13-role lineage closure and cache validation with four payload hashes. Together with the final-fit archive and plan, these are the evidence needed for the 18-role DAG claim.

The packer itself does not recompute the complete transitive teacher/cache-parent union for all roles. It validates the four selected checkpoint identities and full-fit membership, while the external archives and receipts carry the broader ancestry. Keep the final-fit plan, archive intake/fit receipts, W18 lineage/cache validation, and this build receipt with the candidate; do not present the packer alone as proof of transitive cache closure.

## Follow-up records

1. [repair-receipt-r1.json](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/package-source/source-overlay-r1/repair-receipt-r1.json) still says the tests were not run, while ROOT reports seven passed. Update it with the exact command/result (and preserve the current receipt hash as historical if the receipt is immutable).
2. The R2 [README](C:/Users/admin/Documents/Codex/artifacts/final-submission-20261112-r2/package-source/README.md) run example points to the R1 packer path while supplying R2-only compatibility flags, so that example would fail argument parsing. Correct the example before using it as the reproducible command.
3. The packer creates `final_model_set_receipt.json` in a temporary directory and deletes it after packaging. The ZIP and build receipt retain its SHA but not its JSON bytes. Preserve or reconstruct the receipt as a sidecar so that hash remains resolvable.
4. The source checkout check pins HEAD and two source files but does not require a clean tracked worktree or hash every imported `src/wnn` module. The current build is bound to its exact source commit and those two package files; for future reruns, enforce a clean source tree or include the imported-module/source capsule manifest.

## Runtime status

The build receipt correctly marks `runtime_gate=pending_fresh_python311_cpu_fullmix_profile` and `model_construction_tests=not run by this packager`. Package creation, hash checks, and W18 array parity do not establish callback import, serving determinism/reset/causality, runtime, memory, or checker acceptance.

