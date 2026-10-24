---
kind: review
status: complete
created_utc: 2026-10-24T13:20:44Z
updated_utc: 2026-10-24T13:20:44Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RECIPE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_SOURCE_MANIFEST_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/direction_probe_math.py
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/run_direction_probe.py
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_SYNTHETIC_R5.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/ASTRA_CLARIFICATION_R1.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/SPEC_REVIEW_R1.md
---

# Independent code and identity review — direction probe R1

## Disposition

**No code blocker to root admission found.** This review is source/metadata-only: I did not open real prediction or target payloads, import the runner or native code, or run an end-to-end grade. The actor's R5 synthetic receipt validates math and moments on generated arrays; it is not real-data parity evidence. The recipe remains `ready_for_root_admission`, with zero fits and no endpoint/alpha selection.

## Frozen identity

Actual file hashes derived from the current bytes:

- Recipe: `1db1dfd44835956f4d297c469d2e1d9e6cbd6de9d9f63cd28c74510b9c314c65`
- Source manifest: `ba7bafea473ae81a64921fbc145b7360577a2a6a27d5b8d6d9af297df02d7951`
- Math: `7253701f9cf3bed4668a729a3a8b5447f1a4da6edfe50ddabae1bfd044c3d24e`
- Runner: `9e3174a3bd2cd1f5c243e4437087559af13230cda8b5ee360ba3898e237afcd0`
- Synthetic checker: `09c117e35476751cb20c13c7e4ad22163b00238f2b71825860d349e448ed9139`
- R5 synthetic receipt: `02fdaac2d855831729e7df68079c33aee6c17e27210a5e04f34f10cfeacda81c`

The recipe's manifest SHA matches the manifest bytes. The recipe `source_sha256` map equals the manifest `files` map; all 26 pinned files exist and match their hashes. R5's math, runner, and checker hashes match the current sources. The earlier synthetic R1–R4 receipts are preserved; R5 is the current pinned receipt.

## Findings

The math follows the reviewed estimand: fixed target-specific weights; right derivatives through prediction clipping at alpha 0 and real-path alpha 0.1; per-target weighted Pearson derivatives averaged only after each target's own score; and alpha-0 calibration / calibration-plus-A projections. Full weighted cross-products and stable centered component moments support bootstrap reconstruction. Rank uses eigenvalues of the standardized weighted correlation Gram with the frozen relative cutoff. Projection identities and norm shares are checked. The exact prediction-SD epsilon stays-closed branch now reports derivative zero but remains nonlinear for projection, and the synthetic receipt covers both open and closed cases.

R5 reports finite-difference error about `3.6e-8` at step `1e-6` for both derivative points; direct-row versus centered-moment parity is `2.1e-17`, raw-moment parity is `2.2e-16`, and projection reconstruction error is about `1.3e-18`. Its component fixture exercises rank 1 to 2. The import-path check resolves both preflight modules without executing their bodies, so this does not establish runner startup or real-data behavior.

The fixed two-cell amendment is explicit: F0 has five unique predictions across 468 groups / 141 components; F1 A is mandatory as a separate source-qualified cell across 457 groups / 149 components, with its R2 preflight and grade already pinned. The runner re-runs both label-free preflights, compares their reports to saved receipts, checks component/group membership, validates all eight prediction hashes and NPY headers before target materialization, reproduces prior raw and float32 alpha-0.1 scores to `1e-12`, then verifies target and prediction hashes again after scoring. F0 A lag-16 is counted once through its frozen exact-alias hash. Source and artifact ancestry pins are complete.

Target extraction uses the existing 1-MiB chunked physical-scratch reader and memory maps; predictions are memory-mapped and moment tables are small. Root must run through the admitted one-core / 400,000,000-byte / zero-swap / 900-second wrapper. The runner checks that scratch exists but not its free space. The pinned target cache is 337,186,198 bytes; its expanded target and mask members require about 337,140,000 bytes plus NPY headers. **Admission should confirm that physical scratch has that space plus operating headroom.** The run has no measured end-to-end memory or CPU receipt yet.

## Conclusion

The source math, fixed-endpoint comparison, ancestry reuse, pre/post identity checks, and final source pins pass static review. Root admission should independently bind the recipe, verify manifest/map equality, and confirm scratch headroom before any authorized score pass. No conclusion about scientific direction or endpoint quality follows until that admitted run completes.
