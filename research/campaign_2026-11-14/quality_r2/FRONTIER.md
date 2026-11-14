# Active frontier

## Effective R2 precision correction

The entries below this section describe the superseded initial execution. [Amendment](PRECISION_AMENDMENT_R2.json): legacy Muon had BF16 Newton–Schulz; q14_kd90_s4 was stopped after two epochs plus part of epoch2, preserving checkpoints and 30m41s wall cost. Old wnn-q14-deliver and its thermal guard are stopped. Do not restart that delivery script.

Fixed source48776fba8 passes the bug-specific negative/positive CUDA checks and a 120-update full-recipe profile at20.67seq/s. **Running:** wnn-q14-fit-strict-local-pair serializes q14s_local_kd90_s3 then q14s_local_kd100_s3, with wnn-q14-thermal-r2. **Preparing:** six corrected-source Kaggle fits alpha .5/.9/.9+boot8 ×s3/s4; eight-device-hour cloud cap, no upload yet. Conditional final refits must use the new paired cloud evidence and corrected source. No new ZIP yet in this wave.

## Historical initial execution

- First admitted fit: q14_kd90_s4, fixed ep3 vs l14_kd50_s4; RUNNING in wnn-q14-fit-kd90-s4. Epoch 0 complete, finite updates; strict runtime identity verified.
- If paired improvement supports alpha .9, final-data s3/s4 refits and native packages are the production path.
- Independent scouts/critics: bounded proposals and recipe checks; no delegate owns GPU launch.
- Fresh hypotheses need ledger dedup and frozen configuration before fitting.
- Local cost is accruing from the 14:03:49 EET start; cap 8 GPU hours including exports/profiles/finalists. Final elapsed accounting is collected from unit/log receipts.
- Conditional delivery owner: wnn-q14-deliver waits for fixed ep3, applies the frozen paired gate, then serializes two final-data fits and native package checks. It cannot admit an unsupported endpoint or launch simultaneous local GPU fits.

## Confirmed preparation evidence

- q14_kd90_s4 log config matches l14_kd90_s3 exactly except run name/seed, including teacher provenance ([receipt](RUNTIME_IDENTITY_R1.json)).
- Bootstrap coverage: 59.1% postwarmup training rows, 64.4% clipped-label mass, target perturbation RMS .0962 on 200 train sequences; since-weight omitted in this descriptive diagnostic ([receipt](BOOTSTRAP_COVERAGE_R1.json)).
- Fixed teacher/student mean valid .691898 vs .691442/.691420 ([receipt](ZEROFIT_TEACHER_STUDENT_R1.json)); actual contribution contradicts universal residual-correlation dismissal.
- [Critic](CRITIC_R1.md) verifies bootstrap/EMA semantics; final launcher requires strict argument/membership review. [Grok cards and triage](SOURCES_R1.md) record no-op clipped-target suggestion and false chronological-split assumptions.
- Initial screening allocation is now explicit: one local alpha.9 confirmation + four cloud control/bootstrap fits + one local alpha1 pilot. EMA stays reserve, not admitted yet.
- [Independent scout cards](IDEAS_R1.md) are recorded as proposals. A separate zero-fit critic is computing fixed endpoint sensitivity over connected sequence components; no weights are being optimized on these results.
