# Active frontier

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
