# E412 full-data combo package ready for user upload

**Public update — October4:** E412 (`10tq248b`) scored **0.6655**, +0.0014 over0.6641; runtime37m47s, memory625MB. The target gap is0.0005. User reports1/5 submissions used today. See `E412_PUBLIC_RESULT.md` and `E412_PUBLIC_RESULT.json`. This supersedes earlier statements below that public transfer was unmeasured.

Candidate: `submissions/c20261004_combo_s3_i16/submission.zip`.
SHA256: `21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb`.
Fixed epoch3 combo parent plus fixed epoch0 head, equal mixture; development TRAIN+eligible VALID only. Historical holdout excluded from fitting and selection.

Actual i16 callbacks pass unchanged max-error3e-4 and WP5e-6 gates on20 full streams, reset replay and v2/v3 identity. Observed max error0.0001039505, WP delta−1.58286e-7. Original q16x raw-error failure remains recorded; it was not waived.

Clean timing: five alternating fresh-process pairs, three full20k streams per trial, CPU15. Combo i16 median85.271907us/row and303.449557kcycles/row; exact public q16x reference median63.211964us and243.263741kcycles. Ratios1.348984 wall and1.247410 cycles. Anchoring to the user-verified1937-second reference yields43.55 and40.27 minutes respectively. These are uncertain cross-hardware projections, not official runtime measurements or a runtime guarantee.

Exact reference ZIP SHA256 `d986e3f22c38dc68670fbd763d9ff73ffd0c5b86dcdcff52bc2213a6db5ff26b`; user portal reference2m6fzkn0 scored0.6641. New combo public score is unmeasured. No competition upload performed.

Evidence: `E412_NUMERICAL_REVIEW.md`, `checks/e412_i16_package_verification_v2.json`, `checks/e412_clean_package_timing_v1.json`, sealed specs and runner attempts. Local four-fold complete-predictor evidence belongs to E410 and is conditional on exposed evaluation folds.
