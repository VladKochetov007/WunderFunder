# Final four-stage B300 wave: full-primary refits

| Candidate | Official valid WP |
|---|---:|
| primary-refit-01-s1t001002-gru | 0.682425941 |
| primary-refit-02-s1t000009-gru | 0.681210508 |
| primary-refit-03-s1t000000-lstm | 0.682520594 |
| primary-refit-04-s1t001004-transformer | 0.676479699 |
| equal_weight | 0.684699354 |
| stage4_direct_renormalized | 0.684304429 |
| canonical fresh TL3 | 0.692377425 |

Best new mixture: equal four-member weights, WP 0.6846993539799099. Fixed fresh TL3 reference 0.6923774245652725; delta -0.0076780705853626285. Baseline not beaten. Both mixtures were frozen before the four refits; no validation weight search.

All four fresh refits used the same 10,325 eligible earlier official-train sequences, zero fitting valid sequences. Full 1,873-sequence official valid / 3,528,273 scored rows after warmup99. TL3 is a fixed previously evaluated reference, not rescored here; training recipes/budgets differ. Validation is development-exposed, not independent confirmation.

Forty scientific attempts produced 39 configured-final model artifacts and one partial GALAM. Main auxiliary study: 15 base, 6 fine-tune, 12 paired student/control endpoints; two earlier fine-tune pilots are separate. Four fresh full-primary direct-label refits complete the count. The original ledger retains 9 terminal_unverified exits and a stale GALAM running state; artifact completion and current process absence are separately evidenced. No invented exit codes. 56 bounded engineering profiles, 24 completed CPU Optuna mixture trials.

Independent NumPy/PyArrow arithmetic reads raw official labels/mask: all four singletons and both mixtures reproduce within 1e-10; mixtures are FP32 bit-exact. This does not replicate training or establish a public score.

Native train/valid curves are included. Train diagnostic cohort uses 16 fitting sequences/all rows after warmup; valid uses all official valid/is_scored. Their absolute gaps are not directly comparable. Configured-final endpoints remain fixed despite visible trajectories.

[Four full-primary configured-final weights and SHA-verified assets](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-full-primary-checkpoints-20261031-r1). [33 auxiliary configured-final models and pinned replay source capsules](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr53-multistage-checkpoints-20261031-r1). Both are research prereleases, not a serving ABI/submission. Two historical pilots and partial GALAM are excluded from releases.

GPU_RELEASE_RECEIPT.json verifies no live tabdata GPU clients after training/scoring. Foreign NIER/MPS clients are preserved; no signals were sent. Historical evidence is immutable. Large original JSONs are losslessly gzipped with compressed/original byte hashes.
