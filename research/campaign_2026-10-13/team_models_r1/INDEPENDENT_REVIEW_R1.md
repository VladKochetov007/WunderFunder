# Independent source and saved-prediction review

2026-10-13; native roles `team_source_critic` and `team_score_critic`, both gpt-6-luna/max, separate fresh-context sessions. No fitting or edits by reviewers.

## Source critic

Verified original archive hashes and all four weight payload identities. Pair_ftt2 includes the exact two students from the historical 0.6674 distilled_h256_pair submission plus two distinct fttall weights. Xteam contains two of these teammate weights and the exact own C endpoint's Z4/H390s3 weights. Thus both supplied ZIPs execute four recurrent models, not two. The user may refer to a newer two-model endpoint; the currently available old pair does not establish which latest model he meant.

The October6 teammate branch supplies final student training using validation labels, and the own endpoints also have all-data/valid exposure. Later fttall source revision, teacher/initializer lineage and clean fold counterparts were not found. Those unknowns prohibit a held-out claim for the final ZIPs.

## Scoring critic

Under one CPU/500 MB/120 seconds, independently re-scored the saved original pair_ftt2 first-two-group NPY against actual Parquet rows using canonical `src.wnn.gmetric` and `reference_global_wp`. Shape `[2,20000,2]`, float32, finiteness, sequence IDs, step order, warmup, per-group feature/target/mask/prediction hashes and NPY SHA all match. Selected rows:2599 and611. Group scores:0.7590888578259993 and0.6348973566695251. Pooled t0=.6882738529492145, t1=.762850331197937, WP=.7255620920735757 reproduce exactly.

Reviewer did not rerun callbacks: the profile's606916 KiB process peak exceeds the review's500 MB cap. Callback/reset evidence is the actor's original execution, not independent inference replication. This check supports saved-prediction/data/metric consistency only, not full replay or CV quality.

## Corrected limitation

The original runner checked only Parquet byte size before inheriting the intake SHA. Full replay now also streams and verifies the whole-file SHA before inference. The original profile source is preserved as `score_archive_profile_r1.py`; existing profiles remain unchanged and bind their original hash. Per-sequence consumed-array hashes continue to be recorded. No model, metric or prediction computation changed.
