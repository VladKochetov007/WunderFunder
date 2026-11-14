# Original TL3C member replacement diagnostic

This zero-fit diagnostic was admitted before seeing its full-valid outputs. It tests the fixed .34 KD90 + .33 original tiedlead-s5 + .33 original tiedlead-s6 endpoint against the original TL3C .34 fbc1k/.33/.33 endpoint. No coefficient search, fitting, or upload. Full official valid is exposed development evidence; original TL3C training-membership equality with the local reproduction is unverified.

[build_original_fbc.py](build_original_fbc.py) isolates the original event211 member without changing its native runtime or weight bytes. [receipt.json](receipt.json) binds source and diagnostic ZIP identities and retains original checkpoint/ancestor hashes. The resulting `original_fbc_only.zip` is a diagnostic, not a recommended submission. [run_cpu.sh](run_cpu.sh) runs the frozen PR51 receiver on all 1,873 valid sequences in an offline Python 3.11 container, one CPU, 4 GB RAM, physical scratch, at most 3,500 seconds; no GPU and no new fits. `full_valid/report.json` and `full_valid/predictions.npy` are the expected outputs; completion is not claimed yet.

The subtraction formula uses *raw*, unclipped native predictions: TL3C − .34 original fbc1k + .34 KD90. Scores use the unchanged official clipping only after composition. This differs from substituting the locally reproduced teacher, whose checkpoint is not the original member. Floating-point composition and native parity remain to be checked before any recommendation. Any final package must run the actual three models and pass the normal ABI/reset/size/runtime checks. Original assets remain unchanged.

[build_replacement.py](build_replacement.py) has assembled the fixed three-member diagnostic ZIP from the verified original TL3C package and the checked KD90 single. [replacement_receipt.json](replacement_receipt.json) binds the 12,540,686-byte archive, member weights and training provenance. This is preparation, not a measured improvement or release recommendation.

[grade_replacement.py](grade_replacement.py) prepares the raw-prediction substitution score and 2,000 paired resamples of the 579 connected components. It checks native source hashes, reconstructs reported scores with the official/reference implementations, and labels this Torch/native mixture as a proxy until the actual ZIP is evaluated.

[Full-valid output directory](full_valid/README.md) documents the completion/hash guard for its incrementally written predictions and eventual receiver report.

[Independent static review](INDEPENDENT_REVIEW_R1.md) verifies actual ZIP coefficients, weight/runtime byte preservation, proxy arithmetic and source alignment. It makes no empirical gain or runtime approval: original-member replay, proxy grade and exact candidate evaluation must finish first.

The original team TL3 ZIP and the PR51 TL3C ZIP contain the same three native-library SHA256 values (v2/avx2/v3, including v3 `3cb4546229e80cca81cda9cdec1b9c6146b380271077e26f9324d8a4b6ecad9a`). Rebuilding this candidate on the older team native engine would therefore not introduce a different compiled implementation; no duplicate engine profile was admitted on that basis.

## Proxy result and exact candidate check

[Proxy result](REPLACEMENT_PROXY_R1.json): TL3C .6923359001, original native fbc1k .6913220877, replacement proxy .6923634150; delta +.0000275 with a 95% paired-component interval [−.0001855,+.0002365]. This tiny point difference is not an established improvement and may be erased by native arithmetic. [Actual candidate receiver](run_replacement_cpu.sh) therefore evaluates the frozen ZIP on full valid, with one CPU/4GB and a 5,300-second engineering timeout. No new fit or coefficient choice is involved. Original fbc replay passed A-B-A/warmup checks at 555.9MB peak RSS and 1,269.8 seconds on this host.

[Exact candidate output directory](replacement_full_valid/README.md) documents the in-progress native replay and its completion guard.
