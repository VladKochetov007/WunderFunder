# Independent JAX bridge review

E458 verified 11 small CPU recurrent/gradient/update tests. The subsequent real-data bridge is a separate implementation and is not covered by those tests.

An independent Luna critic identified a material indexing error: the TRAIN propensity cache preserves absolute Parquet row-group indices, including an unfilled corrupt group528, while teacher/target caches compact that group away. The first bridge used compact positions for propensity, shifting every group after528. The actor corrected the lookup to absolute indices, added an encoded-index regression fixture and changed the preflight to include eligible TRAIN groups on both sides of528. No incorrect real-data fit was launched.

Other corrections bind teacher group order to target-cache order, retain cache/source hashes, forbid supervision on excluded evaluation groups, require both official mask columns and declare the CPU/TPU backend explicitly. The bounded preflight uses only the first1000 rows of four training and four evaluation streams; it is not a full20k-sequence validation. Full-size compilation, resource measurements and full-sequence parity remain later gates.

Source inspection found the SBD rule, structural/next-change heads, true-mask renormalization before since-decay, separate EMA carry and stop-gradients consistent with the root objective. The staged AdamW update differs from the campaign's Muon and does not reproduce the full optimizer schedule. New bridge tests and the real-data preflight remain unexecuted at this snapshot. Files are under `staging/jax_tpu/`.
