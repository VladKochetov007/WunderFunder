# Independent E-517 evidence review — recomputation

Reviewer: `gpt-6-luna` reasoning max, separate critic session. No project files
were changed and no fit was launched.

**Artifact/calculation audit: PASS. Preregistered screen: FAIL.** The bounded
review ran with one CPU, 1.5 GB memory maximum, and 300-second runtime. It
revalidated the frozen source ancestry/cache/order gates, 432 groups, 123
chains, 41 chains per fold and group counts 150/130/152. Registered manifest,
partition, row-group, runner, loader, preregistration, minimal executor, plot,
both OOF raw-array digests and both `.npy` file digests matched. The two OOF
arrays reconstruct exactly and are byte-identical.

Independent official `global_wp_clip2_v1` endpoints: anchor, gated and global
all `0.6832264634983685`; target endpoints are `0.6733560416224683` and
`0.6930968853742686`. Fold WPs are `0.7534652370359358`, `0.6454556540269862`
and `0.637239361395834`. All three folds have `beta_gate=0` and
`beta_global=0`. The reviewer independently rebuilt the 2,000 registered,
fold-stratified complete-chain bootstrap replicates with seed `3772181504`;
all three paired intervals are `[0, 0]`, matching the stored result. Gate row
and official-weight fractions matched the report.

All four preregistered conditions fail. The stored `development_gate_passed:
false` is correct. This result is on previously exposed development data and
does not imply public transfer.

Minor caveat: the runner writes the per-chain CSV, but the result JSON does not
embed that CSV's digest. It does not affect the verified endpoint decision.
