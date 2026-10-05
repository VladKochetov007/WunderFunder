# Independent review: E486 raw-export reproducer

Reviewed the initial `reproduce_counterfactual_exports.py` source
`cc4c58674a7f6d6824efe412f2dcbe4b22987a7a43f129aac52c92454b3c89d8`, then
re-reviewed the corrected final source
`ef258c69333cba2a72e68dafb22d042d33764665cc10cdd03819c0f77892f01b` and its
shared arithmetic helper `reproduce_activation_exports.py`. This was a source
and synthetic-only review. I did not open E486 `scores.json` or NPZ exports
and did not run the reproducer on actual exports.

The pair/alias mapping is consistent with the E486 scorer. For each of the
nine seed/momentum pairs, the four aliases map to
`seed…__train_{baseline|sine}__eval_{baseline|sine}__beta…`; the full grid has
36 unique keys. The helper clips raw targets and predictions to [-2, 2], uses
absolute target weights, builds additive moments around a fixed shared anchor,
and computes pooled target WP. Resampling multiplies all rows from each
component by the same component multiplicity, so the generated 2,000 draws
are shared across the four cells and every paired contrast. The independent
formula agrees with direct row-weighted Pearson on a synthetic uneven-size
16-component panel, including every bootstrap draw. Synthetic cell-key and
alias checks also passed under one CPU/2 GiB.

The corrected source adds the missing fail-closed checks: the exact nine
seed/momentum pairs, four canonical cell aliases per pair, all six registered
contrast labels and formulas (including `(D-B)-(C-A)`), 36-cell/54-contrast
counts, metric identity, selected total, component IDs and row groups. It
regenerates seed-271828 multiplicities and compares them exactly with the
exported matrix, then uses that shared draw matrix for every cell. It also
refuses to overwrite the fixed receipt. This closes the issues in the initial
review.

The common-anchor moment method itself is sound for pooled weighted Pearson:
subtracting the same anchor from every selected row preserves the correlation,
including when components are resampled with replacement. `scores()` uses the
same clipped target weights and averages the two target correlations to form
the global point and bootstrap values. The script compares per-target/global
cell values, cell bootstrap vectors, contrast point/CI/SD/tail probability,
and exported contrast draws with a `1e-10` maximum absolute tolerance.

One output-safety detail: the script writes its fixed receipt path without a
preexisting-file check or atomic temporary rename. Refuse overwrite (or use a
unique output path) before executing it so a later run cannot silently replace
an earlier arithmetic receipt.

Synthetic command result: direct weighted-Pearson point and all 2,000 paired
bootstrap replicates matched at absolute tolerance `2e-12`; 36 unique cell
keys and the `sine_baseline` mapping passed. Command was run through
`scripts/limited` with `WNN_CPU_QUOTA=100%`, `WNN_THREADS=1`, and
`WNN_MEM_MAX=2G`. No actual export arithmetic was run.

**Disposition: GO for root to run the corrected reproducer on the completed
E486 exports.** The exact registered inventory, formulas and exported draw
identity are now checked. No E486 result was scored or accepted by this
review.
