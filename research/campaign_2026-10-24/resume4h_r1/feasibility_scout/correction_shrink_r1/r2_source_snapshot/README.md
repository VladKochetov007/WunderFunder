# R2 source snapshot

These three source files are the exact R2 implementation reviewed before the
final pre-label audit found that the metadata gate checked candidate-map size
but not its six expected keys. The frozen R2 recipe, manifest and synthetic
receipt remain in the parent folder with their original hashes. Each snapshot
file matches the corresponding source SHA recorded in
`../CORRECTION_SHRINK_SOURCE_MANIFEST_R2.json`.

R2 was never admitted or run on campaign targets. R3 adds the pre-label key-set
check and a synthetic rejection case for a malformed map; use the R3 packet for
any later admission review.
