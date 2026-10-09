# R4 source-binding correction

The first sealed D02 R4 candidate is preserved at sibling directory
`d02_muon_loss_successor_r4`. Its source manifest and offline receipt recorded a
mistyped commit SHA supplied to the sealer. No dataset upload or kernel push
used that candidate.

This R2 candidate uses the actual full Git commit reported by
`git rev-parse 3879b18c`,
`3879b18c21f7100bfc4be205c969cd084e54bb0b`. The source files and science recipe
are unchanged; the source manifest, bootstrap, packet, and upload manifest are
regenerated because the source-commit binding participates in the seals.
R1’s packet, asset bytes, and offline receipt are retained unchanged as the
historical failed seal.
