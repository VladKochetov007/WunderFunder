# H232: teacher capacity × causal event features

Basis: the own E521 scratch GRU384 teacher keeps inputwidth211 and computes eventfeatures before raw-only replaces99channels with zeros. This delivered shape establishes a resource basis for one explicitly different view; it does not demonstrate a predictive width effect.

Candidate: same seed3/4, GRU384×2, four epochs fixedepoch3, WMSE/chunknorm, fold0 excluded, no auxiliary/teacher/propensity; remove only raw-only so causal eventfeatures are live. Freeze code/input/split/source identities and audit eventfeature causality. Normalization changes are a declared consequence of that single input-view change. Reprofile numerical/complete-update resource behavior: cold-column activation statistics can affect fp16 even if tensor dimensions do not change.

Question: does added teacher capacity extract useful event features, and do its targets/residuals improve the fixed student or blend? Compare matching own raw seed checkpoint and independently trained full-view seed, then fix contribution weights before held-out developmental scoring. Measure representation/teacher interactions separately; do not assume teacher standalone gains or ancestor gains add. Existing full-feature small teachers remain relevant history, not automatic exclusion of the capacity interaction.

No fit admitted. Priority after delivering the current wide raw teacher interaction and source-grounded architecture profile; numerical/public priors remain null. Same-seed comparison is one training realization, not a seed variance estimate or a reset of exposed fold0. Additional independently seeded confirmations require their own bounded allocation.
