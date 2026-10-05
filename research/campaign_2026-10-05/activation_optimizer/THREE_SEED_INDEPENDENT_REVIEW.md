# Independent review of the three-seed E470 scorer

**Audit result: the score exports and conditional within-seed contrasts are reproducible from the fixed E470 scorer and selected exports. Interpretation remains exploratory: the three training seeds all reuse the same 16 historically exposed test components. No cross-seed confidence interval or public-score extrapolation is warranted.**

## Integrity and scoring checks

The grader retained 16 registered arms for each of seeds 271828, 271829 and 271830: 48 scores and 30 contrasts per seed (22 primary plus 8 secondary), for 90 within-seed paired-component contrasts. Every score uses `global_wp_clip2_v1` on the same 26,990 selected rows and two targets. The scorer checks all raw selected exports for finite values before clipping, requires identical target arrays/count segmentation across arms, validates each test row-group/sequence ordering against the frozen cohort, verifies test components are unique and disjoint from fit components, compares selected target values against the source parquet mask, and recomputes pooled plus per-target official scores using the original E470 metric implementation. Fit code/tests hashes match the sealed E470 fit specification.

I independently checked provenance beyond the grader's completion gate. The original seed-271828 E470 attempt finished with exit 0 and teardown verified; its actual fit-result SHA and byte size match its attempt artifact receipt. Seed 271829 and 271830 attempts also report `finished`, exit 0, and teardown verified; each actual result file's SHA and size match its attempt artifact record. Both new attempts point to the same sealed validation parquet SHA/size and frozen cohort SHA as E470. Their wrapper SHA and preregistration SHA match the current files. The scorer output records the expected original scorer SHA (`b85eb93a…ea3011`) and its own grader SHA (`f5988909…ae145`); these match the files on disk. The overall score artifact SHA is `2ff1f41782182f327b05c1419f1315cab095dea4273fba54652a9917b085ca62`.

The seed grader checks each attempt's completion status and the replication wrapper hash, but it does not itself compare the attempt's artifact SHA/size or the preregistration hash. Those bindings were checked independently for all three run artifacts in this review; both replication artifacts match exactly. The per-seed attempt receipts and checksums are preserved in `THREE_SEED_ACTIVATION_RESULT.json`.

## What repeated and what moved

The strongest repeated point pattern is `gelu_sine_01` versus plain GELU: its pooled-WP point delta is positive in all six seed/optimizer comparisons. For AdamW the deltas were `+0.00172`, `+0.00071`, and `+0.00277` over seeds 271828, 271829, and 271830; their component-bootstrap intervals exclude zero only in the third seed. For MuonAdamW the deltas were `+0.00882`, `+0.00272`, and `+0.00732`; each seed's conditional component interval is above zero. This is a useful predeclared-to-retain follow-up signal, not a general activation result: it is a secondary contrast introduced after the E470 screen, among 30 exploratory contrasts without multiplicity correction, on the same exposed components. A matched `GELU(x)+0.1*x` control is needed before attributing it to sine curvature.

The larger activation-versus-tanh effects do not keep a stable direction. Under AdamW, plain GELU was `+0.01598`, `+0.01511`, then `−0.00671`; GELU-plus-sine was `+0.01770`, `+0.01582`, then `−0.00394`. Under MuonAdamW, every non-tanh activation was below tanh in seed 271828, above tanh in seed 271829, then below tanh again in seed 271830. This full sign reversal is more informative than the cross-seed recipe mean: activation ordering in this tiny screen is seed-sensitive.

Optimizer ordering also moved. Tanh's MuonAdamW-minus-AdamW delta was `+0.02324`, `−0.02436`, `−0.01799`. Plain GELU's Muon-minus-AdamW delta was negative in all three seeds (`−0.00912`, `−0.02964`, `−0.03455`), as it was for the other six activation choices in seeds 271829 and 271830. These are fixed LR `0.001` recipes; matching nominal LR does not match effective update scale, so they do not rank optimizer families.

Seed 271830 scores were higher than seed 271828 scores for all 16 recipes, while seed 271829 mostly occupied a lower band. Each seed's winner also differed: tanh/MuonAdamW in 271828, GELU/AdamW in 271829, and GELU+sine+cosine/AdamW in 271830. The broad common shift and changing winners make training-seed variability a material factor here. The reported cross-seed means and sample SDs are descriptive summaries only; with three seeds, one fixed component panel, and post-screen contrasts, they are not calibrated estimates of generalization uncertainty.

All 48 fits use the same small 8,235-parameter, width-32, one-layer GRU architecture, one epoch/474 updates, and the same 16 historically exposed diagnostic components. This is not the complete two-layer width-256 model, nor an independent validation sample. Component bootstrap intervals condition on one trained seed at a time and do not include seed, fit-selection, or research-search uncertainty. The artifact explicitly has no pooled interval or public forecast; this review preserves that limit.

## Reproduction identities

- Result JSON: `research/campaign_2026-10-05/activation_optimizer/THREE_SEED_ACTIVATION_RESULT.json`
- Result SHA-256: `2ff1f41782182f327b05c1419f1315cab095dea4273fba54652a9917b085ca62`
- Three-seed grader SHA-256: `f598890975766cf6d0a2b9e05a4595c968409bc38a095cf7208123a0361ae145`
- Immutable single-seed scorer SHA-256: `b85eb93af079d1b003b1fcca2e474d39c5bb8942d95b90a699e13d8da4ea3011`
- Fixed experiment/data/cohort identities and all per-arm prediction-export hashes are recorded in the JSON result.
