## Native execution update — 2026-11-12

Eight full fits launched on all eight L4 cards at 04:04:15 UTC. Each of four arms uses seeds 203301 and 203302: GRU256 depth 1, GRU256 depth 3, TBPTT horizon 2000, and card 05 horizon 4000. The depth and 2000-row arms are declared extensions, not original PR82 cards. All profiles passed 30 updates and state roundtrip checks before fitting. No fixed-E6 success claim yet.

Execution uses the qualified PR80 source commit `6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba`, source SHA `38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d`, rather than the proposal branch's incompatible root training code. K16 raw112, strict FP32, AdamW 3e-4, WD .01, clip 1, 5% warmup and cosine floor .1, six epochs. Horizon 4000 uses B8, horizon 2000 B16, other arms B32/chunk1000; each has nominal 32,000 rows/update. Horizon 4000 has exactly 38,730 updates; the others 38,760 because the final partial batch differs.

Primary validation is the frozen official earlier-train/later-valid partition, split SHA `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`: 10,325 fitting sequences, zero valid ancestry, 1,873 whole valid sequences / 3,528,273 scored rows. Full valid `global_wp_clip2_v1` is recorded after every epoch, with target metrics, checkpoint inventory, fixed train-evaluation cohort, loss, preclip/clipping, LR/update diagnostics, throughput and resource telemetry. Train-eval uses a different row mask from valid and is labelled accordingly. Valid is exposed development data.

At the latest mirrored epoch, depth1 E4 WP is 0.682120793 / 0.682874383. Other arms remain at earlier epochs; these are monitoring observations, not fixed-endpoint comparisons. Existing same-seed K16 E6 controls are 0.683019498 / 0.682629372 and are reused without new fitting. All 18 available epoch graph sets have been saved as PNG and SVG.

Root admission SHA `776d13e9e3bb255d39c3aac7537153e1879375bb16e3d22ac6b533fc71a6caf5`; native controller PID 3844270 / birth 2185337129, detached from SSH. Frozen clock SHA `0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270`: fitting cutoff 07:20:19 UTC, GPU/readout cutoff 08:20:19 UTC. Owned-process birth identities, disk/RSS guards and absolute deadline cleanup are enabled. Foreign GPU4 processes remain untouched.

Muon and since-event weighting adapters are implemented and source-frozen for separate four-candidate profiles. Teacher distillation and the other proposal-only cards remain explicitly pending their implementation/data-ancestry admission; they are not reported as launched.
