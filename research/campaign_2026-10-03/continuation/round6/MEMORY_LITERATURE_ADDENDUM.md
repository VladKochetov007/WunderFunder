# Memory mechanisms: additional primary literature

[Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/references/pdf?id=Hy7PKCFCQ), an ICLR 2019 review draft, studies recurrent state staleness caused by parameter lag. It compares stored-state initialization and burn-in and discusses weaknesses of zero-state initialization. The final publication endpoint triggered browser verification during this lookup; the accessible primary draft is identified explicitly here.

Our inference: recomputing a prefix under current weights tests a related mismatch, but our supervised sequential training has no distributed replay actors or off-policy TD objective. Their empirical improvements cannot establish a benefit in this competition. A local effect, a training improvement, and a physical interpretation remain separate questions.

[Unbiasing Truncated Backpropagation Through Time](https://arxiv.org/abs/1705.08209) introduces ARTBP: randomized truncation lengths with compensation factors produce an unbiased gradient estimator under its construction. It reports synthetic and character-model experiments.

Our next falsifier: enumerate all truncation decisions on a tiny fixed-weight recurrent graph and test the expected compensated gradient against full BPTT. Separately test whether clipping/nonlinear optimizer transforms preserve expected update equivalence. Unbiased gradients do not imply equal expected nonlinear updates, bounded variance, or safe AMP/Muon behavior. This is a prospective instrument, not an authorization for full training or a demonstrated competition gain.

Retrieved 2026-10-03, around 22:17 Kyiv. Current state-drift and descriptive AMP numerical probes remain pending independent final binding reviews.
