# Root source checks and competition contract

Checked 2026-10-17, alongside the two independent source registers. This is a
bounded literature search, not a claim that every relevant publication was read.
Abstract-level checks establish what authors claim; they do not reproduce results.

## Additional primary sources

1. Tilmann Gneiting, *Making and Evaluating Point Forecasts*, arXiv 2009/v2 2010,
   [paper](https://arxiv.org/abs/0912.0902). Root read the abstract and HTML
   exposition of consistent losses/ratios of expectations. Outcome weighting
   changes the optimal point functional. Our weighted-correlation derivation is
   separate mathematics, not a theorem about this contest attributed to the paper.
2. Sam Allen, *Weighted scoringRules: Emphasizing Particular Outcomes When
   Evaluating Probabilistic Forecasts*, JSS 110(8), September 7, 2024,
   [publication](https://www.jstatsoft.org/article/view/v110i08). Root verified
   author, date and abstract. Its proper probabilistic scores are not this
   contest's weighted Pearson metric; the connection is outcome weighting only.
3. Eljas Linna, Kestutis Baltakys, Derrick Manoharan, Alexandros Iosifidis and
   Juho Kanniainen, *Repurposing Deep Limit Order Book Forecasting for
   Scenario-Conditioned Market Impact Modeling*, September 15, 2026,
   [preprint](https://arxiv.org/abs/2609.16930). Root checked title, authors,
   submission date and abstract. The authors test mechanically valid message
   interventions in an interpreted book. Anonymous challenge columns do not
   support those interventions automatically; this motivates a validity audit,
   not fabricated queue perturbations or a transferable accuracy estimate.
4. Adriana Romero et al., *FitNets: Hints for Thin Deep Nets*, 2014/ICLR 2015,
   [paper](https://arxiv.org/abs/1412.6550). The inspected abstract describes
   intermediate teacher representations as student supervision. Our causal
   state-hint transfer is a proposed extension; the paper's image results do not
   establish sequence-state transfer or its success on this task.
5. Aaron van den Oord, Yazhe Li and Oriol Vinyals, *Representation Learning with
   Contrastive Predictive Coding*, 2018,
   [paper](https://arxiv.org/abs/1807.03748). The inspected abstract proposes
   autoregressive latent future prediction with negative sampling. Our proposal
   uses only supplied training sequences and guards against identifying a
   sequence/activity regime instead of learning transferable dynamics.
6. Danijar Hafner, Jurgis Pasukonis, Jimmy Ba and Timothy Lillicrap,
   *Mastering diverse control tasks through world models*, Nature 640, 2025,
   [publication](https://www.nature.com/articles/s41586-025-08744-2). Root inspected
   publication metadata and methods describing latent dynamics and imagination.
   We borrow a possible observation-dynamics architecture, not its RL objective,
   actions, environments, datasets or weights. Our non-interactive stream has no
   controllable actions; causal imagined rollouts would be predictions, not access
   to real future rows. Compounding simulation errors are a major transfer risk.

The root additionally opened the primary [Mamba-3 paper](https://arxiv.org/abs/2603.15569)
and [xLSTM paper](https://arxiv.org/abs/2405.04517), confirming the scouts' central
recurrence/memory claims. Their reported language-model gains are not priors for
our regression score. The two registers contain 26 distinct works; these six
additional works bring the literature register to 32, excluding contest pages.

[`arxiv_search_snapshot.sh`](arxiv_search_snapshot.sh) is an exact copy of the
repository helper at base main; [`arxiv_weighted_search.txt`](arxiv_weighted_search.txt)
retains its bounded `weighted scoring forecasting --max-papers 5` query result.
Returned search hits were not automatically treated as verified or adopted sources.

## Live official checks

The SPA pages were read through the in-app browser because the text web reader
returned no body. [Rules](https://wundernn.io/connectome/docs/rules) confirm supplied
data only, publicly available pretrained models without external market data,
offline inference, five daily submissions and final ranking on a private dataset.
No submission or public strategy sharing was performed in this work.

[Data overview](https://wundernn.io/connectome/docs/data_overview) confirms 112
features, 20,000 rows per sequence, 99 warm-up rows, sequence-local causal state,
and the pooled clipped weighted-Pearson metric. Bid/ask group membership **is**
specified. Within a side, column indices are stable component identifiers;
they do **not** specify depth or best quote. A critic's blanket rejection of
bid/ask side semantics is therefore too strong. Trade price/volume group names
are supplied; a particular feature change is not necessarily a physical trade.

All hypothetical mechanisms must return both outputs on every required row,
update causal state including unscored/warm-up rows, reset between sequences,
and avoid true lagged competition targets or hidden scoring masks as inputs.
