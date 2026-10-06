# Recovered library ideation and bandit footprints

The historical jobs are verified by eight saved task prompts in [the extracted JSON](library_ideation_footprint_2026-10-06.json), with source session paths and line numbers. No model private reasoning was copied.

September25: four Haiku generators received the same task brief and research tree, different randomly chosen pairs of documents, and separate output files. They were asked for8–12new ideas including at least3crazy ideas; each required source concept, data analogy, hypothesis, cheapest test, falsifier and research branch. They could write only their report and run no training. Source reports: haiku_1.md (random matrix theory/FX options), haiku_2.md (execution/noncausality), haiku_3.md (TWAP/VWAP/market efficiency), haiku_4.md (word association/CEX–DEX arbitrage).

Later rounds: r2_a–d.md; r3_a.md explicitly preserves an8–12idea prompt with ≥3crazy ideas from greedy selection/marketmaking/optimal execution. research/tree.md:162–174 records its random-book inbox and per-idea triage.

September27: four separate read-only scouts covered microstructure, DL/RL, validation/transfer, and random contrarian PDFs from Documents/Trading/Naive and Плесков. Their prompts and saved synthesis2026-09-27-brainteaser-synthesis.md confirm book-based generation followed by controller-run measurements. Historical models are reported only to reconstruct past jobs; no new agent was spawned.

October2 bandit: scripts/bandit.py ranks18mechanism arms using knowledge-gradient information value per GPU hour. research/bandit/arms.json stores mu,sd,measurement noise,cost,mechanism family,status and ledger-linked pulls. It is a ranking helper with manually updated evidence, not an autonomous scheduling service. research/campaign_2026-10-03/continuation/record_matched_and_falsifier.py updates arm evidence from matched comparisons. The current bandit state explicitly prohibits summing correlated combo/Muon/chunk/bootstrap effects.

Recovered operating pattern: diverse source-grounded ideation → separate mechanism branches → cheap discriminating tests → arm-level evidence/cost updates → branch allocation → matched combination tests. Separate agents shared a common brief/tree; their outputs are organizationally separated, not statistically independent evidence. Bandit numeric priors and horizon are historical and need refresh before future allocation.
