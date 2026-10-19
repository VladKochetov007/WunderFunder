# R9: carried recurrent history changes the fixed GRU's outputs

Rebuilding the GRU state from only the last256 rows changed its next128-row predictions by **0.128402 RMS**, or **19.354% of the full-history prediction RMS**. Starting from zero at the anchor changed them by0.315698 RMS (47.585%). All16 windows had recent256 relative RMS differences above the predeclared1% descriptive reference; the range was3.366%–73.709%, median8.349%. These are output sensitivities, with no target values or quality score involved.

| State at the anchor | Target0 raw RMS difference | Target1 raw RMS difference | Pooled raw RMS difference | Pooled relative RMS |
|---|---:|---:|---:|---:|
| Rebuilt from recent256 rows | 0.1471437735 | 0.1064091038 | 0.1284020784 | 0.1935396175 |
| Zero | 0.3870544691 | 0.2225284993 | 0.3156977154 | 0.4758491130 |

The fixed [design](DESIGN.md) used one unchanged R01 F0 wide FP32 GRU checkpoint, the first8 fitting components of the existing R3 panel, anchors8192/16384, and128-row rollouts. All branches received exactly the same canonical causal211-channel features, including their intact feature history. Only the recurrent carry differed. FULL remained continuous from row0, RECENT256 began from zero256 rows before the anchor, and ZERO began at the anchor. This therefore measures dependence on recurrent carry older than256 rows conditional on the shared features; it does not erase history already encoded in those features.

The recent256 state remained different after128 identical new rows: per-window relative state L2 differences ranged0.108273–0.383879 (median0.151840). Zero-state differences then ranged0.121520–0.587692 (median0.284738). Raw and clipped prediction summaries were identical in this run. The16 windows and8 components are a fixed descriptive panel, not independent training replications or a fresh population sample.

The measured consequence is that a256-row reset is a substantial intervention for this checkpoint. This keeps long-context and persistent-context ideas relevant. It does **not** establish that the older state improves labels, distinguish dynamic history from sequence identity or other persistent context, or show that reset states are typical of the trained model's state distribution. It does not select a new architecture. The original target-free matching proposal, immutable first99-row context, energy regularization and stale-weight state mismatch remain untested.

All16 windows completed with140288 model-forward rows,0backward calls,0optimizer steps and0quality grades. The model fingerprint stayed unchanged and every parameter `.grad` remained empty. The24 selected fitting NPY files, including8target files, were hashed/mapped for integrity; target values were not accessed and0evaluation arrays were used. Actual arrays/model/state stayed on Beleriand; captured outputs contain scalar summaries only. [Execution records](records/README.md), [independent compact audit](RESULT_AUDIT.json), [resource reconciliation](RESOURCE_RECONCILIATION.json) and [continuation](CONTINUATION.md) bind the result and its limits. No completed earlier screen was rerun.
