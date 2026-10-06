# Live frontier

Updated 2026-10-06T20:42:42.287676+00:00. Contract: [CONTRACT.md](CONTRACT.md).

- Own public best remains **0.6660**; Maxim's reported **0.6674** is the source reference. No new submission or scored candidate.
- E501 resource profile completed: exit0, no OOM, peak job RAM4.52GB, aggregate5.70GB, observed VRAM2.16GB. Its two post-warmup batches yield stage timings but no valid whole-step throughput estimate. The first admission failed on a controller bug before training; that evidence and the tested fix are preserved.
- E502 C0 seed3 full continuation is running from immutable source69b23318, 3 CPU/8GB job cap under the aggregate10 CPU/20GB/5GBVRAM limits. Seed4 is preregistered next. Fixed ep1 after two additional epochs; no intermediate evaluation.
- The family-distillation implementation is committed separately and under independent review. [Source families](SOURCE_FAMILIES.json) bind the existing checkpoints and the clean baseline still to train.
- Private Kaggle GPU v2 is running; TPU v1 was pushed and queued. Terminal hardware/results are still pending. GPU v1 failed because sibling files were not mounted; the one-file revision is a new immutable attempt.
- DeepSeek produced five independent random-literature proposals, now H181–H185 in the tree. Solar Pro4 has completed another five; directed literature tasks continue in fresh contexts. OpenCode's403 and the interrupted GLM attempt are preserved.
- [EDA notes and saved plots](EDA_NOTES.md) describe historical teacher disagreement. These are diagnostics on exposed data.

Next: finish paired C0, audit the new trainer, run own baseline teachers and missing cache exports, then admit the matched C1–C4 slate. Update unknown gain priors only from measured paired endpoints.

The tree preserves all original ledgers and catalogs live files. Its structure/raw-row checks passed; catalog freshness drifted when a running heartbeat changed. Rebuild after terminal evidence arrives rather than treating a mutable live catalog as immutable.
