# What this snapshot covers

Copied source data from **2026-09-30T21:07:38.170820+00:00** through **2026-09-30T21:08:00.928179+00:00** (UTC; Moscow is UTC+3). Compression verification completed at the time recorded in `verification.json`. This is a stable copy of an evolving remote workspace, not a transaction that paused live training.

- **266 remote run directories**: 261 contain files; five are empty failed/aborted attempts whose directories and available logs were preserved. These are indexed in `run_index.tsv`, `run_index.json`, and individual `run-summaries/` JSON files.
- **860 checkpoint files**, including fixed-epoch and best-on-validation files, are inside the archive. They are not interchangeable; consult the corresponding run configuration and log.
- **261 experiment-ledger records** in `research-history/main/experiments.jsonl`. These are research events, including analyses and packaging, rather than one record per model run. Later campaign work is also recorded in `research-history/main/quant-probes-2026-09-29.md`, run logs and result artifacts. Therefore the ledger count and model-directory count need not match.
- **Eight submission ZIPs** are both preserved in the compressed archive and exposed separately in `submissions/`. Six have historical public score reports; two are engineering packages. `SOTA_STATE.json` and `packages.sha256` record exact bytes and hashes.
- Readable research artifacts are exposed in `research-history/main/` and `research-history/cpp/`; original binary NPZ/PNG artifacts remain inside the complete archive.

## Active training at copy cutoff

The existing `tfm_f1` Transformer job was still running and was not stopped. Copied active-family files were:

- `WunderFunder/artifacts/logs/tfm_f1.log`
- `WunderFunder/artifacts/logs/tfm_f1.v1.log`
- `WunderFunder/artifacts/runs/tfm_f1/model.pt`
- `WunderFunder/artifacts/runs/tfm_f1/model_ep0.pt`
- `WunderFunder/artifacts/runs/tfm_f1/model_ep1.pt`
- `WunderFunder/artifacts/runs/tfm_f1/model_ep2.pt`

The copy used before/after byte-size and nanosecond-mtime checks; active-family files were copied twice, and newly appearing/changed run files were rescanned. No source was marked unstable during its final copy. The archived fold-1 Transformer has epochs 0, 1 and 2; it does not claim an epoch-3 completion, final prediction cache or later fine-tuning result. Anything the live job produced after this cutoff belongs to a subsequent snapshot.

`copy_audit.json` inside the archive contains the detailed copy records. No checkpoints were loaded or training metrics recomputed by this export task. Public scores are historical journal reports, not independent competition-service verification.
