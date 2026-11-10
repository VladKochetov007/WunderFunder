# Independent static review: B300 readout R2

## Decision

No blocking finding in the frozen readout helper. This is a code/source review, not an execution receipt or model-quality approval. No source, fit, or native artifacts were changed; I did not run tests, SSH, GPU work, prediction export, or scoring. The implementation actor reports 3 focused CPU tests passed and 1 skipped (the Windows directory-symlink case), with Ruff, compilation, CLI help, and schema parsing passing.

## Frozen identities reviewed

- Receiver: `receiving-readout-r2/b300_readout_r2.py`, SHA-256 `ed54c4174acffe2d94f0de2f4c17d504d2bd5099d7407df3bd7b4131b7891b49`.
- Focused tests: `receiving-readout-r2/test_b300_readout_r2.py`, SHA-256 `4043339a469fe329e8deec47cdde091e84dc90ae9aefd93162d2c5089d16168e`.
- Schema: SHA-256 `bc88c92a6c99c7bc8606a8df8617962b81396f56ef06e76c1a38cafd7ce3d4d3`.
- Runbook: `receiving-readout-r2/README_R2.md`, SHA-256 `f085b7051aa7b9201eb1821990ca0a3eba1d9aa93ce4f85f1fca25c442e115fb`.
- Fit/source/helper identities derive from authority R2 and the immutable PR74 source commit `8a8afec964985a6ba8b13d1c9af1ab00e14c6504`.

## Checks reviewed

- Fit collection starts only after the R2 fit summary, owner, and state report complete. Every cell must have a zero-exit, on-time E4 result, exact source/config/admission/split/pair/UUID fields, approved production profile, and an unchanged whole fit-directory artifact inventory. Frozen fit validators recheck checkpoint, terminal status, metadata, data ancestry, BF16/BF16x9 runtime, device UUIDs, and all four full-valid evaluation events.
- Membership uses the source-qualified canonical `CampaignSplit`, including its exact sparse train IDs and full official-valid list. Fit provenance and prediction exports are compared with exact manifest-derived group lists; the readout does not infer membership from `range(10325)`.
- The exact PR74 `sequence_predict._producer()` includes checkpoint/split/fit-member identity plus teacher, normalized config, and provenance, so the readout's producer checks match the pinned source. Prediction export uses `torch.distributed.run` with world size 2 and the cell's admitted UUID pair. The canonical score command runs with GPU visibility disabled.
- The fixed checkpoint hash is checked before export, after export, and after scoring. The export schema, full-valid roster, producer lineage, rank shards, prediction filenames, score receipts, and all three final monitor metrics (`wp`, `t0`, `t1`) are checked; monitor agreement is bounded by `1e-5`.
- The original source-pinned `compare_panel.py` runs only after all six cells pass, against a separate shadow tree containing read-only hardlinks or copies. Its six scores and checkpoint/score/profile hashes must match the readout receipts. It neither selects another checkpoint nor launches a fit.
- The original first-profile six-hour clock and fit cutoff remain fixed; prediction, scoring, and panel stages have 2,400/1,200/600 second caps. Four approved pair streams run sequentially within each physical pair. The helper injects the reviewed R2 descendant runner and resource guard, which accounts for PID/start-tick descendants across torchrun's per-rank sessions. Retries are disabled and output paths are R2-isolated.

## Scope limit

This review clears the frozen readout code for ROOT's separate deployment/admission step. The six fits are still running; final output completeness, score agreement, and panel receipt remain to be established by native execution. All official-valid results remain exposed development evidence, not fresh confirmation or a public-score claim.
