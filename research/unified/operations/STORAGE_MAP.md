# Source, data and artifact storage

Snapshot: October 17, 2026. A path identifies a storage role, not automatically the latest model or a unique experiment. Never infer completeness from a folder named `current` or a successful provider status.

| Location | Role | How to identify the artifact |
|---|---|---|
| GitHub `VladKochetov007/WunderFunder` | Versioned active code, research records and historical branches | Commit SHA and path; Main audit baseline `0edd432271991250f4718f9b68f33340eff46214` |
| `beleriand:~/WunderFunder` | Supplied Parquet, shared prediction caches, older Git checkout | File hash, row-group identity, frozen provenance |
| `beleriand:~/WunderFunder-creative` | Creative/waves/distillation research; data/cache links to baseline tree | Exact run/checkpoint and source hash |
| `beleriand:~/autoresearch-worktrees` | Saved top15 Git and research history | Preserve refs not reachable from current GitHub |
| `beleriand:~/WunderFunder-main-20261012` | Main snapshot without `.git` | Snapshot manifest; not current GitHub Main |
| `beleriand:~/rs1013` | Late creative experiments and production journal | Completed log, endpoint and receipt; latest material audited through October 15 |
| `beleriand:~/pkgwork` | Native builds, members, package manifests and ZIPs | Full ZIP SHA256 and its member list |
| `beleriand:~/dsjobs`, `~/dsdata` | Cloud job inputs/outputs and collected logs | Provider state plus independent scientific-process/terminal evidence |
| `beleriand:~/hf_stage`, `~/team_in`, `~/xteam` | Exchange copies and cross-team integration | Hash/inode comparison; duplicates are not additional models |
| Local research packages | Point-in-time copies, audit reports and ZIPs | Package manifest and SHA256; local root is not a Git checkout |
| Private Hugging Face model archive | Preserved own weights and metadata | [Archive lookup and verified transfer receipts](../models/huggingface/README.md) |
| Git `archive/2026-10-01` | Historical source/checkpoint archive, including late Transformer supplement | Base + late manifests together: 268 run directories / 865 checkpoint paths |

## Git histories complement one another

Local SSH on the Mac successfully read GitHub as `maloimushiy`. The same account's connector had a narrower installation scope, while `beleriand` had no usable GitHub SSH identity. Those route-specific failures did not mean the repository was inaccessible.

The audited live graph contains 2,281 reachable commits: 2,267 from branch heads plus 14 synthetic PR merge refs. There are 28 branch heads. The old server graph has 386 commits, only 212 of which are reachable from current live refs. Preserve the other 174 top15 commits: the combined histories contain 2,455 unique commits. These counts describe history, not training runs. PR Git refs expose code, not live issue discussions, approvals or UI states.

## Preserve provenance without duplicating payloads

Source code belongs in active packages and entrypoints. A frozen campaign packet retains its own source pins and should not be rewritten when the active trainer changes. Model weights, full prediction arrays, data and credentials remain outside normal Git changes; commit compact manifests and evidence descriptions.

The audit inventoried 84,535 paths in 19 main server roots and 3,587 paths in 11 additional roots, with overlapping copies. All 29 `*-originals` files matched current research counterparts; `hf_stage` included 20 matching artifacts and 11 prediction hardlinks. Do not sum these as independent experiments or unique storage.

Main's October 17 pruning receipt records 463 old prediction-cache files and one incomplete output removed after the research pause. The earlier September cleanup is another historical event. A receipt can validly reference a cache that no longer exists; use hashes and the original measured result, and mark any regeneration as a new action. [Pruning receipt](CACHE_PRUNE_20261017_R1.md).

For small audit receipts see the [evidence index](../reviews/audit-2026-10-17/README.md). The audit read existing artifacts; it did not reconstruct weights, rerun training, regenerate prediction arrays or copy credentials.
