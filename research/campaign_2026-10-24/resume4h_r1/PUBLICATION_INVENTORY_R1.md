# Publication inventory — October 24 campaign

This is a read-only inventory for root's later staging/publication decision. No
files were staged, committed, pushed, or uploaded. Counts below are the
pre-report snapshot; this report adds one small campaign file. Root may also add
the pending batch-mass result, so refresh counts immediately before staging.

## Branch and pull-request state

- Current branch: `experiment/next-cycle-recovery-20261021`, HEAD
  `c3e208b387ead0ec967f9abc4b14275eb8ba94c5`, tracking the same remote head.
- `origin/main` is an ancestor (the branch is 171 commits ahead). Relative to
  `origin/research-team`, the branch is diverged: 212 commits exist only on
  `research-team` and 214 only on this branch. `research-team` is not an
  ancestor, so this checkout is not a suitable publication base.
- Read-only `gh pr list --head experiment/next-cycle-recovery-20261021
  --state all` returned no PR. There is no draft PR for this head to update.
- The index had no staged changes. The worktree had 19 modified tracked files
  (251 insertions, 3 deletions) and unrelated untracked work as listed below.

The experiment-draft workflow asks for an `experiment/<id>` branch based on
`research-team`, one scoped draft per experiment, preservation of unrelated
dirty work, and an indexed README/ledger handoff. This folder is a multi-result
campaign archive rather than one experiment draft. If root publishes the
campaign together, use a new clean worktree from the current `research-team`
head and identify it as a campaign-evidence preservation PR. Do not publish
from the current branch or import its complete diff.

## Proposed campaign file set

At the inventory snapshot, the new campaign directory contained 479
Git-addable files totaling 29,394,345 bytes. The following seven full GitHub
file-list response snapshots are not referenced elsewhere in the campaign; the
smaller PR detail and metadata files remain. Excluding these leaves 472
Git-addable files totaling 7,914,563 bytes:

```text
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR25_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR26_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR27_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR32_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR33_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR37_FILES_GH.json
research/campaign_2026-10-24/resume4h_r1/pr_audit_25_40/PR38_FILES_GH.json
```

The proposed artifact set is therefore **all nonignored files beneath
`research/campaign_2026-10-24/` except those seven paths**. This retains the
campaign and child READMEs, frozen recipes, source manifests, synthetic and
preflight receipts, result reports, plots, and review records. The inventory
scan found no in-campaign references to the excluded filenames. Before
publication, refresh the counts and include root's pending batch-mass result
and any newly added, reviewed campaign outputs.

The campaign root README links the resume README, which links its result and
subfolder READMEs. Add a short entry for this inventory to
`research/campaign_2026-10-24/resume4h_r1/README.md`; also ensure the campaign
remains linked from `RESEARCH.md` and the unified research index in the clean
publication branch. These index files are currently dirty in the shared
checkout, so port only the reviewed campaign hunks there rather than staging
their current whole-file versions.

## Keep out of this campaign staging set

Do not stage the following 19 currently modified tracked files from this
checkout as-is. Some contain relevant campaign navigation or ledger entries,
but each is mixed with previous work and needs selective review/porting on the
clean publication branch:

```text
AGENTS.md
RESEARCH.md
research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md
research/experiment_drafts/b300-fast-local-kaggle-20261023/README.md
research/experiment_drafts/b300-fast-local-kaggle-20261023/kaggle/README.md
research/experiments.jsonl
research/hypotheses.jsonl
research/unified/README.md
research/unified/ideas/README.md
research/unified/inference/README.md
research/unified/methods/README.md
research/unified/operations/README.md
research/unified/reviews/README.md
research/unified/sources/README.md
research/unified/validation/README.md
specs/current-state.md
src/wnn/training/age_readout_portable_identity.py
src/wnn/training/age_readout_portable_validate.py
src/wnn/training/age_readout_provenance.py
```

The following untracked work also exists outside the campaign and is excluded
from the proposal:

```text
research/experiment_drafts/b300-fast-local-kaggle-20261023/ACTOR_SOURCE_SHA256_R3.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/INDEPENDENT_SOURCE_REVIEW_R3.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_PROFILE_COLLECTION_R2.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_PROFILE_PUSH_R2.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_PROFILE_STATUS_R2.jsonl
research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_PROFILE_TERMINAL_R2.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_RESOURCE_REDUCTION_R1.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/PROFILE_DRIVER_FIXTURE_R2.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/RESOURCE_AND_DELIVERY_AMENDMENT_R2.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/SOURCE_OVERLAY_MANIFEST_R3.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/SOURCE_OVERLAY_PREPARE_R3.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/SOURCE_STATIC_CHECK_R3.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
research/experiment_drafts/b300-fast-local-kaggle-20261023/kaggle/PROFILE_CPU_OPERATOR_R2.md
research/experiment_drafts/b300-fast-local-kaggle-20261023/profile_kaggle_r2.kernel.log
research/experiment_drafts/b300-fast-local-kaggle-20261023/profile_kaggle_r2.stderr.log
research/unified/reviews/astra_ultra_audit_20261023_r1/
research/unified/sources/ASTRA_ULTRA_REVIEW_SOURCES_20261023_R1.md
```

The b300 and October23 Astra files are separate preservation work. The
campaign's own receipts may link to their provenance, but their full payloads
are not campaign dependencies to stage here.

## Large and ignored artifacts

- `.gitignore` ignores both built packages under `submissions/`: the S3 ZIP
  (12,537,972 bytes, SHA-256
  `56ebf18d26187541e539a31078a50dcbeb698e395ca0ecde5e1174aa2e929ded`) and
  S3/S4 ZIP (16,161,940 bytes, SHA-256
  `8bf614168d97d50b58557db3a1f5befa16f4912b5b8b1fbf609b087f62f96be2`). Keep
  these local/private; campaign receipts carry their identities and evaluation
  evidence. Do not force-add or upload them as part of publication.
- `.gitignore` also excludes `cache/valid_targets.npz` (337,186,198 bytes) and
  research `.npz` files, including the 2,359,790-byte derived component-moment
  file `feasibility_scout/direction_probe_r1/DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz`.
  Keep the target cache external. For the derived moments, the campaign result
  records the hash and path; publish the bounded JSON result/plots and treat
  the ignored NPZ as optional external reproducibility material.
- Recovered remote payloads remain under physical `research_tmp`; campaign
  receipts record their hashes and locations. Do not copy payloads into Git.
- No `.gitattributes` or tracked Git LFS files were found. The retained
  campaign files are small text/plot/metadata artifacts; the largest retained
  Git-addable file at scan time was 784,715 bytes. LFS is not needed for this
  proposal. The seven excluded response dumps are text, not LFS candidates.

## Recommended handoff

Root owns the clean-worktree creation, selective index/ledger port, staging,
commit and PR. Before those actions, refresh the campaign file list after the
pending batch result lands, confirm child README links, and re-check that no
new task or unrelated file has entered the staging set. The report itself is
not a publication authorization.
