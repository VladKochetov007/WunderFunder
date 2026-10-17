---
name: create-experiment-draft
description: "Create or update a research experiment draft PR for this repository; use proactively when proposing, implementing, handing off or recording a scientific experiment."
---

# create-experiment-draft

Read AGENTS.md and research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md. Use research/unified/templates/EXPERIMENT_DRAFT_PR.md for the English PR body.

1. Inspect main/research-team and dirty state; preserve unrelated changes. Fork experiment/<id> from research-team in an isolated worktree. Never assume a local branch is published.
2. Apply conduct-creative-research for mechanism, independent criticism and evidence ownership; arxiv-search for relevant primary literature. Apply b300-training when preparing accelerator code. Do not launch fits merely to prepare a draft.
3. Put README, PR.md, recipe.json and code/adapters under research/experiment_drafts/<id>/. Describe every useful file and frozen source in README and link the packet from the unified ideas index in the same commit. Append H/E ledger records, do not edit history.
4. Answer all eleven template sections concretely. Bind30 paired seed/fold cells per arm, strictFP32, clean ancestry, official evaluation, useful effect and failure interpretation. No queue manager or quantization. Engineering parity checks are not30-run quality comparisons.
5. Provide exact commands for implemented paths and explicitly name missing adapters/cache bindings/B300 profiles. Never describe a plan printer as a complete trainer. Record every run/metric/archive and its source SHA; use the repository reporting API when available.
6. Prepare locally, then obey current publication authority. The latest October17 instruction authorizes gh, branch pushes and draft PR publication, superseding the initial hold. Under that authority, push research-team first, push experiment branch, use gh pr create --draft --base research-team --body-file <packet>/PR.md. No external messages or uploads otherwise.
7. Update unified notes with actual PR URL, run manifests, results and merge SHA. Merge experiment evidence into research-team, then promote a coherent integration PR to main. Preserve rejected/negative evidence and exact source ancestry.

Repository arXiv invocation: `bash skills/arxiv-search/arxiv_search.sh B300 --max-papers 5` (use the bounded support wrapper). The Bash helper already adds `all:`; use simple search text, not a doubled field prefix.
