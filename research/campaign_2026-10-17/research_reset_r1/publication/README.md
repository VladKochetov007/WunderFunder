# Git publication and draft-PR status

The reviewed repair and research commits were published through Git SSH to
`research-team` and `codex/research-reset-no-fit-20261017`. The frozen integration
base for the25proposal branches is `ed40bea53b66f65a437f9ea3658a982ebdbe5251`.
No model candidate, weight, cache or competition submission was promoted.

- [BRANCHES.json](BRANCHES.json) records all25branch names, exact commits, common
  base, packet-file hashes and verified remote URLs. Each branch starts from the
  same research-team base and installs one packet under
  `research/experiment_drafts/rr17-XX/`, with its parent/unified index entries.
- [PUSH_BRANCHES.log](PUSH_BRANCHES.log) preserves the successful atomic SSH push;
  [REMOTE_REFS.txt](REMOTE_REFS.txt) preserves the post-push SHA verification.
- [prepare_branches.py](prepare_branches.py) is the exact installation operation,
  using an isolated sparse worktree and existing reviewed staging packets. It
  creates proposal commits only, refuses an existing target worktree, and runs
  no model, fit or hardware profile. Do not rerun against existing branches.
- [INTEGRATION_PR.md](INTEGRATION_PR.md) is the ready English body for promotion
  to main once an authenticated GitHub API session is available. The individual
  packet PR.md files target research-team.

`gh auth status` reports no logged-in GitHub host. Git SSH read/write succeeds;
repository availability is not the blocker. **No new PR was created**, and no
PR URL or merge is fabricated. The existing14PRs remain historical references.
`main` still points to the previously published maintenance snapshot; the new
reviewed work is on research-team and its named proposal branches. The prepared
integration PR would promote the shared fixes/evidence, not claim a model win.

For each individual proposal, create a draft with its committed PR.md, base
research-team and exact branch from the receipt when API authentication is
available. Future execution still requires explicit fit authority and all recipe
bindings; branch publication supplies neither.

[Staging packet index](../drafts/README.md) · [Campaign](../README.md)
