# Source and experiment state

After restoring the archive, `snapshot/source-state/repository.bundle` contains the remote repository's reachable local branch history. Per-copy folders include HEAD, branch, status, log, a tracked source tar, and staged/unstaged work patches. They exclude Git configuration and credentials.

```bash
git clone snapshot/source-state/repository.bundle restored-research
cd restored-research
git switch quant-probes
git apply ../snapshot/source-state/WunderFunder/tracked-worktree.patch
# Restore experiment outputs, including untracked research findings.
rsync -a ../snapshot/WunderFunder/artifacts/ artifacts/
rsync -a ../snapshot/WunderFunder/research/ research/
mkdir -p cache
rsync -a ../snapshot/WunderFunder/cache-metadata/ cache/
git worktree add ../restored-cpp cpp-engine
```

Restore the C++ submissions directory into the C++ worktree separately. Main and C++ archived HEAD hashes in source-state are authoritative if branch tips are subsequently updated elsewhere. Source-state patches preserve the state at copy cutoff; they should be reviewed before applying to any newer checkout. The dataset symlink is recorded as an exclusion and must be re-created for the new machine's actual data path.
