# PR30 streaming-student panel

## Current runtime amendment

PR30's current prospective panel is four executable fits: label-only and KD
compact Transformer students at F0/seed1001 and F2/seed1002. It uses the shared
raw-112 sequence runtime from [draft PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32), pinned to implementation source
`d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6`. The amendment is documented in
[RUNTIME_AMENDMENT.md](RUNTIME_AMENDMENT.md) and the machine-readable
[runtime_amendment.json](runtime_amendment.json). The runtime is implemented but
unexecuted; the panel is not launched, and B300 acceptance remains pending.

- [PR.md](PR.md): the amended 11-section handoff, with pinned checkout and commands.
- [PR_CP30_FROZEN.md](PR_CP30_FROZEN.md): byte-preserved original PR30 body.
- [recipe.json](recipe.json): byte-preserved historical CP30 recipe; it is not the
  current launch config and is not trainer CLI input.
- The current exact executable configs are in the shared runtime packet at
  `research/experiment_drafts/b300-transformer-distillation-runtime/configs/`.

The panel costs four new fits. It reuses the exact two teachers/caches and four
GRU label/KD endpoints from PR28 only when source, split, configuration, seed,
cache and endpoint identities match. No output, score, profile, or acceptance
result exists for this amended panel. No CPU-serving, ZIP-readiness, or
parameter/FLOP-matched GRU-versus-Transformer claim is made.

## Historical CP30 (superseded)

The original proposal compared three student backbones across 30 paired cells
per arm (180 planned student fits), with student windows 256/1024, anchored teacher targets up to 4096 rows, and
an alpha-0.9 label/KD mixture. Those specifications remain byte-for-byte in
[recipe.json](recipe.json) and the original handoff is preserved in
[PR_CP30_FROZEN.md](PR_CP30_FROZEN.md). They are historical only. The current
four-fit streaming amendment supersedes them for the executable PR30 panel;
this does not create training authorization.
