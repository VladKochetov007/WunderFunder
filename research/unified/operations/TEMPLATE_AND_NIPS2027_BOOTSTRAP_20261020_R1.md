# Universal template refresh and NIPS2027 initialization

## Scope and authority

The user assigned template modernization as a side task and explicitly allowed
a `gpt-6.1-sol` agent at medium reasoning. The subsequent instruction requested
a new `/home/vlad/development/NIPS2027` project from the completed template and
moving the supplied LLM-diversity context archive there. Main competition
research continues independently; no NIPS2027 experiments were authorized or run.

## Delivered artifacts

- `/home/vlad/development/CompetitionTemplate` is clean at
  `9220a1289d057d9c9a0a3eae7e27e77c333a5c8c`. Its 30 top-level skill directories
  expose 38 skill entrypoints through 76 relative Claude/Grok skill links.
  `CLAUDE.md` and `GROK.md` link to `AGENTS.md`.
- Existing portable research instructions were preserved and audited; additions
  include generalized B300 and experiment-draft workflows, arXiv, Hugging Face,
  NVIDIA Kaggle, marimo-pair and Ponytail. Provider budgets, metrics, precision
  and competition rules remain configurable rather than inheriting Wunder policy.
- Ponytail5.3.2 and marimo-pair0.0.20 were imported at the revisions in the
  [Wunder import manifest](../../../docs/skill-imports-20261020.json), with
  licenses. The template's `docs/skill-audit.json` records which existing skills
  were audited versus refreshed from upstream; this is not a claim that every
  third-party skill is the latest upstream release.
- Useful MIT-licensed NVIDIA Kaggle tools were copied from
  `/home/vlad/development/ChemComp/projects/nvidia-kaggle` into the template's
  `.agents/skills/nvidia-kaggle-skill`. Source revision
  `2b78cf29f5f30680764292a6592de8d53d4147a8` matched upstream at inspection.
  Original source, credentials and local runtime state were preserved.
- Wunder skill changes from actor commit
  `4d166b4f78fe6e2e19fe79b446c02afe3810854a` were integrated as `0505bc777`.

## New research workspace and exact document move

The committed template copier created `/home/vlad/development/NIPS2027`.
Its initial local commit is `7bdbaa9be89b4808eeb8a3729a12115c670bf39f`; no
remote was created. The exact 148,310-byte file was moved from
`/home/vlad/Downloads/llm_diversity_autoresearch_complete_context_2026-10-15.md`
to `research/unified/sources/reviews/` inside that project.

SHA-256 before and after the move:
`f1bf0d4058e1ef39cb06269c83a3dff537c2f87faa20740348b91bd33ce744e9`.
The original Downloads file was removed only after matching the destination.
Root independently checked the destination hash and source absence.

The source README describes the historical conversation, proposed frozen-model
collaboration study, unverified external claims and outstanding work. Its
`docs/project-initialization.json` records template revision and move provenance.
The requested workspace name does not settle a conference or deadline.

## Checks and limits

The side agent reported passing static skill/frontmatter, local-link, portable
inventory, Python syntax and changed-code Ruff checks. Actual copier execution
and new-project checks passed: 78 relative links including instruction links,
archive integrity, clean Git, an empty mode0600 environment file and disabled
execution. Root checked the project Git status, instruction links and archive
hash. No environments, hooks, telemetry, provider operations or scientific fits
were started by this side task. These are tooling checks, not research results.
