# Agent skills used in this project

Copies of the skills the research sessions loaded, kept with the code so the workflow is
reproducible. Project-local skills that were already in the repo stay where the harnesses load
them from:

| Skill | Location | Used for |
|---|---|---|
| conduct-creative-research | `.agents/skills/`, `.claude/skills/` | research controller: hypotheses, falsifiers, ledger, actor-critic agents |
| python-standards and other `.agents/skills/*` | `.agents/skills/` | code style, plotting, paper tooling |
| marimo-pair | `skills/marimo-pair` | driving live marimo notebooks (`notebooks/`) for visual EDA |
| alpha-squad | `skills/alpha-squad` | multi-lens market hypothesis generation (microstructure, causal, flows) |
| mft-research-experts | `skills/mft-research-experts` | data sentinel, skeptic, forensic auditor validation roles |
| arxiv-search | `skills/arxiv-search` | arXiv literature scouting (`arxiv_search` CLI) |
| caveman | `skills/caveman` | terse response mode |

To use one with Claude Code, copy or symlink its folder into `~/.claude/skills/` (or
`.claude/skills/` in this repo); for Codex-style agents use `~/.agents/skills/`.
