---
name: ponytail-help
description: Quick reference for the vendored Ponytail coding and review skills. Use when the user asks how to use Ponytail.
license: MIT
---

# Ponytail skills

This project vendors skills only, from Ponytail 5.3.2. Invoke `ponytail` for a
small complete implementation, `ponytail-review` for a diff review,
`ponytail-audit` for a repository audit, `ponytail-debt` for recorded shortcuts,
and `ponytail-gain` for the upstream benchmark description.

Use the skill name supported by the active harness (for example `$ponytail` in
Codex). Levels `lite`, `full` and `ultra` describe requested behavior; say
“stop ponytail” to end it. Project instructions and the current user request
take precedence over suggested formatting, checks and session persistence.

No plugin hooks, prompt band, task ledger, session import, auto-activation,
telemetry, webhooks or global configuration are installed. Plugin-only commands
from upstream documentation are unavailable in this skills-only import.

Upstream documentation: https://github.com/DietrichGebert/ponytail
