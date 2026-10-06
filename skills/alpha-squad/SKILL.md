---
name: alpha-squad
description: "Generate bounded, testable market hypotheses from relevant accounting, flows, networks, microstructure, and identification lenses. Adapt the roles to prediction, trading, or causal-effect questions."
---

# Alpha Squad

Use this skill when the user wants market hypotheses generated from complementary lenses.

## Role Map

- `fundamentalist` -> `agents/fundamentalist.md`
- `vulture` -> `agents/vulture.md`
- `network-architect` -> `agents/network-architect.md`
- `book-physicist` -> `agents/book-physicist.md`
- `causal-detective` -> `agents/causal-detective.md`

## Workflow

1. Read `README.md` and the active research contract; identify the objective and available data.
2. If the user specified one applicable role, use that role with a bounded task.
3. Otherwise select one or two relevant lenses, within the contract's worker and hypothesis budgets. Do not spawn every role by default.
4. Have each selected proposer record one hypothesis, comparison, falsifier and resource estimate before seeing other proposals. Share results only after this independent pass.
5. Require an observable prediction and an evidence-linked rationale. Counterparty, constraint, DAG and decay fields are conditional on the claim, not mandatory for prediction tasks.
6. Use a separately scoped critic to check the registered claim. Stop after the contracted review cycle; further probes or revisions consume the same trial budget and require a new record.
