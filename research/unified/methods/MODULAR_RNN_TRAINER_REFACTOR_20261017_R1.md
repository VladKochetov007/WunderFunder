---
kind: method
status: complete
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - src/wnn/training/README.md
  - scripts/check_training_structure.py
  - .github/workflows/training-quality.yml
---

# Modular recurrent trainer refactor

## Question and authority

The user requests semantic decomposition of scripts/train_rnn.py, reusable code, preserved efficiency, uv/Ruff and propagation to main and research-team. They separately authorize one native **gpt-6.1-sol / medium** critic for this refactor. This code-maintenance scope does not restart paused fits or authorize rentals. The original file had3738lines; its SHA256 was `c4bd341ebc0e1dcd5b6eec717332fa609c7947439b7bd38d77588a97106b1de8` (main baseline a8774e695d26dd2eee69b8d06ffdaec146d1babd).

## Mechanism and method

The [public entrypoint](../../../scripts/train_rnn.py) now has53lines. The [training package](../../../src/wnn/training/README.md) separates CLI/validation, datasets/caches/membership, causal preprocessing, model construction/initialization, losses, update execution, state, resume/checkpoints and reporting into56focused modules. Largest Python module at this snapshot iscompatibility.py with283lines. Each module is indexed in the package README.

Explicit attrs records own per-run input/readout settings, paths, dependencies, datasets/network and optimizer/stream clocks. Phases bind local tensor references; no generic locals/exec state routing, tensor copying or extra GPU synchronization is introduced. Reusable helper APIs exist independently of the orchestrator. The legacy facade preserves imported settings/exports and custom-name importlib use without monkey-patching modules. Normal training uses explicit objects.

The numerical expressions, dtype settings, masks, sampler order, recurrent resets/detaches, frozen target clone and optimizer/QAT/scaler/scheduler/SWA/EMA order were retained by extraction. All172CLI argument declarations match the original arguments/defaults/choices/help in the critic's static comparison. The historical FP16 default is preserved; FP32 research commands still opt in explicitly.

Parser consumers were adapted: [E540 stdlib AST preflight](../../../scripts/e540_cli_preflight.py) reads actual modular registrations and feature choices, [ancestry audit](../../../scripts/audit_z3_ancestry.py) calls the new parser while retaining a historical-source fallback, and [explicit E540 source manifests](../../../scripts/build_e540_packet.py) include the implementation package. Existing rglob-based source packet builders already collect its Python files. Frozen packets and running source snapshots were not rewritten.

Source provenance now hashes the thin CLI and implementation package. Strict resume still rejects source changes and restores RNG last. Source-only portable packets may lack pyproject/uv.lock; their absence is explicit, present files are hashed, and actual attrs version is recorded. attrs26.1.0 became a direct research dependency using its existing locked version; uv resolved the same107packages and changed only root dependency records. No Torch/numpy version changed. The offline lock attempt lacked cached numpy registry metadata; normal uv lock then completed successfully.

## Independent review and corrections

Read-only native role `training_refactor_critic`, actual model gpt-6.1-sol, reasoning medium, reviewed architecture, consumers and the implementation in bounded passes. It found missing facade dependencies/exports, a missing controlled-loss handoff, bypassed model/prediction hooks and an unconditional dependency-file requirement that would break source-only Kaggle packets. These were corrected. The final file review confirms the source-packet blocker resolved and reports no outstanding concrete findings within its static scope. This is independent code criticism, not empirical approval.

## Evidence and interpretation

`uv run --no-sync ruff format` and `uv run --no-sync ruff check` passed for the new package, entrypoint and changed CLI/quality helpers (also the changed ancestry/manifest scripts). `uv run --no-sync python scripts/check_training_structure.py` passed: module cap300, function cap200, annotations/public docstrings and no global/nonlocal. A normal `scripts/train_rnn.py --help` invocation exited0 without data/GPU work. Local commands reused the existing environment without syncing it to a worktree, under20%CPU, one thread and bounded memory on physical scratch.

The [lint-only CI](../../../.github/workflows/training-quality.yml) applies the same structural policy to main/research-team and their PRs using uv/Ruff without installing Torch. Actions were pinned to inspected upstream release commits: [checkout v7.0.1](https://github.com/actions/checkout/tree/3d3c42e5aac5ba805825da76410c181273ba90b1) and [setup-uv v10.3.0](https://github.com/astral-sh/setup-uv/tree/1c37ad07a6a961277cf70c0d37d6f313000f5884). This is an engineering guard, not a research queue.

## Limits and validity

No tests were added or run; no training, numeric parity comparison, hardware profile or throughput benchmark ran. Static preservation of operations is not a measured runtime-equivalence certificate. Python phase calls/state handoffs add orchestration overhead whose magnitude has not been measured; there are no added tensor transfers/reductions or per-step source scans. Source hashing runs once during setup. Profile warning capture now restores its warning context instead of replacing a process-global warning hook; standard logging retains legacy record content and flush behavior.

No model-quality, leaderboard or scientific mechanism result is claimed. Existing caches/weights/recipes keep their original exposure and ancestry. An exact continuation from pre-refactor source must not be silently accepted; use the pinned historical source for that continuation.

## Decision and next step

Integrate the shared refactor into main and research-team and synchronize the open draft branches without merging their ideas. Keep the small CLI and structural checks as permanent policy. Future authorized numerical/performance checks should use matched fixed inputs and the original source with unchanged seeds, precision and update budget; do not infer parity or throughput from lint alone.

## Source records

- [ADR0041](../../../specs/adr/0041-modular-training-code.md), [workflow](EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md), [published experiment catalogue](../../experiment_drafts/README.md).
- [Complete component map and usage](../../../src/wnn/training/README.md), [Ruff/uv configuration](../../../pyproject.toml), [locked environment](../../../uv.lock).

## Publication receipt

Shared implementation commit `eb3309a322c1e24b3cc157eaefe3ca0e0290cd42` was pushed to main, research-team and all14 published experiment branches using ordinary merges and atomic pushes. Catalogue README conflicts were resolved by retaining both the shared integration section and each draft's own idea entry. GitHub verified all14 PRs remain drafts targeting research-team, with exact catalogue head matches and MERGEABLE status. The [catalogue](../../experiment_drafts/CATALOG_20261017_R1.json) retains original, pre-refactor and integrated heads. The first [research-team CI run](https://github.com/VladKochetov007/WunderFunder/actions/runs/39036319996) passed; it runs lint/format/structure checks only.
