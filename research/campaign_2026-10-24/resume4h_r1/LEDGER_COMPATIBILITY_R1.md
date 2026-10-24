# Historical experiment identifier compatibility

Two historical append-only experiment rows use `experiment_id` instead of `id`; the final catalog initially rejected them. The loader now accepts this alias only in the experiments ledger, only when `id` is absent, and only for a nonempty string. Raw rows and hashes remain unchanged. Existing explicit IDs and hypothesis validation are unchanged.

Ruff formatting/checks and all seven research-tree tests pass, including raw CRLF/hash preservation and rejection of the alias for hypotheses. Independent Luna review passes. This is navigation compatibility, with no training or scoring change. The failed [R1 receipt](../../bandit_tree/generated/RESUME4H_BUILD_CHECK_R1.json) is preserved; [R2](../../bandit_tree/generated/RESUME4H_BUILD_CHECK_R2.json) records the repaired full-checkout result.
