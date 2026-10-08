# H217 packet V5R2 — exact recipe-byte guard

Status: `pending_root_review`. This packet does not authorize a provider create, push, profile, fit, export, or grade.

V5R2 keeps the reviewed V5 recipe bytes, runtime contract, run wrapper, and sealed project. The only execution change is that `worker.py` hashes `recipe.json` and compares it with the pinned SHA256 `c18b96fda798ead5cffbb089bd79589c5ce3aa5149cc81a49d2b4eb8985c9eac` before parsing or starting a phase. A mismatch, a missing file, or malformed bytes stop startup. There is no hash skip.

The semantic validators from V5 remain. They are not a general CLI parser. Duplicate flags and extra sampler flags are rejected because they change the recipe bytes.

## Exact commands

Build, from the repository root:

```sh
python3 research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/build_h217_v5r2_packet.py --output research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/h217_packet_v5r2.tar
```

Focused tests, from the repository root:

```sh
python3 -m pytest -q research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/test_h217_v5r2_recipe_guard.py
```

## Pending root review

`fit_authorized`, `provider_create_authorized`, `push_authorized`, `grade_authorized`, and `profile_authorized` are false. Root reviews the changed-member receipt, confirms the recipe and project bytes match V5, and admits any paid cohort separately.
