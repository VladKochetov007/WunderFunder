# Research artifact templates

Use these schemas for long-running or resumable campaigns. Keep every JSONL append-only. Store large outputs separately and reference them by content hash and path.

Use only fields relevant to the task; these templates do not require a new workflow engine or every file below. A compact manifest and append-only result ledger are enough. Keep exact run configurations and commands, and write outputs under unique run IDs rather than overwriting evidence. Append status updates with the ID of the record they supersede.

## Contents

1. Workspace layout
2. Research contract
3. Hypothesis card
4. Experiment record
5. Evidence and claim records
6. Active frontier
7. Grader output
8. Campaign dashboard
9. Empirical selection supplement

## Workspace layout

```text
research/
  contract.yaml
  state.json
  frontier.md
  hypotheses.jsonl
  experiments.jsonl
  evidence.jsonl
  claims.jsonl
  decisions.jsonl
  artifacts/
  graders/
  reports/
```

Keep hidden graders outside the actor-visible workspace. Put candidate implementations in isolated worktrees or containers.

## Research contract

```yaml
question: ""
target_claim: ""
objective_mode: "competition|engineering|discovery"
utility: "primary outcome, constraints, and cost tradeoff"
deliverable:
  type: ""
  schema: ""
baseline:
  artifact: ""
  metrics: {}
verification:
  tier: "A|B|C|D"
  public_metrics: []
  private_metrics: []
  invariants: []
  falsifiers: []
scope:
  allowed_data: []
  allowed_tools: []
  mutable_paths: []
  forbidden_actions: []
budgets:
  wall_time_s: 0
  tokens: 0
  cpu_hours: 0
  gpu_hours: 0
  money: 0
selection_budget: # Empirical search only; set actual limits before the search.
  candidate_trials: 0
  proxy_variants: 0
  evaluation_queries: 0
stop:
  success: []
  plateau: ""
  abort: []
human_gates: []
assumptions: []
```

## Hypothesis card

```json
{
  "id": "H-0001",
  "parent_ids": [],
  "mechanism_family": "representation/factorization",
  "statement": "",
  "proposed_change": "",
  "representation_change": null,
  "expected_mechanism": "",
  "claim_scope": "",
  "assumptions": [],
  "predicted_observable": "",
  "predicted_direction_or_delta": "",
  "cheap_test": "",
  "falsifier": "",
  "known_failure_modes": [],
  "estimated_cost": {},
  "proposed_by": "",
  "committed_at": "",
  "status": "not_tested|probing|screen_failed|insufficient_evidence|supported|replicated|contradicted"
}
```

Require `proposed_change`, `expected_mechanism`, `claim_scope`, `cheap_test`, and `falsifier`. The mechanism is a proposed explanation until tested. Leave `representation_change` null for a useful incremental change; novelty is not an eligibility gate outside a discovery objective.

## Experiment record

```json
{
  "id": "E-0001",
  "hypothesis_ids": ["H-0001"],
  "branch": "",
  "actor": "",
  "code_commit": "",
  "config_path": "",
  "config_hash": "",
  "reference_hashes": [],
  "environment_hash": "",
  "data_hashes": [],
  "grader_hash": "",
  "command": "",
  "seeds": [],
  "started_at": "",
  "finished_at": "",
  "wall_time_s": 0,
  "token_cost": 0,
  "cpu_hours": 0,
  "gpu_hours": 0,
  "money": 0,
  "public_metrics": {},
  "private_metrics": {},
  "invariants_passed": false,
  "artifact_hashes": [],
  "notes": ""
}
```

Record the predicted outcome before filling measured metrics.

The config and reference hashes must identify the exact compared procedures or artifacts, not just their family names. Include grid contents and generation commands when a scan informs selection. Registration and results are separate append-only records linked by run ID; do not fill outcomes into the committed registration. For empirical runs, attach the supplement below.

## Evidence and claim records

```json
{
  "evidence_id": "EV-0001",
  "kind": "primary-source|experiment|formal-check|physical-measurement|review",
  "locator": "URL or artifact path",
  "content_hash": "",
  "supports": [],
  "contradicts": [],
  "limitations": [],
  "verified_by": [],
  "created_at": ""
}
```

```json
{
  "claim_id": "C-0001",
  "text": "",
  "status": "not_tested|screen_failed|insufficient_evidence|supported|replicated|contradicted",
  "claim_scope": "",
  "supporting_evidence": [],
  "contradicting_evidence": [],
  "assumptions": [],
  "external_validity": "",
  "reproduction_ids": [],
  "updated_at": ""
}
```

Never promote a claim to `supported` solely from an LLM review.

- `not_tested`: no discriminating experiment yet.
- `screen_failed`: the tested recipe missed a spending or promotion gate; the broader mechanism is not thereby refuted.
- `insufficient_evidence`: the available design, precision, exposure, or reproduction cannot resolve the stated claim.
- `supported`: independently verified evidence supports the explicit scope, such as a checked proof or an observed fixed-artifact score. An adaptively selected score alone supports no generalization claim.
- `replicated`: the unchanged empirical claim survives claim-appropriate independent evidence; identify the seed repetitions, implementation checks and/or unexposed or prospective evaluations actually performed.
- `contradicted`: a valid discriminating test contradicts the stated claim within its assumptions and scope. Failure to detect a small effect is not proof of its absence.

## Active frontier

Keep `frontier.md` short enough to fit in one agent message:

```markdown
# Active frontier

## Target and baseline

## Supported claims

## Live mechanism families

| Hypothesis | Mechanism | Evidence | Risk | Next test | Cost |
| --- | --- | --- | --- | --- | --- |

## Contradictions and evaluator risks

## Recently closed dead ends

## Budget remaining

## Evaluation exposure and confirmation status

## Next three decisions
```

Regenerate this view from the ledger. Do not overwrite the raw records it summarizes.

## Grader output

```json
{
  "candidate_id": "",
  "valid": false,
  "violations": [],
  "reward": {
    "task_quality": 0,
    "robustness": 0,
    "novelty": 0,
    "reproduction": 0,
    "cost_efficiency": 0
  },
  "public_private_gap": null,
  "suspicious_improvement": false,
  "required_followups": [],
  "grader_hash": "",
  "artifacts": []
}
```

Use validity as a hard gate. Keep the reward vector visible; derive any scalar search score in the controller so it remains auditable.

## Campaign dashboard

Track at least:

- best valid private score and delta from baseline;
- independently reproduced score;
- improvement per dollar, token, and GPU-hour;
- public-private gap;
- reward-hacking or leakage incidents;
- fraction of proposed mechanisms tested;
- mechanism diversity among surviving branches;
- duplicate and rediscovered dead ends;
- human minutes per accepted claim;
- critic precision, recall, and calibration when critic predictions are logged.

## Empirical selection supplement

Attach this only to empirical search records. Read [empirical-validation.md](empirical-validation.md) for how to use it. Fill counts and planned repetitions before a run; append realized counts/results afterwards. Reference shared exposure/config records by ID to avoid duplicating them in every row.

```json
{
  "run_id": "E-0001",
  "phase": "discovery|confirmation",
  "preregistration": {
    "path": "",
    "sha256": "",
    "committed_at": "",
    "prior_results_seen": []
  },
  "evaluation": {
    "unit": "",
    "split_hash": "",
    "source_group_hash": "",
    "exposure_record_ids": [],
    "exposure": "unexposed|exposed|unknown",
    "prior_uses": [],
    "primary_metric": "",
    "comparison": "",
    "minimum_useful_effect": null,
    "stopping_rule": ""
  },
  "selection": {
    "campaign_id": "",
    "attempt_index": 1,
    "trial_record_ids": [],
    "candidate_trials_seen": 0,
    "proxy_variants_seen": 0,
    "evaluation_queries_seen": 0,
    "posthoc_choices": []
  },
  "seed_pairs": [
    {"candidate_seed": 3, "control_seed": 3, "candidate_config_hash": "", "control_config_hash": ""}
  ],
  "uncertainty": {
    "evaluation_sampling": "",
    "training_variability": "",
    "selection_and_transfer_limits": ""
  },
  "prospective_confirmation": {
    "available": false,
    "source": null,
    "frozen_artifact_hash": null,
    "registered_comparison_hash": null,
    "result_record_id": null
  }
}
```

Each `seed_pairs` entry identifies candidate and control seeds and their config hashes; use an empty array with a reason in `training_variability` when no stochastic fitting occurs. The preregistration must fix the planned pairs before outcomes. Record every completed pair, not only favorable seeds. Exposure records retain source/group lineage across repartitioning. A confirmation result cites its frozen artifact and comparison, and reports what independent evidence was actually obtained. If none is available, keep `available: false`; another resplit or agent review is not a replacement.
