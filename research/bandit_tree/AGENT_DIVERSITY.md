# Agent and model diversity

Mixing coding agents and model families is useful for idea variance and for finding
different implementation mistakes. It does not turn agent agreement into empirical
evidence. Every candidate still faces the same frozen evaluator, resource limits and
serving endpoint.

This migration uses the existing Luna/max setting. The user's request to diversify
afterward is recorded; specific future model selections are not silently substituted
for earlier model exclusions. The workflow accepts any explicitly selected agent
or model through per-job provenance rather than a hard-coded provider.

## A bounded rotation

| Role | Context | Output |
|---|---|---|
| Independent ideator | Common task brief plus a different book/article and distinct creativity lens; hide other ideators' answers until commitment | New arm card, source analogy, predicted observable, cheapest test and falsifier |
| Builder | One selected card, relevant source/data contract and public checks | Isolated implementation, exact commands and resource estimate |
| Critic using another model family | Specification first, then committed implementation | Concrete failure modes and executable checks |
| Reproducer | Frozen claim and minimal mechanism description | Independent implementation or exact replay with scope stated |
| Controller | Committed proposals, deterministic results and full lineage | Allocation and belief-update events |

Use at most the campaign's existing concurrency allowance. Run one reviewed heavy
GPU experiment at a time unless aggregate measurements permit more; model diversity
is not permission to launch extra simultaneous compute.

Rotate builder and critic roles between lineages. Different coding tools can be
helpful, but two tools using the same model and shared conversation may be highly
correlated. Record provider/model, coding tool, role, prompt hash, allowed context,
source document, parent arm, branch/output paths and evaluator version for each job.

Keep independent ideation blind until the cards are saved. Afterwards share measured
evidence on the common board, including failures. Implement combinations only as
explicit interaction nodes with their own controls; do not merge incompatible
conclusions or add their reported deltas.

Prefer variation in source material, representations, testing methods and model
families over asking several agents the same broad question. A book is an analogy
source, not evidence that its mechanism works in this dataset. Random selection
should be recorded with the title, sampled sections and seed if used.

## Future job record

```json
{
  "job_id": "permanent unique ID",
  "node_id": "arm:chosen-lineage",
  "role": "ideator | builder | critic | reproducer",
  "coding_tool": "chosen tool",
  "model": "explicitly selected model",
  "model_family": "recorded family",
  "prompt_sha256": "hash of saved task prompt",
  "source_documents": [],
  "context_artifacts": [],
  "independence_boundary": "What other agents' outputs were hidden",
  "output_artifacts": [],
  "experiment_ids": []
}
```

Archive the actual task prompt, source list and output beside the arm. Append a
journal event linking this record; preserve unsuccessful jobs. Promote evidence
from verified experiments, not the confidence or number of agents endorsing an idea.
