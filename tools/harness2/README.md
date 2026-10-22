# Harness² in the research workflow

This adapter uses Google's actual [Harness²](https://github.com/google-research/harness2)
recursion, selection, reflection and persistent library APIs. The reviewed upstream
revision is `1cfe254e8d25b15f37048c3730134686abe60335`. All 39 upstream Python files
are checked against the committed manifest, with CRLF normalized to LF.

The integration optimizes **research handoff instructions and skills**. It does not
optimize model weights, dispatch training, authorize SSH, change experiments or select
models by WP. Scientific recipes, membership, metric and launch authorization remain
under the existing research controller and immutable grader. A workflow composition
selected by Harness² is not an independently verified scientific improvement.

## Where to use it

Before an experiment handoff, prepare a small task bundle containing the question,
implementation notes and permitted factual process evidence. Probe alternative
instructions on that same task, compose a final instruction set and inspect the
resulting handoff. After selection, retain process lessons with their full harness
and evidence. Read the matching library before the next compatible task.

Start with `k=1`. Both modes cost four executor rollouts per unit of `k`: base,
general-purpose proposal, domain proposal and composition. Selection and reflection
add editor calls. The wrapper allows at most `6*k+2` editor attempts and `4*k`
executor attempts, serializes workers, limits `k` to 1 or 2, disables hidden transport
retries, and shares one wall deadline of at most 14,400 seconds across the loop.
These are workflow-probe budgets, not extra scientific replications.

## Installation and offline verification

Use a separate Python 3.12/3.13 research environment. Submission dependencies and
the existing project lockfile are unchanged. Upstream recommends Linux; the direct
API avoids its POSIX-only benchmark scheduler. Our deterministic path is also tested
on native Windows; model-backed Windows execution is not established.

```bash
python3.12 -m venv ../harness2-env
../harness2-env/bin/python -m pip install -r tools/harness2/requirements.txt pytest

# No model calls, workers, GPU work or Harness² installation needed for planning:
python scripts/harness2_workflow.py plan --wall-seconds 600

# Actual upstream Propose–Probe–Compose, selection, reflection and library storage:
../harness2-env/bin/python scripts/harness2_workflow.py smoke \
  --output ../harness2-runs/lifecycle-parallel --wall-seconds 120
../harness2-env/bin/python scripts/harness2_workflow.py smoke \
  --mode sequential --output ../harness2-runs/lifecycle-sequential --wall-seconds 120

PYTHONPATH=src ../harness2-env/bin/python -m pytest -q \
  tests/test_harness2_task.py tests/test_harness2_workflow.py
```

The deterministic fixture reproduces a reporting error: calling checkpointed,
exported and evaluated items all "completed". The fixture editor adds a lifecycle
reporting instruction, and real upstream probes execute the resulting policy.
This demonstrates integration correctness, not LLM reasoning or a model-quality gain.
It uses no model credentials, competition data, Torch or accelerators.

## Real research handoffs

Copy the schema in `examples/lifecycle-task.json`. Declare a safe task ID/domain,
the work request, a mapping from destination names to repository-relative source
files, and explicit protected files such as the frozen scientific metric and recipe.
Only the listed small text files enter the task context. Do not supply ground truth,
grading outputs, private paths or credentials. Inputs and protected bytes are checked
before and after execution. A source change needs a new attempt, not an edited receipt.

The editable surface initially includes `AGENTS.md` and skill `SKILL.md` files only.
Executable scripts, hooks, plugins and delegated agents are deliberately outside this
adapter's current surface. Root project AGENTS and scientific code are not rewritten;
each rollout gets a separate workspace.

For model-backed work, supply both the upstream editor channel/model and an explicit
worker JSON file. Harness² supports `opencode` and `claude_cli` editors; Codex is a
possible executor, not an upstream editor channel. There is no default model call.

Example worker file (replace the model with an actually available authorized model):

```json
{
  "argv": ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
           "--model", "YOUR_AUTHORIZED_MODEL", "-C", "{workspace}", "{prompt}"],
  "timeout_seconds": 120
}
```

The worker receives `task.json`, `source/` and the proposed harness in its workspace.
It must write `deliverable.md` plus a factual `trajectory.md`. `{workspace}` and
`{prompt}` expand within individual argv elements, without a shell. Use existing
private authentication; never embed tokens in task or worker files.

```bash
../harness2-env/bin/python scripts/harness2_workflow.py refine \
  --task path/to/task.json --worker path/to/worker.json \
  --channel opencode --model YOUR_PROVIDER/YOUR_AUTHORIZED_EDITOR_MODEL \
  --output ../harness2-runs/actual-handoff-001 --wall-seconds 600
```

Real editor/worker runs can incur provider costs; no model-backed invocation was used
to validate this integration. Call/time ceilings are not a dollar cap. The worker
retains responsibility for its explicit allowed actions. GPU visibility is removed
from the worker environment, but staging and hash checks are not an OS sandbox or
network isolation. Run untrusted workers in an appropriately isolated environment.
The wrapper never issues the project's root-owned scientific admission receipts.

## Durable evidence and lessons

Every attempt writes `IDENTITY.json`, upstream contexts/rollouts and `RECURSION.json`.
Success writes `adopted/` and `COMPLETE.json`; failure preserves `FAILURE.json` and
worker logs. Repeating an identical completed command verifies the identity/adopted
hashes and returns the old result without rerunning workers. Interrupted attempts
require a new output directory; there is no silent refit or retry.

The library stores the selected full tree and reflection under a namespace bound to
upstream source, input/protected hashes, base harness, prompt/domain, models, worker,
mode and budgets. Only compatible earlier lessons are read. A changed recipe or
model creates another namespace; transfer across namespaces requires separate review.
No lesson is written for failed selection or missing reflection. Lessons concern
observed process behavior, not scientific grades. Upstream selection chooses among
compositions, not the base; preserve base results for external comparison rather than
calling selection proof of improvement.

Keep scientific WP/bootstrap results in the existing experiment packet and GitHub PR
after immutable official grading. Do not convert continuous WP into Harness² rubric
pass counts or put grader verdicts into proposal/composition contexts. Do not publish
private model audit transcripts, credentials, datasets or model weights.
