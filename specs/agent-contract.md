# Agent assignment and handoff

Each assignment references the objective specification and the ADRs relevant to its task. Give minimal sufficient context. Distinguish known evidence from hypotheses, and identify the exact final artifact the agent owes.

Required assignment fields:

| Field | Content |
|---|---|
| Objective | Research-tree arm and the uncertainty/result to resolve |
| Owner and role | Actor, critic, grader, cloud operator or reproducer; distinguish execution ownership |
| Basis | Source passages, parent recipe/checkpoint lineage and evidence links |
| Deliverable | Concrete file/run/result; explicitly say code-only when that is the intended scope |
| Comparison | Candidate/control, metric/mask, seeds, fixed endpoint and exposure |
| Context boundary | Allowed evidence; competing proposals/private checks withheld when useful |
| Write boundary | Isolated branch/worktree, output directory, immutable files |
| Budget | CPU/RAM/VRAM, time, provider/quota and maximum trials |
| Dependencies | Only actual prerequisites of this arm; identify who resolves each |
| Failure policy | Diagnose, preserve failure, repair with new ID; never silently retry or rewrite results |
| Completion evidence | Exact commands, source/data hashes, artifacts, measured resources, score/report and limitations |

A lead scientific actor owns the route to a trained model and predictions, coordinated with controller admission. A cloud operator owns actual provider execution and terminal artifact collection. A critic gives concrete pass/fail concerns within a bounded review window. A grader evaluates the frozen comparison independently of the actor. Root owns allocation, resource admission and synthesis. Avoid leaving the experiment between a code builder and a monitor with neither responsible for its result.

Handoff must distinguish: proposed, implemented, verified on synthetic cases, profiled on actual data, fitted, predictions exported, evaluated, packaged and publicly scored. Return the last completed stage, pending owner/action, source revision and paths. Test counts and commits must never be reported as model-quality gains.

Use native Luna/max under the current user preference and actual authorized external providers for model diversity. Keep at most three subordinate roles active with isolated write ownership. Do not let agents change their own grader, budget authority or frozen recipe. A source-reading agent must not launch fits unless that execution scope is explicitly delegated.
