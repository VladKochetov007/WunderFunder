# Research ledger

Every idea gets a hypothesis row before or while it is tested, and every run
that informs a claim gets an experiment row. Refuted and inconclusive results
stay in the ledger; they are not deleted or rewritten.

| file | content |
|---|---|
| `hypotheses.jsonl` | `H-###`: statement, mechanism, falsifier, status (`open`, `probing`, `supported`, `refuted`, `inconclusive`), evidence ids |
| `experiments.jsonl` | `E-###`: command, commit, data split, result, claim, reproducibility |
| `frontier.md` | current best score, supported claims, live mechanisms, debunked list, TODO |
| `findings-*.md`, `profiling-*.md` | dated narrative notes |
| `artifacts/` | small tracked JSON results; large run outputs live in the ignored top-level `artifacts/` |

Rules: validation numbers name the split (`valid600` = 600 sequences from
`permutation(1873, seed=12345)`, `full` = all 1873). Probes fitted on validation
sequences say so. A claim from one run is provisional.
