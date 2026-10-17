# Repository map

Use root guides for orientation, then follow the READMEs next to the code or
evidence. Many experiment folders are immutable dated records; names such as
current, final or best in an old path do not make that item current.

| Path | Purpose |
|---|---|
| [README.md](../README.md) | Project entry point and links to the maintained guides. |
| [AGENTS.md](../AGENTS.md) | Current agent and repository instructions. |
| [docs/](README.md) | Competition, development and architecture guides. |
| [specs/](../specs/README.md) | Objective, status, execution contracts and decision records. |
| [RESEARCH.md](../RESEARCH.md) | Research evidence navigation. |
| [research/unified/](../research/unified/README.md) | Maintained subject index and synthesis. |
| [Repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md) | 17 October repository and evidence audit. |
| [Current model catalog](../research/unified/models/CURRENT_MODEL_CATALOG.md) | Indexed model sources, ancestry and package evidence. |
| [Validation protocols](../research/unified/validation/VALIDATION_PROTOCOLS.md) | Evaluation populations, panels and evidence interpretation. |
| [Storage map](../research/unified/operations/STORAGE_MAP.md) | Storage locations and verified archive receipts. |
| Campaign folders under research/ | Dated experiment recipes, logs, results and campaign receipts. |
| research/hypotheses.jsonl, research/experiments.jsonl | Append-only scientific records. |
| research/bandit_tree/ | Hypothesis, experiment and combination relationships. |
| [src/wnn/](../src/wnn/) | Reusable data, metrics, features, inference and research components. |
| [src/wnn/training/](../src/wnn/training/README.md) | Canonical modular training implementation and component map. |
| [scripts/](../scripts/) | Command-line tools for training, evaluation, packaging and campaign-specific work. |
| [native/](../native/) | Rust implementation, kernels and Python bindings for native inference. |
| [baseline/](../baseline/README.md) | Competition starter model and reference callback. |
| [submission/](../submission/) | Callback and build-source variants used in package construction. |
| [submissions/](../submissions/) | Dated release packages, manifests, gates and result receipts. |
| [artifacts/](../artifacts/) | Selected run histories and analysis outputs. |
| cache/ | Prediction and intermediate caches; availability and retention vary. |
| private_assets/, delivery/, kaggle_kernel/ | Private or provider-specific packets and delivery material. Check provenance and access boundaries before reuse. |
| [tests/](../tests/) | Automated unit, integration and marked data/GPU checks. |
| [Makefile](../Makefile) | Maintained quality, lint, format and structure-check targets. |
| [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock) | Runtime/development dependency groups, tool settings and locked environment. |

Local data, private weights, credentials and generated artifacts may be
available only on an authorized machine or in private storage. A tracked path
does not imply that it is a complete source dataset, a release-ready package or
a current research artifact. Follow the exact source hashes and manifests.
