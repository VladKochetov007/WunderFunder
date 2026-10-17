# Development setup

The repository pins Python 3.13 for research development. Submission code must
remain compatible with the scorer's Python 3.11 environment. See
[pyproject.toml](../pyproject.toml), [.python-version](../.python-version)
and the [submission guide](submission_guide.md) for the declared contracts.

## Install the development environment

Install the runtime and development dependencies without the optional research
group:

```
uv sync --locked --no-default-groups --group dev
```

The research dependency group includes CUDA-enabled PyTorch and analysis
packages. Add it only on a compatible machine when the task requires those
dependencies:

```
uv sync --locked --group research --group dev
```

## Routine checks

The repository Makefile defines the checks used by the lint-only CI workflow:

```
make quality
make quality-tests
```

quality runs Ruff lint and format checks plus the training-structure checker.
quality-tests runs focused unit tests for that checker. Run these targets from
the repository root; they use the tool versions and paths declared in Makefile.

For manual checks outside those maintained targets, use:

```
uv run pytest
uv run ruff check
uv run ruff format --check
uv run basedpyright
```

Pytest marks checks that require competition Parquet files, a GPU or a longer
runtime. Select those checks only when their requirements are available. Ruff
uses a 100-character configured line length and targets Python 3.11 syntax.

For changes under src/wnn/training/ or the trainer entrypoint, also run the
structural checker required by the project instructions:

```
uv run python scripts/check_training_structure.py
```

Read the [training component map](../src/wnn/training/README.md) before adding
training behavior. scripts/train_rnn.py remains a compatibility entrypoint;
new behavior belongs in a focused reusable component. The training map lists
the contracts and the specific checks for that code.
