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

The repository Makefile defines the static checks and checker unit tests used by CI:

```
make quality
make quality-tests
```

quality runs Ruff lint and format checks plus the training-structure checker.
quality-tests runs six stdlib-only unit tests for that checker. Run these targets from
the repository root; they use the tool versions and paths declared in Makefile.

## CPU compatibility checks

With PyTorch 2.14, attrs, NumPy, PyArrow, SciPy, pytest and the relevant test
dependencies already installed, run the bounded trainer suite:

```
make training-tests
```

It uses .venv/bin/python by default. To use a separately provisioned CPU
environment:

```
make training-tests TEST_PYTHON=/absolute/path/to/cpu-env/bin/python
```

The target sets the compute-library thread budgets to one and imports the
current checkout through PYTHONPATH. It checks legacy API compatibility, feature
assembly, checkpoint/resume behavior and causal features using synthetic data.
Two optional checks require real competition files and skip when those files
are absent. It does not install packages or start a competition experiment.
CI runs only the static checks and stdlib checker tests; it does not provision
PyTorch or claim CUDA parity. The [maintenance receipt](../research/unified/operations/MAIN_MAINTENANCE_20261017.md)
records the tested environment and results.

## Other development checks

For checks outside those maintained targets, use an already prepared environment:

```
uv run --no-sync pytest
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync basedpyright
```

Pytest marks checks that require competition Parquet files, a GPU or a longer
runtime. Select those checks only when their requirements are available. Ruff
uses a 100-character configured line length and targets Python 3.11 syntax.

For changes under src/wnn/training/ or the trainer entrypoint, also run the
structural checker required by the project instructions:

```
make training-structure
```

Read the [training component map](../src/wnn/training/README.md) before adding
training behavior. scripts/train_rnn.py remains a compatibility entrypoint;
new behavior belongs in a focused reusable component. The training map lists
the contracts and the specific checks for that code.
