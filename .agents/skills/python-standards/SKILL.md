---
name: python-standards
description: "Python coding standards  — type annotations, naming, error handling, async, module structure, and documentation conventions. Triggers: writing new Python modules, adding functions, reviewing code, 'follow standards', 'code style', and any Python file creation or significant edit."
---

# Python Development Standards

Apply these rules to all Python code in this project.

## Module Structure

- Every module starts with `from __future__ import annotations`, then a blank line, then imports.
- Import order: stdlib > third-party > local (`dataiku_gitops.*`). Use ruff or isort to enforce.
- Keep modules focused. If a module grows beyond ~300 lines, consider splitting it.

## Formatter (ruff format) — don't fight it

`ruff format` (via `make format` / pre-commit) is the source of truth for
whitespace, line breaks, and wrapping. Write code that's already formatter-clean
so commits don't generate churn. A few concrete rules that come up constantly:

- **Use the configured line length** in `pyproject.toml` (currently100). Ruff is the wrapping source of truth; do not assume its upstream default.
- **Do not hand-wrap argument lists.** Write the call on one line and let the
  formatter decide whether it fits. If it doesn't, ruff will split it — your
  hand-picked break points will just be redone.
- **Magic trailing comma**: a trailing comma in a call / def / list / tuple /
  dict opts into multi-line formatting and keeps each item on its own line.
  Omit the trailing comma when you want ruff to collapse to one line if it fits.
- **Don't hand-format `raise TypeError(f"...")` or `argparse.add_argument(...)`
  across multiple lines**. If the expression fits, write it on one line; if it
  doesn't, write it as one logical line with a trailing comma before the `)`
  and let ruff lay it out.
- **Don't break function signatures manually when they fit**. Example: ruff
  collapses `def _date_bounds(\n    df, time_column\n) -> tuple[...]:` back to
  one line if the whole signature is under 88 chars.
- **Before declaring a change done**, run `uv run ruff format <path>` and
  `uv run ruff check <path>` — or at minimum `ruff format --check` so you see
  the diff the pre-commit hook would apply. Fix the code, don't `# noqa` it.
- **Don't pre-break long docstring lines** hoping to stay under 88 chars if
  the break hurts readability. The formatter doesn't touch docstring content.

## Type Annotations

- Annotate every function and method signature (parameters and return types).
- Use built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, ...]`.
- Use PEP 604 unions: `str | None`, not `Optional[str]`.
- Do not use `Any` without a comment explaining why it is unavoidable.
- Do not silence type errors with `# type: ignore` without a comment.

## Classes

This project uses **attrs** and **cattrs**:

```python
from attrs import define, field, frozen

@frozen
class MyValue:
    x: int
    y: str = "default"

@define
class MyService:
    _items: list[str] = field(factory=list, init=False, repr=False)
```

- Use `@frozen` for immutable value objects / DTOs.
- Use `@define` for services with mutable internal state.
- Use `field(init=False, repr=False)` for private or derived fields.
- Never use mutable literal defaults; always use `field(factory=...)`.
- Use `cattrs.structure` / `cattrs.unstructure` for serialization.
- Use Pydantic only for settings classes that load from env vars.

## Functions

- Prefer small, pure functions with minimal side effects.
- Keep functions short and cohesive.
- Default arguments must be immutable literals or `None`.
- Avoid flag arguments; prefer clearer function names.

## Error Handling

- Raise specific, descriptive exceptions.
- Never use bare `except:` or `except Exception:` without re-raising or logging.
- Do not use exceptions for normal control flow.
- Validate at boundaries (CLI args, MCP tool inputs, Dataiku API responses), then trust internal objects.

## Naming

| Thing | Convention | Example |
|---|---|---|
| Modules, packages | `snake_case` | `sql_compat` |
| Classes | `PascalCase` | `FlowNode` |
| Functions, methods, variables | `snake_case` | `pull_zone` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT` |
| Private attributes | leading `_` | `_cache` |

- Use intent-revealing names.
- Boolean variables and functions should start with `is_`, `has_`, `can_`, or `should_`.

## Documentation

Use **Google-style** docstrings (`Args:`, `Returns:`, `Raises:`).

- Every public function should have a docstring describing what it does and why.
- Document parameters, return values, and exceptions for non-trivial functions.
- Comment the **why**, not the **what** in inline comments.

## Security

- Never commit secrets, tokens, or credentials. Use `.env` files loaded via `python-dotenv`.
- Never log sensitive payloads (API keys, cookie values).
- Validate and sanitize all external inputs at MCP tool boundaries.

## Avoid

- `global` and `nonlocal`.
- Monkey-patching outside tests.
- Implicit `None` returns for value-returning functions.
- Catching and silently discarding exceptions.
- String formatting with `%` or `.format()`; use f-strings.
- `print` for observability; use `logging` or structured output.

## Recurrent training architecture

Use `src/wnn/training/README.md` to find existing reusable components. Keep scripts/train_rnn.py a small compatibility CLI, and use explicit TrainingEnvironment/InputConfiguration objects for new code. Do not clone a full trainer for an experiment. The package and entrypoint have enforced300-line module and200-line function budgets; run `uv run python scripts/check_training_structure.py` with Ruff. Source/CLI/AST-packet compatibility and resume provenance are part of a refactor's scope. Static checks alone do not establish numerical or throughput parity. See ADR0041 and its research receipt.
