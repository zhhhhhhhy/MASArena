# project_multi_agents_benchmark

## Quick Start

We highly recommend you to use [uv](https://docs.astral.sh/uv/) to manages project dependencies

After installation, sync the project's dependencies with the environment:

```bash
uv sync
```

And we equipped a pre-commit hook for this project:

```bash
pre-commit install
```

When you want to add a dependency to the project:

```bash
uv add [package]
```

The packages installed by `pip` would NOT be added into project dependencies.

### Optional

You could use ruff as linter and formatter

```bash
ruff check
ruff format
```
