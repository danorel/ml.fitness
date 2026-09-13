# Autograd

Minimal scalar-valued autograd engine + notebooks.

## Setup

Own `uv` environment (Python 3.12).

```bash
cd src/autograd
uv sync                # creates .venv, installs deps from pyproject.toml
uv run python -m ipykernel install --user --name autograd --display-name "autograd (uv)"
```

Open `notebooks/*.ipynb` and select kernel **autograd (uv)**.

To add a dependency: `uv add <package>`.
