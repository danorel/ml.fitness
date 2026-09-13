# Ballmfs

LLM experiments (`LLM.ipynb`).

## Setup

Own `uv` environment (Python 3.12).

```bash
cd src/ballmfs
uv sync                # creates .venv, installs deps from pyproject.toml
uv run python -m ipykernel install --user --name ballmfs --display-name "ballmfs (uv)"
```

Open `LLM.ipynb` and select kernel **ballmfs (uv)**.

To add a dependency: `uv add <package>`.
