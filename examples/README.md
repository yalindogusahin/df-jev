# Examples

- `test_jevpandas.ipynb` — a ready-to-open notebook that exercises `jevpandas` from Jupyter:
  it loads the 1,000-row synthetic fixture (`data/incidents.parquet`; a matching CSV is also
  provided), then runs a live run against the official Jev endpoint and an offline run against a
  mocked endpoint (no server, no API key).

Open it with:

```bash
uv sync --extra dev --extra notebook  # adds pandas, jupyterlab, ipykernel, tqdm, pytest, ruff to .venv
uv run jupyter lab examples/test_jevpandas.ipynb
# or: uv run jupyter notebook examples/test_jevpandas.ipynb
```

Launching through `uv run` makes the notebook kernel the project's `.venv`, which already has
pandas. If you open the notebook in another Jupyter/VS Code kernel instead, select the
`.venv` Python (or the "Python (jevpandas)" kernel) as the interpreter — otherwise pandas
won't be importable there.

Set `YOUR_KEY` in the live cell (or point `JevClient` at a local OpenJev/SemIf server) before
running the live path. The local and offline cells run as-is.
