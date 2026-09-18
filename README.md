# df-jev

Explore a pandas dataframe using natural-language conditions. Find incidents by meaning,
classify records, or score them against a rubric, then inspect and export the results.

The first example finds **unresolved customer access problems** in 100 synthetic incidents.
The app calls your Jev deployment, configured with `TYPESAFE_BASE_URL` (default
`https://api.typesafe.ai/v1`); it also supports other TypeSafe System One compatible endpoints.
It does not need a generative chat model.

## Run

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra app --extra dev
export TYPESAFE_BASE_URL=https://api.typesafe.ai/v1
export TYPESAFE_API_KEY=dummy
export TYPESAFE_MODEL=jev-latest
uv run streamlit run app.py
```

Open http://localhost:8501. Use **Sample incidents** or upload CSV/Parquet, select the columns
to read, and preview the first ten rows before running the full dataset. Adjust the threshold,
inspect uncertain matches, count results by a column, and export CSV files. The review table
lets you mark matches/non-matches and compare your labels with predictions at the current threshold.
Labels can be exported; they are not saved after a new run or browser refresh.

Only selected columns are sent to the configured endpoint. Original columns remain in the
result locally. Uploading alone makes no inference requests. Samples and their expected labels
are synthetic and are not evidence of accuracy on production data.

## Python

```python
import pandas as pd
from df_jev import JevClient, JevFrame

df = pd.read_csv("data/incidents.csv")

with JevClient() as client:
    incidents = JevFrame(df, client)
    # All rows, including probabilities and explicit row errors.
    evaluated = incidents.evaluate(
        "Customers cannot access the service and the problem remains unresolved.",
        columns=["subject", "message"],
    )
    matches = incidents.filter(
        "Customers cannot access the service and the problem remains unresolved.",
        columns=["subject", "message"],
        threshold=0.7,
    )
    print(matches.groupby("service").size())

    classified = incidents.classify(
        "What is the main topic?",
        choices={
            "access": "Login or authentication",
            "billing": "Payments or refunds",
            "technical": "Other service problems",
            "other": "Anything else",
        },
        columns=["subject", "message"],
    )
    scored = incidents.score(
        "How severely does this currently disrupt customers?",
        levels=["No current disruption", "Partly impaired", "Unable to use the service"],
        columns=["subject", "message"],
    )
```

`evaluate`, `classify`, and `score` return copies with additional columns. Index order and
duplicate index labels are preserved. Use `name=` to choose an output prefix and avoid collisions
when adding multiple judgments. `filter` raises if any row fails; use `evaluate` to inspect
partial results. Missing selected values are encoded as JSON null; entirely empty rows are
reported as errors without making a model call. Score values range from zero to `len(levels)-1`.

Requests run sequentially, one per row. The client retries transient network errors, HTTP 429,
and server errors; failures are never turned into negative predictions. Identical successful
requests are cached in memory (up to 10,000 entries per client/session). Changing the instruction,
selected data, endpoint, or requested model invalidates the relevant cache entry. Threshold changes
need no requests. Pin the model version, or clear the cache when an alias changes. No API keys
or datasets are stored in a disk cache.

Dataframe operations perform counting, grouping, and date arithmetic. Model probabilities and
confidence are backend-reported values, not verified accuracy guarantees. OpenJev and hosted Jev
have different models and must be evaluated separately. This first version does not translate
arbitrary chat into code, generate explanations, run joins, or train on review labels.

## Validate

```bash
uv run pytest
uv run ruff check .
# Calls the configured server on 100 synthetic rows and writes ignored output/ files:
uv run python scripts/evaluate_sample.py
```

The live script exports all predictions, matching incidents, and a summary with a confusion
matrix, timing, returned model IDs, and a check that repeating the run uses cached decisions.
Expected labels are kept in `data/incident_labels.csv` and are never sent to the model.

## Protocol references

- [TypeSafe primitives](https://docs.typesafe.ai/introduction)
- [Jev model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [OpenJev / SemIf upstream](https://github.com/TheoLeeCJ/SemIf)

The current local server advertises OpenJev 0.1 backed by DiffusionGemma 26B-A4B.
Its `/openapi.json` defines the protocol used here; no assumptions about the upstream
repository's current model are needed.
