# jevpandas

Explore a pandas dataframe using natural-language judgments from [Jev](https://docs.typesafe.ai/introduction)
(TypeSafe System One) compatible endpoints. Find incidents by meaning, classify records, or score
them against a rubric — from a notebook or a script. No generative chat model needed.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
export TYPESAFE_BASE_URL=https://api.typesafe.ai/v1
export TYPESAFE_API_KEY=your-key
export TYPESAFE_MODEL=jev-latest
```

`TYPESAFE_BASE_URL` and `TYPESAFE_MODEL` default to the official Jev endpoint
(`https://api.typesafe.ai/v1`, `jev-latest`). Any TypeSafe System One compatible endpoint works;
pass `base_url`, `api_key`, and `model` to `JevClient` to override.

## Notebook

```python
import pandas as pd
from jevpandas import JevClient, JevFrame, noul, choice, score, tqdm_progress

df = pd.read_csv("data/incidents.csv")

with JevClient() as client:
    incidents = JevFrame(df, client)

    # All rows, including probabilities and explicit row errors.
    evaluated = incidents.evaluate(
        "Customers cannot access the service and the problem remains unresolved.",
        columns=["subject", "message"],
        workers=4,  # process rows concurrently, order preserved
        progress=tqdm_progress(),  # optional progress bar
    )
    matches = incidents.filter(
        "Customers cannot access the service and the problem remains unresolved.",
        columns=["subject", "message"],
        threshold=0.7,
        workers=4,
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

`evaluate`, `classify`, and `score` return a `JevResult` (a `pandas.DataFrame` subclass) with
additional columns and a run summary in its HTML repr. Index order and duplicate index labels are
preserved. Use `name=` to choose an output prefix and avoid collisions when adding multiple
judgments. `filter` raises if any row fails; use `evaluate` to inspect partial results. Missing
selected values are encoded as JSON null; entirely empty rows are reported as errors without making
a model call. Score values range from zero to `len(levels)-1`.

### Ask several questions in one pass

`ask` sends every question for a row in a single request, which is much cheaper than a separate
pass per question:

```python
result = incidents.ask(
    {
        "access": noul("Is this a current, unresolved customer access problem?"),
        "topic": choice(
            "What is the main topic?", {"access": "Login", "billing": "Payments", "other": "Other"}
        ),
        "severity": score("How severe is the current disruption?", ["None", "Impaired", "Outage"]),
    },
    columns=["subject", "message"],
    workers=4,
)
```

Each question produces `{name}_{field}` columns: `access_probability`, `topic_label`,
`severity_value`, plus `{name}_error`, `{name}_model`, `{name}_cached` per question.

### Concurrency and caching

Rows run sequentially by default (`workers=1`); pass `workers=N` to evaluate them concurrently
with a thread pool. Results are always assembled in the original row order, and a failed row never
affects the others. The client retries transient network errors, HTTP 429, and server errors;
failures are never turned into negative predictions. Identical successful requests are cached in
memory (up to 10,000 entries per client/session), guarded by a lock so parallel runs are safe.
Changing the instruction, selected data, endpoint, or requested model invalidates the relevant
cache entry. Threshold changes need no requests. Pin the model version, or call `clear_cache()`
when an alias changes. No API keys or datasets are stored in a disk cache.

### Notebook display

`JevResult` renders with a summary line (rows, elapsed time, cache hits, errors, model) above the
table. Run metadata is also available as `result.metadata` and stored in `result.attrs["jev"]`.

## Notes

Model probabilities and confidence are backend-reported values, not verified accuracy guarantees.
OpenJev and hosted Jev have different models and must be evaluated separately. This version does
not translate arbitrary chat into code, generate explanations, run joins, or train on review
labels. Samples and their expected labels in `data/` are synthetic and are not evidence of
accuracy on production data.

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
