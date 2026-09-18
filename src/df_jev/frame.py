"""Row-wise semantic judgments that preserve the input dataframe."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .client import JevClient, JevError

Progress = Callable[[int, int], None]


class JevFrame:
    def __init__(self, dataframe: pd.DataFrame, client: JevClient):
        if not dataframe.columns.is_unique or not all(
            isinstance(c, str) for c in dataframe.columns
        ):
            raise ValueError("Dataframe columns must have unique string names")
        self.dataframe = dataframe
        self.client = client

    def evaluate(
        self,
        condition: str,
        *,
        columns: Sequence[str],
        name: str = "match",
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        """Return all rows with a probability and any request error for each row."""
        return self._run({"type": "noul", "instructions": condition}, columns, name, progress)

    def filter(
        self,
        condition: str,
        *,
        columns: Sequence[str],
        threshold: float = 0.7,
        name: str = "match",
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        """Return matching rows. Raise on failed rows rather than silently exclude them."""
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between zero and one")
        result = self.evaluate(condition, columns=columns, name=name, progress=progress)
        if result[f"{name}_error"].notna().any():
            raise JevError("Some rows failed. Use evaluate() to inspect row errors and retry.")
        return result.loc[result[f"{name}_probability"] >= threshold].copy()

    def classify(
        self,
        instruction: str,
        choices: Mapping[str, str] | Sequence[str],
        *,
        columns: Sequence[str],
        name: str = "category",
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        if isinstance(choices, (str, bytes)):
            raise TypeError("Supply a list of labels or a mapping of labels to descriptions")
        if not isinstance(choices, Mapping) and len(set(choices)) != len(choices):
            raise ValueError("Category labels must be unique")
        criteria = dict(choices) if isinstance(choices, Mapping) else {c: c for c in choices}
        if not 2 <= len(criteria) <= 255 or not all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
            for k, v in criteria.items()
        ):
            raise ValueError("Supply 2–255 nonempty category labels and descriptions")
        return self._run(
            {"type": "choice", "instructions": instruction, "criteria": criteria},
            columns,
            name,
            progress,
        )

    def score(
        self,
        instruction: str,
        levels: Sequence[str],
        *,
        columns: Sequence[str],
        name: str = "score",
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        if (
            isinstance(levels, (str, bytes))
            or not 2 <= len(levels) <= 10
            or not all(isinstance(level, str) and level.strip() for level in levels)
        ):
            raise ValueError("Supply 2–10 nonempty ordered score levels")
        return self._run(
            {"type": "score", "instructions": instruction, "criteria": list(levels)},
            columns,
            name,
            progress,
        )

    def _run(
        self, question: dict, columns: Sequence[str], name: str, progress: Progress | None
    ) -> pd.DataFrame:
        if not isinstance(question["instructions"], str) or not question["instructions"].strip():
            raise ValueError("Write a nonempty instruction")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be nonempty")
        if isinstance(columns, str) or not columns or len(set(columns)) != len(columns):
            raise ValueError("Select one or more distinct columns")
        missing = set(columns) - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Unknown columns: {sorted(missing)}")
        kind = question["type"]
        fields = {
            "noul": ["probability"],
            "choice": ["label", "confidence", "probabilities"],
            "score": ["value", "confidence", "probabilities"],
        }[kind]
        fields += ["error", "model", "cached"]
        output_names = [f"{name}_{field}" for field in fields]
        if set(output_names) & set(self.dataframe.columns):
            raise ValueError("Output columns already exist; choose another name")
        # pandas handles missing values, numpy scalars, and timestamps before JSON encoding.
        states = json.loads(
            self.dataframe[list(columns)].to_json(orient="records", date_format="iso")
        )
        output = {field: [] for field in fields}
        started = time.perf_counter()
        for position, state in enumerate(states):
            row = dict.fromkeys(fields)
            row["cached"] = False
            try:
                if all(value is None or value == "" for value in state.values()):
                    raise JevError("Selected fields contain no data")
                response, cached = self.client.decide(state, question)
                answer = response["answers"]["result"]
                row.update(model=response["model"], cached=cached)
                if kind == "noul":
                    row["probability"] = answer["noul"]
                else:
                    row["label" if kind == "choice" else "value"] = answer[
                        "choice" if kind == "choice" else "score"
                    ]
                    row["confidence"] = answer["confidence"]
                    row["probabilities"] = json.dumps(answer["probabilities"], sort_keys=True)
            except JevError as exc:
                row["error"] = str(exc)
            for field in fields:
                output[field].append(row[field])
            if progress:
                progress(position + 1, len(states))
        result = self.dataframe.copy()
        for field, values in output.items():
            # Assign by position, including when the input index has duplicate labels.
            result[f"{name}_{field}"] = values
        for field in ["probability", "value", "confidence"]:
            if field in output:
                result[f"{name}_{field}"] = pd.to_numeric(
                    result[f"{name}_{field}"], errors="coerce"
                )
        result.attrs["jev"] = {
            "question": question,
            "columns": list(columns),
            "base_url": self.client.base_url,
            "requested_model": self.client.model,
            "elapsed_seconds": time.perf_counter() - started,
            "rows": len(states),
            "cache_hits": sum(output["cached"]),
            "errors": sum(error is not None for error in output["error"]),
        }
        return result
