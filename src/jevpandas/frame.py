"""Row-wise semantic judgments that preserve the input dataframe."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import questions as q
from .client import JevClient, JevError
from .result import JevResult

Progress = Callable[[int, int], None]

_VALUE_FIELDS = {
    "noul": ["probability"],
    "choice": ["label", "confidence", "probabilities"],
    "score": ["value", "confidence", "probabilities"],
}
_NUMERIC_FIELDS = {
    "noul": ["probability"],
    "choice": ["confidence"],
    "score": ["value", "confidence"],
}
_META_FIELDS = ["error", "model", "cached"]


class JevFrame:
    """Evaluate natural-language questions against a dataframe, one row at a time.

    ``evaluate``, ``filter``, ``classify``, ``score`` ask a single question;
    ``ask`` sends several questions per row in one HTTP request. Pass
    ``workers=N`` to process rows concurrently; results keep the input order.
    """

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
        workers: int = 1,
    ) -> JevResult:
        """Return all rows with a probability and any request error for each row."""
        return self._run(
            {"type": "noul", "instructions": condition}, columns, name, progress, workers
        )

    def filter(
        self,
        condition: str,
        *,
        columns: Sequence[str],
        threshold: float = 0.7,
        name: str = "match",
        progress: Progress | None = None,
        workers: int = 1,
    ) -> JevResult:
        """Return matching rows. Raise on failed rows rather than silently exclude them."""
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between zero and one")
        result = self.evaluate(
            condition, columns=columns, name=name, progress=progress, workers=workers
        )
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
        workers: int = 1,
    ) -> JevResult:
        return self._run(q.choice(instruction, choices), columns, name, progress, workers)

    def score(
        self,
        instruction: str,
        levels: Sequence[str],
        *,
        columns: Sequence[str],
        name: str = "score",
        progress: Progress | None = None,
        workers: int = 1,
    ) -> JevResult:
        return self._run(q.score(instruction, levels), columns, name, progress, workers)

    def ask(
        self,
        questions: Mapping[str, dict],
        *,
        columns: Sequence[str],
        progress: Progress | None = None,
        workers: int = 1,
    ) -> JevResult:
        """Ask several named questions per row in a single request per row.

        Questions are built with :func:`jevpandas.noul`, :func:`jevpandas.choice`, and
        :func:`jevpandas.score`. Each produces ``{name}_{field}`` columns, e.g.
        ``access_probability``, ``topic_label``, ``severity_value``.
        """
        if not isinstance(questions, Mapping):
            raise TypeError("Supply a mapping of question names to questions")
        if not questions:
            raise ValueError("Supply at least one question")
        specs: dict[str, dict] = {}
        for qname, question in questions.items():
            if not isinstance(qname, str) or not qname.strip():
                raise ValueError("Question names must be nonempty strings")
            if not isinstance(question, dict):
                raise TypeError("Each question must be built with noul(), choice(), or score()")
            if qname in specs:
                raise ValueError("Question names must be unique")
            q.validate(question)
            specs[qname] = question
        return self._run_multi(specs, columns, workers, progress)

    def _run(
        self,
        question: dict,
        columns: Sequence[str],
        name: str,
        progress: Progress | None,
        workers: int,
    ) -> JevResult:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be nonempty")
        return self._run_multi({name: question}, columns, workers, progress)

    def _run_multi(
        self, questions: dict, columns: Sequence[str], workers: int, progress: Progress | None
    ) -> JevResult:
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("Workers must be a positive integer")
        if isinstance(columns, str) or not columns or len(set(columns)) != len(columns):
            raise ValueError("Select one or more distinct columns")
        missing = set(columns) - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Unknown columns: {sorted(missing)}")
        output_names: list[str] = []
        numeric: set[str] = set()
        for qname, question in questions.items():
            q.validate(question)
            kind = question["type"]
            output_names += [f"{qname}_{field}" for field in _VALUE_FIELDS[kind] + _META_FIELDS]
            numeric.update(f"{qname}_{field}" for field in _NUMERIC_FIELDS[kind])
        if set(output_names) & set(self.dataframe.columns):
            raise ValueError("Output columns already exist; choose another name")
        # pandas handles missing values, numpy scalars, and timestamps before JSON encoding.
        states = json.loads(
            self.dataframe[list(columns)].to_json(orient="records", date_format="iso")
        )
        started = time.perf_counter()
        rows = self._run_rows(states, questions, workers, progress)
        result = JevResult(self.dataframe.copy())
        for column in output_names:
            result[column] = [row[column] for row in rows]
        for column in numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result.attrs["jev"] = {
            "questions": questions,
            "columns": list(columns),
            "base_url": self.client.base_url,
            "requested_model": self.client.model,
            "elapsed_seconds": time.perf_counter() - started,
            "rows": len(states),
            "cache_hits": sum(row["cached"] for row in rows),
            "errors": sum(row["error"] is not None for row in rows),
        }
        return result

    def _run_rows(
        self, states: list, questions: dict, workers: int, progress: Progress | None
    ) -> list:
        if workers <= 1:
            rows = []
            for position, state in enumerate(states):
                rows.append(self._process_row(state, questions))
                if progress:
                    progress(position + 1, len(states))
            return rows
        rows = [None] * len(states)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._process_row, state, questions): position
                for position, state in enumerate(states)
            }
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
                completed += 1
                if progress:
                    progress(completed, len(states))
        return rows

    def _process_row(self, state: dict, questions: dict) -> dict:
        row = {"model": None, "cached": False, "error": None}
        for qname, question in questions.items():
            for field in _VALUE_FIELDS[question["type"]] + _META_FIELDS:
                row[f"{qname}_{field}"] = None
        try:
            if all(value is None or value == "" for value in state.values()):
                raise JevError("Selected fields contain no data")
            response, cached = self.client.decide_many(state, questions)
            row["model"] = response["model"]
            row["cached"] = cached
            for qname in questions:
                row[f"{qname}_model"] = response["model"]
                row[f"{qname}_cached"] = cached
            for qname, question in questions.items():
                answer = response["answers"][qname]
                kind = question["type"]
                if kind == "noul":
                    row[f"{qname}_probability"] = answer["noul"]
                else:
                    row[f"{qname}_label" if kind == "choice" else f"{qname}_value"] = answer[
                        "choice" if kind == "choice" else "score"
                    ]
                    row[f"{qname}_confidence"] = answer["confidence"]
                    row[f"{qname}_probabilities"] = json.dumps(
                        answer["probabilities"], sort_keys=True
                    )
        except JevError as exc:
            row["error"] = str(exc)
            for qname in questions:
                row[f"{qname}_error"] = str(exc)
        return row
