"""Builders for TypeSafe System One questions.

These return plain dicts in the wire format, so they can be inspected, logged,
or reused across calls. ``ask`` and the single-question operations validate
anything built here before it is sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def noul(instructions: str) -> dict:
    """A true/false judgment, returned as a 0–1 probability."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: Mapping[str, str] | Sequence[str]) -> dict:
    """Pick one category: a mapping of label -> description, or a list of labels."""
    return {"type": "choice", "instructions": instructions, "criteria": _criteria(criteria)}


def score(instructions: str, levels: Sequence[str]) -> dict:
    """Score the state on an ordered rubric; lowest level first."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def validate(question: dict) -> None:
    """Validate a question built by :func:`noul`, :func:`choice`, or :func:`score`."""
    if not isinstance(question, dict) or question.get("type") not in {"noul", "choice", "score"}:
        raise ValueError("Unknown question type")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Write a nonempty instruction")
    if question["type"] == "choice":
        criteria = question.get("criteria")
        if (
            not isinstance(criteria, Mapping)
            or not 2 <= len(criteria) <= 255
            or not all(
                isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                for k, v in criteria.items()
            )
        ):
            raise ValueError("Supply 2–255 nonempty category labels and descriptions")
    elif question["type"] == "score":
        criteria = question.get("criteria")
        if (
            not isinstance(criteria, Sequence)
            or isinstance(criteria, (str, bytes))
            or not 2 <= len(criteria) <= 10
            or not all(isinstance(level, str) and level.strip() for level in criteria)
        ):
            raise ValueError("Supply 2–10 nonempty ordered score levels")


def _criteria(choices: Mapping[str, str] | Sequence[str]) -> dict:
    if isinstance(choices, (str, bytes)):
        raise TypeError("Supply a list of labels or a mapping of labels to descriptions")
    if not isinstance(choices, Mapping) and len(set(choices)) != len(choices):
        raise ValueError("Category labels must be unique")
    return dict(choices) if isinstance(choices, Mapping) else {c: c for c in choices}
