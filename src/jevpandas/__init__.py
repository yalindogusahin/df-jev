"""Natural-language operations on pandas dataframes."""

from __future__ import annotations

from collections.abc import Callable

from .client import JevClient, JevError
from .frame import JevFrame
from .questions import choice, noul, score
from .result import JevResult

__all__ = [
    "JevClient",
    "JevError",
    "JevFrame",
    "JevResult",
    "choice",
    "noul",
    "score",
    "tqdm_progress",
]


def tqdm_progress(bar=None, **kwargs) -> Callable[[int, int], None]:
    """Return a ``progress`` callback backed by a tqdm bar.

    Pass an existing bar or let one be created from ``**kwargs``. Requires tqdm
    only when no bar is supplied.

    >>> from jevpandas import tqdm_progress
    >>> result = frame.evaluate(condition, columns=["message"], progress=tqdm_progress())
    """
    if bar is None:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("pip install tqdm to use tqdm_progress()") from exc
        bar = tqdm(**kwargs)
    last = [0]

    def report(done: int, total: int) -> None:
        bar.total = total
        bar.update(done - last[0])
        last[0] = done

    return report
