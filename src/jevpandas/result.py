"""A dataframe subclass with a notebook-friendly summary for judgment results."""

from __future__ import annotations

import pandas as pd


class JevResult(pd.DataFrame):
    """A result frame that carries run metadata and shows it in notebooks.

    Behaves like a normal :class:`pandas.DataFrame`; the only additions are a
    ``metadata`` property and a richer HTML repr for Jupyter.
    """

    @property
    def _constructor(self) -> type[pd.DataFrame]:
        return JevResult

    @property
    def metadata(self) -> dict:
        """Run metadata recorded by the operation that produced this frame."""
        return self.attrs.get("jev", {})

    def _repr_html_(self) -> str:
        meta = self.metadata
        summary = ""
        if meta:
            summary = (
                "<div style='margin-bottom:6px;font-size:0.85em;color:#444'>"
                f"<b>jev</b> · {meta['rows']:,} rows · {meta['elapsed_seconds']:.2f}s · "
                f"{meta['cache_hits']:,} cached · {meta['errors']:,} errors · "
                f"model <code>{meta['requested_model']}</code></div>"
            )
        return summary + super()._repr_html_()
