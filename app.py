"""Run with: uv run streamlit run app.py"""

import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from df_jev import JevClient, JevFrame

ROOT = Path(__file__).parent
CONDITION = (
    "The incident describes a current, unresolved problem that prevents customers from logging in "
    "or accessing the service. Resolved incidents, planned maintenance, internal employee access "
    "issues, and performance issues without loss of access do not match."
)

st.set_page_config(page_title="df-jev · Data explorer", page_icon="🔎", layout="wide")
st.title("Explore your data by meaning")
st.write(
    "Describe what matters, inspect the matching records, and keep the results as a dataframe."
)

with st.sidebar:
    st.header("Connection")
    base_url = st.text_input(
        "Endpoint", os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
    )
    model = st.text_input("Model", os.getenv("TYPESAFE_MODEL", "jev-latest"))
    api_key = st.text_input("API key", os.getenv("TYPESAFE_API_KEY", "dummy"), type="password")
    st.caption("Only your selected columns are sent to this endpoint when you run an operation.")
    config = (base_url, model, api_key)
    if st.session_state.get("client_config") != config:
        try:
            new_client = JevClient(base_url, api_key, model)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if "client" in st.session_state:
            st.session_state.client.close()
        st.session_state.client = new_client
        st.session_state.client_config = config
    client = st.session_state.client
    if st.button("Clear cached decisions"):
        client.clear_cache()
        st.success("Cache cleared. Your next run will request fresh decisions.")
    st.caption(
        "Decisions are cached in this browser session. Refreshing the page clears the session."
    )

source = st.radio("Data source", ["Sample incidents", "Upload a file"], horizontal=True)
try:
    if source == "Sample incidents":
        raw = (ROOT / "data" / "incidents.csv").read_bytes()
        df = pd.read_csv(BytesIO(raw))
        st.caption("100 synthetic incidents, including resolved cases and misleading keywords.")
    else:
        uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet"])
        if uploaded is None:
            st.info("Upload a dataset to begin, or use the sample incidents.")
            st.stop()
        raw = uploaded.getvalue()
        df = (
            pd.read_parquet(BytesIO(raw))
            if uploaded.name.lower().endswith(".parquet")
            else pd.read_csv(BytesIO(raw))
        )
except (OSError, ValueError, TypeError) as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()

if df.empty:
    st.info("This dataset has no rows. Choose a file containing records.")
    st.stop()
if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
    st.error("Column names must be unique strings.")
    st.stop()
if "review_label" in df.columns:
    st.error("Rename the reserved column 'review_label' before uploading.")
    st.stop()

dataset_id = sha256(raw).hexdigest()
with st.expander(f"Dataset · {len(df):,} rows · {len(df.columns)} columns", expanded=False):
    st.dataframe(df.head(20), hide_index=True, use_container_width=True)
default_columns = [c for c in ["subject", "message"] if c in df.columns]
if not default_columns:
    default_columns = list(df.select_dtypes(include=["object", "string"]).columns[:2])
columns = st.multiselect(
    "Columns to read", list(df.columns), default=default_columns, key=f"columns_{dataset_id}"
)
operation = st.radio("Operation", ["Find matches", "Classify", "Score"], horizontal=True)
defaults = {
    "Find matches": CONDITION,
    "Classify": "What is the main topic of this incident?",
    "Score": "How severely does the current incident disrupt customers' use of the service?",
}
instruction = st.text_area(
    "Describe the condition" if operation == "Find matches" else "Instruction",
    defaults[operation],
    key=f"instruction_{operation}",
    height=110,
)
criteria = None
if operation == "Classify":
    options = st.text_area(
        "Categories · one per line, optionally label: description",
        "access: Login and authentication failures\nbilling: Payments, charges and refunds\n"
        "technical: Other software errors or outages\nother: None of these categories",
    )
    pairs = [line.partition(":") for line in options.splitlines() if line.strip()]
    criteria = {
        label.strip(): description.strip() if sep else label.strip()
        for label, sep, description in pairs
    }
    if len(criteria) != len(pairs):
        st.error("Category labels must be unique.")
        st.stop()
elif operation == "Score":
    rubric = st.text_area(
        "Ordered levels · lowest first, one per line",
        "No current disruption\nSome functionality is impaired\nCustomers cannot use the service",
    )
    criteria = [line.strip() for line in rubric.splitlines() if line.strip()]

signature = sha256(
    json.dumps([dataset_id, columns, operation, instruction, criteria, config]).encode()
).hexdigest()
left, right = st.columns(2)
preview = left.button(
    "Preview first 10 rows",
    disabled=not columns or not instruction.strip(),
    use_container_width=True,
)
full = right.button(
    f"Run all {len(df):,} rows",
    type="primary",
    disabled=not columns or not instruction.strip(),
    use_container_width=True,
)
if preview or full:
    subset = df.head(10) if preview else df
    bar = st.progress(0, text="Starting…")

    def progress(done, total):
        bar.progress(done / total, text=f"Processed {done:,} of {total:,} rows")

    try:
        frame = JevFrame(subset, client)
        kwargs = {"columns": columns, "name": "jev", "progress": progress}
        if operation == "Find matches":
            result = frame.evaluate(instruction, **kwargs)
        elif operation == "Classify":
            result = frame.classify(instruction, criteria, **kwargs)
        else:
            result = frame.score(instruction, criteria, **kwargs)
        st.session_state.run = {
            "signature": signature,
            "result": result.reset_index(drop=True),
            "id": uuid4().hex,
            "preview": preview,
            "source_columns": list(df.columns),
        }
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    finally:
        bar.empty()

run = st.session_state.get("run")
if not run:
    st.info("Start with a preview to check how your wording performs.")
    st.stop()
if run["signature"] != signature:
    st.info("Your dataset or question changed. Run a preview to see updated results.")
    st.stop()

result = run["result"]
metadata = result.attrs["jev"]
valid = result.jev_error.isna()
st.subheader("Results" + (" · preview" if run["preview"] else ""))
st.caption(
    f"{len(result):,} rows processed in {metadata['elapsed_seconds']:.2f}s · "
    f"{metadata['cache_hits']:,} cached · "
    f"Model: {', '.join(result.jev_model.dropna().unique()) or 'no successful responses'}"
)
if not valid.all():
    st.warning(
        f"{(~valid).sum():,} rows failed. They remain in the full export and review table. "
        "Rerun to retry; successful decisions are cached."
    )

if operation == "Find matches":
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.7, 0.01)
    st.caption(
        "This is the model's reported probability, not a verified accuracy estimate. "
        "Changing the threshold does not call the model."
    )
    predicted = result.jev_probability >= threshold
    view = st.radio(
        "Show", ["Matches", "All rows", "Near threshold", "Failed rows"], horizontal=True
    )
    masks = {
        "Matches": valid & predicted,
        "All rows": pd.Series(True, index=result.index),
        "Near threshold": valid & ((result.jev_probability - threshold).abs() <= 0.15),
        "Failed rows": ~valid,
    }
    visible = result.loc[masks[view]].sort_values(
        "jev_probability", ascending=False, na_position="last"
    )
elif operation == "Classify":
    chosen = st.multiselect("Show categories", list(criteria), default=list(criteria))
    visible = result.loc[valid & result.jev_label.isin(chosen)]
else:
    minimum = st.slider("Minimum score", 0.0, float(len(criteria) - 1), 0.0, 0.1)
    visible = result.loc[valid & (result.jev_value >= minimum)].sort_values(
        "jev_value", ascending=False
    )

st.write(f"**{len(visible):,} rows in this view**")
value_columns = [
    c for c in result.columns if c not in df.columns and c not in ["jev_cached", "jev_model"]
]
st.dataframe(visible[value_columns + list(df.columns)], hide_index=True, use_container_width=True)
group_column = st.selectbox(
    "Count this view by",
    ["None"] + list(df.columns),
    index=1 + list(df.columns).index("service") if "service" in df.columns else 0,
)
if group_column != "None" and not visible.empty:
    counts = visible[group_column].fillna("(missing)").astype(str).value_counts().rename("Rows")
    st.bar_chart(counts.head(30))
    if len(counts) > 30:
        st.caption("Showing the 30 largest groups.")

with st.expander("Review predictions · all processed rows"):
    st.caption(
        "Labels stay attached to their original rows as you change the result view. "
        "Download your labels before starting a new run or refreshing the page."
    )
    review = result.copy()
    review["review_label"] = "Unreviewed"
    review_options = (
        ["Unreviewed", "Match", "Not a match", "Unsure"]
        if operation == "Find matches"
        else ["Unreviewed", "Correct", "Incorrect", "Unsure"]
    )
    edited = st.data_editor(
        review,
        key=f"review_{run['id']}",
        hide_index=True,
        use_container_width=True,
        column_order=["review_label"] + value_columns + list(df.columns),
        disabled=list(result.columns),
        column_config={
            "review_label": st.column_config.SelectboxColumn(
                "Your judgment", options=review_options, required=True
            )
        },
    )
    if operation == "Find matches":
        labeled = valid & edited.review_label.isin(["Match", "Not a match"])
        if labeled.any():
            actual = edited.loc[labeled, "review_label"].eq("Match")
            guesses = predicted.loc[labeled]
            tp, fp, fn = (
                (guesses & actual).sum(),
                (guesses & ~actual).sum(),
                (~guesses & actual).sum(),
            )
            st.write(
                f"Agreement on {labeled.sum()} reviewed rows: {(guesses == actual).mean():.1%}"
            )
            st.write(f"True positives: {tp} · False positives: {fp} · False negatives: {fn}")
    else:
        labeled = valid & edited.review_label.isin(["Correct", "Incorrect"])
        if labeled.any():
            st.write(
                f"Marked correct: {edited.loc[labeled, 'review_label'].eq('Correct').mean():.1%} "
                f"of {labeled.sum()} reviewed rows"
            )
    st.download_button(
        "Download reviewed rows", edited.to_csv(index=False).encode(), "reviewed.csv", "text/csv"
    )

a, b, c = st.columns(3)
a.download_button(
    "Export this view",
    visible.to_csv(index=False).encode(),
    "matches.csv",
    "text/csv",
    use_container_width=True,
)
b.download_button(
    "Export all processed rows",
    result.to_csv(index=False).encode(),
    "enriched.csv",
    "text/csv",
    use_container_width=True,
)
export_metadata = {
    **metadata,
    "dataset_sha256": dataset_id,
    "preview": run["preview"],
    "returned_models": list(result.jev_model.dropna().unique()),
    "view": view if operation == "Find matches" else operation,
    "threshold": threshold if operation == "Find matches" else None,
    "visible_rows": len(visible),
    "categories": chosen if operation == "Classify" else None,
    "minimum_score": minimum if operation == "Score" else None,
}
c.download_button(
    "Export run details",
    json.dumps(export_metadata, indent=2),
    "run.json",
    "application/json",
    use_container_width=True,
)
