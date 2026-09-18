"""Live evaluation against the configured endpoint; outputs are ignored by git."""

import json
import time
from pathlib import Path

import pandas as pd
from generate_sample import CONDITION

from df_jev import JevClient, JevFrame

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data/incidents.csv")
    labels = pd.read_csv(ROOT / "data/incident_labels.csv").set_index("incident_id")
    output = ROOT / "output"
    output.mkdir(exist_ok=True)

    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(f"Processed {done}/{total}", flush=True)

    with JevClient() as client:
        frame = JevFrame(df, client)
        result = frame.evaluate(CONDITION, columns=["subject", "message"], progress=progress)
        valid = result.match_error.isna()
        predicted = result.match_probability >= 0.7
        actual = result.incident_id.map(labels.expected_match)
        summary = {
            **result.attrs["jev"],
            "fixture": "100 synthetic rows from 20 scenario templates, each used with 5 services",
            "threshold": 0.7,
            "returned_models": result.match_model.dropna().unique().tolist(),
            "true_positives": int((valid & predicted & actual).sum()),
            "false_positives": int((valid & predicted & ~actual).sum()),
            "true_negatives": int((valid & ~predicted & ~actual).sum()),
            "false_negatives": int((valid & ~predicted & actual).sum()),
            "agreement_on_successful_rows": float((predicted[valid] == actual[valid]).mean())
            if valid.any()
            else None,
        }
        result.to_csv(output / "enriched.csv", index=False)
        result.loc[valid & predicted].to_csv(output / "matches.csv", index=False)
        result.loc[valid & (predicted != actual)].to_csv(output / "disagreements.csv", index=False)
        if valid.all():
            started = time.perf_counter()
            cached = frame.evaluate(CONDITION, columns=["subject", "message"])
            summary["cached_run_seconds"] = time.perf_counter() - started
            summary["cached_run_hits"] = int(cached.match_cached.sum())
            assert cached.match_cached.all(), "The repeated evaluation should be served from cache"
            assert cached.match_probability.equals(result.match_probability)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if not valid.all():
            raise SystemExit("Some rows failed; inspect output/enriched.csv")


if __name__ == "__main__":
    main()
