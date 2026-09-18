# First live run — 2026-09-18

Endpoint: `http://<internal-endpoint>:8000`, returned model: `openjev-0.1`.

Used `scripts/evaluate_sample.py` on the 100 synthetic incidents, sending only
`subject` and `message`. The condition asks for current unresolved customer access
problems and excludes resolved issues, planned maintenance, internal employee
access issues, and performance degradation without loss of access. Threshold: 0.7.

| Result | Count |
| --- | ---: |
| True positives | 40 |
| True negatives | 60 |
| False positives | 0 |
| False negatives | 0 |
| Request errors | 0 |

The sequential run took 26.112 seconds. Repeating it on the same client took
0.0054 seconds with all 100 decisions served from memory.

This is an integration smoke test using 20 authored scenarios, each repeated
with five service names. It is not 100 independent examples and does not establish
accuracy or probability calibration on real data. Review real incidents before
choosing a working threshold. This result applies to this local deployment,
not to TypeSafe's hosted Jev model.

Run the script again to create `output/enriched.csv`, `output/matches.csv`,
`output/disagreements.csv`, and `output/summary.json`. Those generated files are
excluded from Git.
