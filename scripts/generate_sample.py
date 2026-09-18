"""Generate reproducible synthetic fixtures; never use these as production benchmarks."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITION = (
    "The incident describes a current, unresolved problem that prevents customers from logging in "
    "or accessing the service. Resolved incidents, planned maintenance, internal employee access "
    "issues, and performance issues without loss of access do not match."
)
# Positive and negative cases include status reversals, implicit failures, and distractors.
CASES = [
    (
        "Login fails after release",
        "Since the release, customers receive invalid-session errors when signing in. The incident is still open; no workaround is available.",
        True,
    ),
    (
        "Customers locked out",
        "Account holders are stuck on the sign-in screen. Retrying does not help. Investigation is ongoing.",
        True,
    ),
    (
        "Authentication loop",
        "The login page redirects customers back to itself indefinitely. This is still happening and they cannot enter the application.",
        True,
    ),
    (
        "Dashboard unavailable",
        "Every customer request returns HTTP 503. Customers cannot open the dashboard. Recovery has not completed.",
        True,
    ),
    (
        "SSO regression",
        "Following yesterday's upgrade, customer SSO sessions are rejected. We have not restored access yet.",
        True,
    ),
    (
        "Invitation failure",
        "New customers accept their invitation but see access denied. They cannot enter their workspace. A fix is pending.",
        True,
    ),
    (
        "Expired certificates",
        "The expired gateway certificate prevents customers from opening the service. Certificate replacement is in progress and service is still inaccessible.",
        True,
    ),
    (
        "Partial recovery",
        "The rollback restored most accounts, but several customers still cannot log in. The incident remains open for those accounts.",
        True,
    ),
    (
        "Login restored",
        "Customers could not log in after the release. We rolled back and confirmed all customer accounts can now access the service. Incident resolved.",
        False,
    ),
    (
        "Duplicate payment",
        "A customer was charged twice. Their account and login work normally. The extra charge has not been refunded.",
        False,
    ),
    (
        "Slow reports",
        "Customers can sign in and use all functions, but reports take longer to load. We are investigating the latency.",
        False,
    ),
    (
        "Maintenance notice",
        "A planned maintenance window next week may temporarily prevent customer logins. There is no current outage.",
        False,
    ),
    (
        "Staff VPN",
        "Internal employees cannot connect to the corporate VPN. Customer access to the product is unaffected.",
        False,
    ),
    (
        "Help article request",
        "Please write documentation explaining what a customer should do if they cannot log in. No customer is currently reporting this problem.",
        False,
    ),
    (
        "Past outage question",
        "A customer asks why login failed last month. The outage was resolved that day and they can access the application now.",
        False,
    ),
    (
        "Feature request",
        "Customers request passkey login support. Existing password authentication works and they can access their accounts.",
        False,
    ),
    (
        "Billing resolved",
        "A duplicate charge was refunded. The customer confirmed everything is resolved and access has always worked.",
        False,
    ),
    (
        "Search error",
        "Customers can log in, browse records, and use their workspace. The advanced search filter shows an error. Basic search still works.",
        False,
    ),
    (
        "Deployment succeeded",
        "Post-deployment checks initially showed a login warning. Customer login probes now pass, and no users lost access. No open incident.",
        False,
    ),
    (
        "Admin permission request",
        "A customer can access their workspace and normal features, but requests an upgrade to the administrator role. No access failure is reported.",
        False,
    ),
]


def main():
    rows = []
    for variant in range(5):
        for case_number, (subject, message, expected) in enumerate(CASES):
            number = variant * len(CASES) + case_number + 1
            service = ["customer-portal", "analytics", "billing-console", "workspace", "reports"][
                variant
            ]
            rows.append(
                {
                    "incident_id": f"INC-{number:04d}",
                    "created_at": str(date(2026, 8, 20) + timedelta(days=number % 29)),
                    "service": service,
                    "region": ["eu-west", "us-east", "ap-south"][number % 3],
                    "version": f"2.{variant + 1}.{case_number % 4}",
                    "subject": subject,
                    "message": f"Service: {service}. {message}",
                    "expected_match": expected,
                }
            )
    random.Random(42).shuffle(rows)
    (ROOT / "data").mkdir(exist_ok=True)
    with (ROOT / "data/incidents.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=[k for k in rows[0] if k != "expected_match"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "expected_match"} for row in rows)
    with (ROOT / "data/incident_labels.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["incident_id", "expected_match"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)


if __name__ == "__main__":
    main()
