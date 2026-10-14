"""Opt-in, real HTTP inference smoke. Coordinate device/cache use before running."""

import argparse
import json
import math
from pathlib import Path
import httpx
from backend.app import MODEL_ID, REVISION

REQUESTS = [
    {
        "state": "Fictional document excerpt: Each party must keep the other party’s confidential "
        "information private and use it only to evaluate the proposed collaboration.",
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose the document category that best matches this "
                "excerpt.",
                "criteria": {
                    "confidentiality_agreement": "An agreement about keeping "
                    "shared information "
                    "confidential.",
                    "services_agreement": "An agreement describing services, "
                    "deliverables and payment.",
                    "meeting_notes": "Notes recording a meeting discussion "
                    "and action items.",
                },
            }
        },
    },
    {
        "state": "Fictional clause: Either party may end this agreement by giving the other party at "
        "least 30 days of written notice.",
        "questions": {
            "decision": {
                "type": "noul",
                "instructions": "Does the provided clause explicitly require at least "
                "30 days of written notice to end the agreement? "
                "Check only the supplied words.",
                "criteria": {
                    "false": "No: the clause does not explicitly state that "
                    "requirement.",
                    "true": "Yes: the clause explicitly states at least 30 "
                    "days of written notice.",
                },
            }
        },
    },
    {
        "state": "Fictional work item: A draft vendor agreement is waiting for review. The team’s "
        "internal target is tomorrow. A reviewer must check the payment terms before the team "
        "can finish its review.",
        "questions": {
            "decision": {
                "type": "score",
                "instructions": "Rate document-review priority using the supplied "
                "internal workflow levels. Use only the stated timing "
                "and work status; do not infer a legal deadline or "
                "legal risk.",
                "criteria": [
                    "Archived: review is complete and no action remains.",
                    "Low: review is needed but no internal target date is stated.",
                    "Routine: review is needed with an internal target more "
                    "than a week away.",
                    "Soon: review is needed with an internal target in "
                    "two to seven days.",
                    "First: review is still needed and the internal target "
                    "is today or tomorrow.",
                ],
            }
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="docs/real-inference-smoke.json")
    args = parser.parse_args()
    rows = []
    with httpx.Client(base_url=args.url, timeout=300) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        runtime = health.json()
        assert runtime["inference"] == "real"
        assert runtime["device"] in ("cuda", "cpu")
        for request in REQUESTS:
            response = client.post("/api/predict", json=request)
            response.raise_for_status()
            result = response.json()
            assert result["revision"] == REVISION and result["model"] == MODEL_ID
            assert result["kind"] == request["questions"]["decision"]["type"]
            assert result["truncated"] is False
            assert (
                result["device"] == runtime["device"]
                and result["profile"] == runtime["profile"]
            )
            assert math.isclose(sum(result["probabilities"].values()), 1, abs_tol=1e-5)
            assert all(math.isfinite(v) for v in result["probabilities"].values())
            rows.append({"request": request, "result": result})
    Path(args.output).write_text(
        json.dumps(
            {
                "inference": "real",
                "revision": REVISION,
                "device": runtime["device"],
                "profile": runtime["profile"],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("Three real native query types passed; evidence:", args.output)


if __name__ == "__main__":
    main()
