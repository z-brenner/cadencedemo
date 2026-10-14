"""Opt-in, real HTTP inference smoke. Coordinate GPU use before running."""

import argparse
import json
import math
from pathlib import Path
import httpx
from backend.app import MODEL_ID, REVISION

REQUESTS = [
    {
        "state": "Can you play some jazz for dinner?",
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose the intent that best matches this request.",
                "criteria": {
                    "play_music": "Play music or a specific genre.",
                    "book_restaurant": "Find or book a restaurant.",
                    "get_weather": "Check the weather forecast.",
                },
            }
        },
    },
    {
        "state": {
            "message": "Congratulations! You won a free holiday. Enter your bank details to claim it."
        },
        "questions": {
            "decision": {
                "type": "noul",
                "instructions": "Does this message look like spam?",
                "criteria": {
                    "false": "An ordinary message without spam signals.",
                    "true": "An unsolicited or deceptive promotional message.",
                },
            }
        },
    },
    {
        "state": "Customer: I was charged twice. Reply: I will check both transactions and refund the duplicate charge.",
        "questions": {
            "decision": {
                "type": "score",
                "instructions": "Rate the helpfulness of this reply using the ordered levels.",
                "criteria": [
                    "Not helpful.",
                    "Slightly helpful.",
                    "Moderately helpful.",
                    "Helpful.",
                    "Very helpful.",
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
        assert health.json()["inference"] == "real"
        for request in REQUESTS:
            response = client.post("/api/predict", json=request)
            response.raise_for_status()
            result = response.json()
            assert result["revision"] == REVISION and result["model"] == MODEL_ID
            assert result["kind"] == request["questions"]["decision"]["type"]
            assert result["truncated"] is False
            assert math.isclose(sum(result["probabilities"].values()), 1, abs_tol=1e-5)
            assert all(math.isfinite(v) for v in result["probabilities"].values())
            rows.append({"request": request, "result": result})
    Path(args.output).write_text(
        json.dumps({"inference": "real", "revision": REVISION, "rows": rows}, indent=2)
        + "\n"
    )
    print("Three real native query types passed; evidence:", args.output)


if __name__ == "__main__":
    main()
