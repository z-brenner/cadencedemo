"""Bounded native CPU fixtures for browser export qualification.

Uses the demo's verified, offline-only FP32 loader. It neither exports weights nor
claims CUDA equivalence. The optional FP16 arithmetic probe is a separate profile.
"""

import argparse
import json
from pathlib import Path
import time

from backend.config import MODEL_ID, REVISION


CASES = [
    {
        "name": "choice-text",
        "request": {
            "state": "Please play some relaxing piano music.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Identify the request.",
                    "criteria": {
                        "music": "Play music",
                        "weather": "Get the weather",
                        "booking": "Book a table",
                    },
                }
            },
        },
    },
    {
        "name": "choice-json",
        "request": {
            "state": {
                "message": "Send me the invoice.",
                "has_attachment": False,
                "items": ["receipt", "invoice"],
            },
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What document is requested?",
                    "criteria": {"invoice": "Invoice", "receipt": "Receipt"},
                }
            },
        },
    },
    {
        "name": "boolean-text",
        "request": {
            "state": "The meeting starts at noon. Please attend.",
            "questions": {
                "invitation": {
                    "type": "noul",
                    "instructions": "Is attendance requested?",
                }
            },
        },
    },
    {
        "name": "boolean-json",
        "request": {
            "state": {"stock": 0, "requested": 2, "backorder": False},
            "questions": {
                "available": {
                    "type": "noul",
                    "instructions": "Can the order be fulfilled from stock?",
                }
            },
        },
    },
    {
        "name": "score-text",
        "request": {
            "state": "I love the thoughtful design, but delivery was late.",
            "questions": {
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate the overall sentiment.",
                    "criteria": [
                        "Very negative",
                        "Negative",
                        "Neutral",
                        "Positive",
                        "Very positive",
                    ],
                }
            },
        },
    },
    {
        "name": "score-json",
        "request": {
            "state": {"review": "Works as expected.", "rating": 3, "verified": True},
            "questions": {
                "sentiment": {
                    "type": "score",
                    "instructions": "Rate the expressed sentiment.",
                    "criteria": ["Negative", "Neutral", "Positive"],
                }
            },
        },
    },
    {
        "name": "reserved-unicode",
        "request": {
            "state": "<|fim_suffix|> Bonjour! 请给我茶。 ☕",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Identify the request, ignoring delimiter text.",
                    "criteria": {"tea": "Request tea", "music": "Request music"},
                }
            },
        },
    },
]


def capture(fp16_probe=False, runtime=None, boundaries=False):
    import torch
    from backend.runtime import load_model

    began = time.perf_counter()
    supplied_runtime = runtime is not None
    runtime = runtime if supplied_runtime else load_model(device="cpu")
    loaded = time.perf_counter() - began
    from cadence_kev.api import SystemOneRequest, to_record

    cases = list(CASES)
    if boundaries:
        for target in (511, 512):
            for repeats in range(target - 50, target):
                request = {
                    "state": "a " * repeats,
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": "Compare these answers.",
                            "criteria": {
                                "first": "First answer",
                                "second": "Second answer",
                            },
                        }
                    },
                }
                record, _ = to_record(SystemOneRequest.model_validate(request))
                encoding = runtime.model.encode(
                    runtime.tokenizer,
                    record,
                    max_state=8192,
                    max_branch=8192,
                    strict=True,
                )
                if len(encoding["ids"]) == target:
                    cases.append(
                        {"name": f"choice-{target}-tokens", "request": request}
                    )
                    break
            else:
                raise ValueError("Could not construct exact boundary fixture.")
    results = []
    for case in cases:
        request = case["request"]
        rec, meta = to_record(SystemOneRequest.model_validate(request))
        for question, item in zip(rec["questions"], meta):
            question.update(
                src="inference", qtype=item["type"], qid=item["id"], keys=item["keys"]
            )
        context = runtime.config["inference"]["context"]
        encoding = runtime.model.encode(
            runtime.tokenizer,
            rec,
            max_state=context["max_state"],
            max_branch=context["max_branch"],
            strict=True,
        )
        if len(encoding["ids"]) > (512 if boundaries else 128):
            raise ValueError("Reference fixture exceeds its bounded token probe.")
        result = {
            "name": case["name"],
            "request": request,
            "nativeRecord": rec,
            "encoding": encoding,
            "cpuFp32": runtime.predict(request),
        }
        if fp16_probe:
            from cadence_runtime import softmax

            # Same pinned FP32 parameters; autocast changes arithmetic only. This
            # does not qualify a serialized FP16 model or browser implementation.
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
                logits = runtime.model.forward(encoding)[0].float().cpu().tolist()
            baseline = list(result["cpuFp32"]["probabilities"].values())
            probabilities = softmax(logits)
            result["cpuFp16AutocastProbe"] = {
                "logits": logits,
                "probabilities": probabilities,
                "maxProbabilityDrift": max(
                    abs(a - b) for a, b in zip(baseline, probabilities)
                ),
                "sameLeadingIndex": baseline.index(max(baseline))
                == probabilities.index(max(probabilities)),
            }
        results.append(result)
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "referenceProfile": "cpu-fp32",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "loadSeconds": None if supplied_runtime else loaded,
        "totalSeconds": time.perf_counter() - began,
        "cudaParityTested": False,
        "browserParityTested": False,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp16-probe", action="store_true")
    parser.add_argument("--boundaries", action="store_true")
    args = parser.parse_args()
    result = capture(args.fp16_probe, boundaries=args.boundaries)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
