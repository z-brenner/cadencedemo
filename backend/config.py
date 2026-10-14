"""Immutable public model identity shared without importing the API or Torch."""

MODEL_ID = "foolish-bandit/cadence-0.8b"
REVISION = "17cdf36034677e406033fbf97637b4db1984ca70"
PROFILES = {"cuda": "cuda_bf16_fp32", "cpu": "cpu_fp32"}
OPTION_LIMITS = {"cuda": 32, "cpu": 16}
