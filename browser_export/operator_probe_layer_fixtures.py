"""Stage already-exported trained-layer fixtures; never import Torch/load Cadence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stage(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    source_model = source / "linear-layer.onnx"
    target = destination / source_model.name
    if not target.exists():
        target.symlink_to(source_model.resolve())
    if target.resolve() != source_model.resolve():
        raise ValueError(
            "Staging model path does not match the requested trained layer"
        )
    cases = []
    for length in (63, 64, 65, 128):
        name = f"trained_delta_layer_{length}"
        fixture = source / f"layer-{length}.npz"
        with np.load(fixture, allow_pickle=False) as arrays:
            hidden = arrays["hidden"]
            expected = arrays["expected"]
            if hidden.dtype != np.float32 or expected.dtype != np.float32:
                raise ValueError("Require saved FP32 trained-layer fixtures")
            hidden_file, expected_file = f"{name}-hidden.f32", f"{name}-expected.f32"
            hidden.astype("<f4").tofile(destination / hidden_file)
            expected.astype("<f4").tofile(destination / expected_file)
            cases.append(
                {
                    "name": name,
                    "operator": "actual trained Cadence DeltaNet layer (FP32)",
                    "file": source_model.name,
                    "bytes": source_model.stat().st_size,
                    "sha256": hashlib.sha256(source_model.read_bytes()).hexdigest(),
                    "sourceFixtureSha256": hashlib.sha256(
                        fixture.read_bytes()
                    ).hexdigest(),
                    "feeds": {
                        "hidden": {
                            "type": "float32",
                            "dims": list(hidden.shape),
                            "dataFile": hidden_file,
                        }
                    },
                    "expected": None,
                    "expectedFiles": {"output": expected_file},
                }
            )
    (destination / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "modelBytes": source_model.stat().st_size,
                "staging": str(destination),
                "weightsCopied": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.out)
