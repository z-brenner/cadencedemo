"""Size exports without Torch or tensor deserialization.

Read base headers only and hash public-package bytes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

WIDTHS = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4}
MAX_HEADER = 16 * 1024 * 1024


def inspect_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header.")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER:
            raise ValueError("Safetensors header exceeds metadata-only budget.")
        header = json.loads(stream.read(size))
    payload = path.stat().st_size - 8 - size
    result = {
        "textParameters": 0,
        "visionParameters": 0,
        "otherParameters": 0,
        "tensorBytes": 0,
    }
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape = tensor["shape"]
        if any(not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError("Invalid tensor shape.")
        count = math.prod(shape)
        start, end = tensor["data_offsets"]
        if (
            not 0 <= start <= end <= payload
            or end - start != count * WIDTHS[tensor["dtype"]]
        ):
            raise ValueError("Tensor offsets do not match the saved payload.")
        group = (
            "text"
            if ".language_model." in name
            else "vision"
            if ".visual." in name
            else "other"
        )
        result[group + "Parameters"] += count
        result["tensorBytes"] += end - start
    return result


def preflight(package, base, staging):
    manifest = json.loads(
        Path(__file__).parents[1].joinpath("backend/model-manifest.json").read_text()
    )
    for name, expected in manifest.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Package does not match the pinned demo manifest: " + name)
    config = json.loads((package / "cadence_config.json").read_text())
    headers = [inspect_header(path) for path in base.glob("*.safetensors")]
    if not headers:
        raise ValueError("No cached base safetensors headers found.")
    text = sum(item["textParameters"] for item in headers)
    learned = config["learned_parameters"]
    if text + learned != config["model_parameters"]:
        raise ValueError(
            "Text backbone and learned geometry do not match the frozen model."
        )
    tokenizer = sum(
        (base / name).stat().st_size
        for name in ["tokenizer.json", "tokenizer_config.json"]
    )
    free = shutil.disk_usage(staging).free
    fp32 = (text + learned) * 4
    mixed = text * 2 + learned * 4
    return {
        "kind": "metadata-only; no model conversion or inference",
        "model": "foolish-bandit/cadence-0.8b",
        "revision": "17cdf36034677e406033fbf97637b4db1984ca70",
        "base": config["base_model"],
        "parameters": {
            "textBackbone": text,
            "learnedAdapterAndHead": learned,
            "total": text + learned,
        },
        "weightStorageEstimates": {
            "fp32Bytes": fp32,
            "mixed16BitBackboneFp32LearnedBytes": mixed,
        },
        "cachedTokenizerBytes": tokenizer,
        "stagingFreeBytes": free,
        "fp32WeightsFitBeforeGraphOrWorkingCopies": free >= fp32,
        "actualBrowserArtifactBytes": None,
        "browserParityVerified": False,
        "notes": [
            "Estimates preserve the text backbone and unmerged learned parameter count; graph/lowering assets are additional.",
            "Mixed storage is a size estimate, not a qualified browser precision profile.",
            "Base tensor payloads were not read; model revision and numerical conversion still require coordinated verification.",
            "Free tmpfs space shares RAM and is not permission to stage a full export there.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            preflight(args.package, args.base_snapshot, args.staging_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
