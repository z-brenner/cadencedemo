"""Load the reviewed, immutable public Cadence runtime; never select latest."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from backend.app import MODEL_ID, REVISION


def load_model():
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    local = os.environ.get("CADENCE_MODEL_DIR")
    if local:
        path = Path(local).resolve()
    else:
        path = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                token=False,
                local_files_only=offline,
                allow_patterns=[
                    "*.py",
                    "cadence_kev/*.py",
                    "cadence_config.json",
                    "adapter.safetensors",
                    "head.safetensors",
                    "requirements.txt",
                ],
            )
        )
    # Local directories must match the same public revision, including reviewed code.
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    for name, expected in manifest.items():
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                "Local model differs from the pinned public revision: " + name
            )
    sys.path.insert(0, str(path))
    runtime = importlib.import_module("cadence_runtime")
    return runtime.Cadence.from_pretrained(str(path), local_files_only=offline)
