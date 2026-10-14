"""Explicit runtime selection with pinned package and CPU adapter verification."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from backend.config import MODEL_ID, REVISION


def selected_device():
    device = os.environ.get("CADENCE_DEVICE", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError(
            "CADENCE_DEVICE must be cpu or cuda; automatic fallback is disabled."
        )
    return device


def verified_package():
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
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    for name, expected in manifest.items():
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                "Local model differs from the pinned public revision: " + name
            )
    return path


def verify_cpu_adapter():
    manifest = json.loads(
        Path(__file__).with_name("cpu-adapter-manifest.json").read_text()
    )
    digest = hashlib.sha256(
        Path(__file__).with_name(manifest["file"]).read_bytes()
    ).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("CPU adapter differs from the reviewed coordinator artifact.")


def check_cached_modules(package):
    expected = {
        "cadence_runtime": package / "cadence_runtime.py",
        "cadence_kev": package / "cadence_kev" / "__init__.py",
        "cadence_kev.api": package / "cadence_kev" / "api.py",
        "cadence_kev.model": package / "cadence_kev" / "model.py",
        "cadence_kev.checkpoint": package / "cadence_kev" / "checkpoint.py",
        "backend.cpu_runtime": Path(__file__).with_name("cpu_runtime.py"),
    }
    for name, module in tuple(sys.modules.items()):
        if name in expected or name.startswith("cadence_kev."):
            source = getattr(module, "__file__", None)
            if (
                name not in expected
                or source is None
                or Path(source).resolve() != expected[name].resolve()
            ):
                raise ValueError(
                    "Unreviewed cached module: " + name + "; start a fresh process."
                )


def load_adapter_module(name, source):
    module = importlib.import_module(name)
    actual = getattr(module, "__file__", None)
    if actual is None or Path(actual).resolve() != source.resolve():
        raise ValueError(
            "Runtime module did not load from its verified source: " + name
        )
    return module


def load_model(*, device=None):
    device = selected_device() if device is None else device
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    if device == "cpu":
        if not offline:
            raise ValueError(
                "CPU FP32 requires pre-cached pinned dependencies and HF_HUB_OFFLINE=1."
            )
        if sys.flags.optimize:
            raise ValueError(
                "Run CPU FP32 without Python -O so the reviewed adapter's reconstruction checks stay active."
            )
        from huggingface_hub import constants

        if not constants.HF_HUB_OFFLINE:
            raise ValueError(
                "Offline mode was set after Hub import; start a fresh process with HF_HUB_OFFLINE=1."
            )
        verify_cpu_adapter()
    path = verified_package()
    check_cached_modules(path)
    sys.path.insert(0, str(path))
    if device == "cpu":
        adapter = load_adapter_module(
            "backend.cpu_runtime", Path(__file__).with_name("cpu_runtime.py")
        )
        return adapter.CpuCadence(str(path), profile="fp32", threads=4)
    runtime = load_adapter_module("cadence_runtime", path / "cadence_runtime.py")
    kwargs = {"local_files_only": offline}
    upstream_cache = os.environ.get("CADENCE_UPSTREAM_CACHE_DIR")
    if upstream_cache:
        kwargs["upstream_cache_dir"] = upstream_cache
    return runtime.Cadence.from_pretrained(str(path), **kwargs)
