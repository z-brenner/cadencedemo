import json
from pathlib import Path
import pytest
from backend import runtime


def test_local_package_is_checked_before_import(monkeypatch, tmp_path):
    monkeypatch.setenv("CADENCE_MODEL_DIR", str(tmp_path))
    manifest = json.loads(
        Path(runtime.__file__).with_name("model-manifest.json").read_text()
    )
    first = next(iter(manifest))
    (tmp_path / first).write_text("altered code")
    with pytest.raises(ValueError, match="differs from the pinned public revision"):
        runtime.load_model()


def test_manifest_pins_source_and_learned_weights():
    manifest = json.loads(
        Path(runtime.__file__).with_name("model-manifest.json").read_text()
    )
    assert set(manifest) == {
        "cadence_runtime.py",
        "cadence_config.json",
        "cadence_kev/__init__.py",
        "cadence_kev/api.py",
        "cadence_kev/model.py",
        "cadence_kev/checkpoint.py",
        "requirements.txt",
        "adapter.safetensors",
        "head.safetensors",
    }
    assert all(len(digest) == 64 for digest in manifest.values())
