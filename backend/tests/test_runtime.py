import json
from pathlib import Path
import pytest
from backend import runtime


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_local_package_is_checked_before_import(monkeypatch, tmp_path, device):
    monkeypatch.setenv("CADENCE_DEVICE", device)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
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


def test_device_defaults_to_cuda_and_rejects_silent_auto(monkeypatch):
    monkeypatch.delenv("CADENCE_DEVICE", raising=False)
    assert runtime.selected_device() == "cuda"
    for value in ["auto", "mps", "CPU", ""]:
        monkeypatch.setenv("CADENCE_DEVICE", value)
        with pytest.raises(ValueError, match="cpu or cuda"):
            runtime.selected_device()


def test_cuda_selection_uses_the_pinned_loader(monkeypatch, tmp_path):
    import types

    calls = []
    native = object()

    class Cuda:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return native

    monkeypatch.setenv("CADENCE_DEVICE", "cuda")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path, raising=False)
    monkeypatch.setattr(
        runtime,
        "load_adapter_module",
        lambda name, source: types.SimpleNamespace(Cadence=Cuda),
    )
    assert runtime.load_model() is native
    assert calls == [(str(tmp_path), {"local_files_only": True})]


def test_cpu_selection_uses_only_the_reviewed_fp32_adapter(monkeypatch, tmp_path):
    import types

    calls = []
    native = object()

    def construct(package, *, profile, threads):
        calls.append((package, profile, threads))
        return native

    monkeypatch.setenv("CADENCE_DEVICE", "cpu")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path, raising=False)
    monkeypatch.setattr(
        runtime,
        "load_adapter_module",
        lambda name, source: types.SimpleNamespace(CpuCadence=construct),
    )
    assert runtime.load_model() is native
    assert calls == [(str(tmp_path), "fp32", 4)]


def test_cpu_needs_offline_cache_and_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("CADENCE_DEVICE", "cpu")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path, raising=False)
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE=1"):
        runtime.load_model()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def missing(name, source):
        raise ImportError("missing CPU adapter")

    monkeypatch.setattr(runtime, "load_adapter_module", missing)
    with pytest.raises(ImportError, match="missing CPU adapter"):
        runtime.load_model()


def test_cpu_adapter_source_must_match_its_provenance(monkeypatch):
    manifest = json.loads(
        Path(runtime.__file__).with_name("cpu-adapter-manifest.json").read_text()
    )
    assert manifest["license"] == "Apache-2.0" and manifest["profile"] == "fp32"
    runtime.verify_cpu_adapter()

    class WrongDigest:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(runtime.hashlib, "sha256", lambda _: WrongDigest())
    with pytest.raises(ValueError, match="reviewed coordinator artifact"):
        runtime.verify_cpu_adapter()


@pytest.mark.parametrize(
    "name,origin",
    [
        ("cadence_runtime", "cadence_runtime.py"),
        ("cadence_kev", "cadence_kev/__init__.py"),
        ("cadence_kev.api", "cadence_kev/api.py"),
        ("backend.cpu_runtime", "cpu_runtime.py"),
    ],
)
def test_cached_modules_cannot_bypass_verified_source(
    monkeypatch, tmp_path, name, origin
):
    import sys
    import types

    fake = types.ModuleType(name)
    fake.__file__ = str(tmp_path / "unreviewed" / origin)
    monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("CADENCE_DEVICE", "cpu")
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path)
    with pytest.raises(ValueError, match="cached module"):
        runtime.load_model()


def test_cpu_rejects_late_offline_setting(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setenv("CADENCE_DEVICE", "cpu")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path)
    with pytest.raises(ValueError, match="start a fresh process"):
        runtime.load_model()


def test_newly_imported_module_origin_is_checked(monkeypatch, tmp_path):
    import types

    fake = types.ModuleType("cadence_runtime")
    fake.__file__ = str(tmp_path / "wrong.py")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake)
    with pytest.raises(ValueError, match="verified source"):
        runtime.load_adapter_module("cadence_runtime", tmp_path / "cadence_runtime.py")


def test_cuda_custom_upstream_cache_is_forwarded(monkeypatch, tmp_path):
    import types

    calls = []

    class Cuda:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return object()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("CADENCE_UPSTREAM_CACHE_DIR", str(tmp_path / "upstream-cache"))
    monkeypatch.setattr(runtime, "verified_package", lambda: tmp_path)
    monkeypatch.setattr(
        runtime,
        "load_adapter_module",
        lambda name, source: types.SimpleNamespace(Cadence=Cuda),
    )
    runtime.load_model(device="cuda")
    assert calls == [
        (
            str(tmp_path),
            {
                "local_files_only": True,
                "upstream_cache_dir": str(tmp_path / "upstream-cache"),
            },
        )
    ]
