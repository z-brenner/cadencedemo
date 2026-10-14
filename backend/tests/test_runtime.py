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


@pytest.fixture
def isolated_model_fixture(monkeypatch, tmp_path):
    """Known harmless source and manifest, with no weights or model dependencies."""
    import hashlib
    import importlib
    import sys

    package = tmp_path / "model"
    package.mkdir()
    (package / "cadence_kev").mkdir()
    sources = {
        "cadence_runtime.py": """from cadence_kev.api import clean_value
class Cadence:
    @staticmethod
    def from_pretrained(path, **kwargs):
        import cadence_fixture_dependency
        return (cadence_fixture_dependency.VALUE, clean_value())
""",
        "cadence_kev/__init__.py": "",
        "cadence_kev/api.py": "from .model import VALUE\ndef clean_value(): return VALUE\n",
        "cadence_kev/model.py": "VALUE = 'verified relative import'\n",
        "cadence_kev/checkpoint.py": "from .api import clean_value\n",
    }
    for name, source in sources.items():
        (package / name).write_text(source)
    manifest = {
        name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in sources.items()
    }
    fake_backend = tmp_path / "backend"
    fake_backend.mkdir()
    (fake_backend / "model-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "__file__", str(fake_backend / "runtime.py"))
    monkeypatch.setenv("CADENCE_MODEL_DIR", str(package))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "meta_path", sys.meta_path.copy())
    # Own fixture imports must not escape this test into another model's cache.
    names = [
        "cadence_runtime",
        "cadence_kev",
        "cadence_kev.api",
        "cadence_kev.model",
        "cadence_kev.checkpoint",
        "cadence_fixture_dependency",
    ]
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    dependency_dir = tmp_path / "installed"
    dependency_dir.mkdir()
    (dependency_dir / "cadence_fixture_dependency.py").write_text(
        "VALUE = 'installed dependency'\n"
    )
    monkeypatch.syspath_prepend(str(dependency_dir))
    importlib.invalidate_caches()
    yield package
    for name in names + ["cadence_kev.extra"]:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("extra", ["module", "package"])
def test_model_directory_extras_cannot_shadow_dependencies(
    isolated_model_fixture, extra
):
    package = isolated_model_fixture
    if extra == "module":
        (package / "cadence_fixture_dependency.py").write_text(
            "VALUE = 'unverified shadow'\n"
        )
    else:
        (package / "cadence_fixture_dependency").mkdir()
        (package / "cadence_fixture_dependency/__init__.py").write_text(
            "VALUE = 'unverified shadow'\n"
        )
    # Manifest verification succeeds, but extra code must not gain import precedence.
    assert runtime.verified_package() == package
    assert runtime.load_model(device="cuda") == (
        "installed dependency",
        "verified relative import",
    )


def test_clean_verified_model_imports_without_global_path_change(
    isolated_model_fixture,
):
    import sys

    before = sys.path.copy()
    assert runtime.load_model(device="cuda") == (
        "installed dependency",
        "verified relative import",
    )
    assert sys.path == before
    from cadence_kev.checkpoint import clean_value

    assert clean_value() == "verified relative import"


def test_unlisted_model_submodule_is_not_importable(isolated_model_fixture):
    import importlib

    package = isolated_model_fixture
    (package / "cadence_kev/extra.py").write_text("VALUE = 'unverified extra'\n")
    runtime.load_model(device="cuda")
    with pytest.raises(ImportError, match="Unreviewed model module"):
        importlib.import_module("cadence_kev.extra")


@pytest.mark.parametrize("name", ["cadence_runtime", "cadence_kev"])
def test_matching_file_name_alone_does_not_authenticate_cached_module(
    monkeypatch, tmp_path, name
):
    import sys
    import types

    module = types.ModuleType(name)
    module.__file__ = str(
        tmp_path
        / (
            "cadence_runtime.py"
            if name == "cadence_runtime"
            else "cadence_kev/__init__.py"
        )
    )
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="Unreviewed cached module"):
        runtime.check_cached_modules(tmp_path)


def test_unchecked_bytecode_cannot_replace_verified_source(isolated_model_fixture):
    import py_compile

    source = isolated_model_fixture / "cadence_kev/model.py"
    original = source.read_bytes()
    source.write_text("VALUE = 'unverified bytecode'\n")
    py_compile.compile(
        str(source), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH
    )
    source.write_bytes(original)
    assert runtime.load_model(device="cuda") == (
        "installed dependency",
        "verified relative import",
    )


def test_repeated_load_reuses_authenticated_modules_without_finder_growth(
    isolated_model_fixture,
):
    import sys

    first = runtime.load_model(device="cuda")
    finders = len(sys.meta_path)
    assert runtime.load_model(device="cuda") == first
    assert len(sys.meta_path) == finders


def test_model_directory_already_on_import_path_is_rejected(
    monkeypatch, isolated_model_fixture
):
    monkeypatch.syspath_prepend(str(isolated_model_fixture))
    with pytest.raises(ValueError, match="already in Python's import search path"):
        runtime.load_model(device="cuda")


def test_model_root_cannot_act_as_dependency_through_ancestor_path(
    monkeypatch, isolated_model_fixture
):
    package = isolated_model_fixture.with_name("cadence_fixture_dependency")
    isolated_model_fixture.rename(package)
    (package / "__init__.py").write_text("VALUE = 'unverified root alias'\n")
    monkeypatch.setenv("CADENCE_MODEL_DIR", str(package))
    monkeypatch.syspath_prepend(str(package.parent))
    with pytest.raises(ImportError, match="Dependency resolves inside model directory"):
        runtime.load_model(device="cuda")


def test_cached_dependency_from_model_directory_is_rejected(
    monkeypatch, isolated_model_fixture
):
    import sys
    import types

    fake = types.ModuleType("cadence_fixture_dependency")
    fake.__file__ = str(isolated_model_fixture / "cadence_fixture_dependency.py")
    fake.VALUE = "unreviewed cached dependency"
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    with pytest.raises(ValueError, match="Unreviewed cached dependency"):
        runtime.load_model(device="cuda")


def test_source_change_between_verification_and_import_is_rejected(
    isolated_model_fixture,
):
    import importlib
    from backend.model_imports import install_model_imports

    package = runtime.verified_package()
    manifest = json.loads(
        Path(runtime.__file__).with_name("model-manifest.json").read_text()
    )
    install_model_imports(package, manifest)
    (package / "cadence_runtime.py").write_text("VALUE = 'changed source'\n")
    with pytest.raises(ValueError, match="Model source changed before import"):
        importlib.import_module("cadence_runtime")
