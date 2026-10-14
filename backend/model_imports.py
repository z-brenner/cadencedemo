"""Import only manifest-listed model source; never expose its directory as a path."""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

MODEL_SOURCES = {
    "cadence_runtime": "cadence_runtime.py",
    "cadence_kev": "cadence_kev/__init__.py",
    "cadence_kev.api": "cadence_kev/api.py",
    "cadence_kev.model": "cadence_kev/model.py",
    "cadence_kev.checkpoint": "cadence_kev/checkpoint.py",
}


def inside_package(source, package):
    path = Path(source)
    return path.absolute().is_relative_to(package) or path.resolve().is_relative_to(
        package
    )


class VerifiedSourceLoader(importlib.abc.Loader):
    def __init__(self, source, digest):
        self.source = source
        self.digest = digest

    def create_module(self, spec):
        return None  # Standard Python module initialization.

    def exec_module(self, module):
        # Compile the bytes we hash, rather than allowing unchecked cached bytecode
        # or a change between verification and a second source read.
        source = self.source.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.digest:
            raise ValueError("Model source changed before import: " + module.__name__)
        exec(compile(source, str(self.source), "exec"), module.__dict__)


class ModelSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package, manifest):
        self.package = package
        self.manifest = manifest

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODEL_SOURCES:
            if fullname.startswith("cadence_kev."):
                raise ImportError("Unreviewed model module: " + fullname)
            # An ancestor search root could expose this directory under another
            # package name. Check the normal dependency spec before any execution.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                origins = [spec.origin] if spec.has_location else []
                origins.extend(spec.submodule_search_locations or [])
                if any(inside_package(origin, self.package) for origin in origins):
                    raise ImportError(
                        "Dependency resolves inside model directory: " + fullname
                    )
            return None  # Dependencies retain their ordinary installed resolution.
        relative = MODEL_SOURCES[fullname]
        source = self.package / relative
        loader = VerifiedSourceLoader(source, self.manifest[relative])
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=loader,
            # Empty package search path: children resolve only through this finder.
            submodule_search_locations=[] if fullname == "cadence_kev" else None,
        )


def install_model_imports(package, manifest):
    package = package.resolve()
    for name, module in tuple(sys.modules.items()):
        if name in MODEL_SOURCES:
            continue
        source = getattr(module, "__file__", None)
        origins = [source] if source else []
        origins.extend(getattr(module, "__path__", []) or [])
        if any(inside_package(origin, package) for origin in origins):
            raise ValueError(
                "Unreviewed cached dependency: " + name + "; start a fresh process."
            )
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate == package or candidate.is_relative_to(package):
            raise ValueError(
                "Model directory is already in Python's import search path; "
                "remove it and start a fresh process."
            )
    # One model per worker. Keep the finder for predict-time relative imports;
    # repeated load attempts replace it without growing process-global state.
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, ModelSourceFinder)
    ]
    sys.meta_path.insert(0, ModelSourceFinder(package, manifest))
