from importlib import import_module
from pkgutil import walk_packages

from xmetai.utils import Registry

MODELS: Registry = Registry("models")
METRICS: Registry = Registry("metrics")
DATASETS: Registry = Registry("datasets")
CONDITIONERS: Registry = Registry("conditioners")

__all__ = ["CONDITIONERS", "DATASETS", "METRICS", "MODELS", "import_builtin_registrations"]

_BUILTIN_REGISTRATION_PACKAGES = (
    "xmetai.metrics",
    "xmetai.data",
    "xmetai.models",
)
_BUILTINS_IMPORTED = False


def _import_package_modules(package_name: str) -> None:
    package = import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    for module_info in walk_packages(package_path, prefix=f"{package.__name__}."):
        if not module_info.ispkg:
            import_module(module_info.name)


def import_builtin_registrations() -> None:
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    for package_name in _BUILTIN_REGISTRATION_PACKAGES:
        _import_package_modules(package_name)
    _BUILTINS_IMPORTED = True
