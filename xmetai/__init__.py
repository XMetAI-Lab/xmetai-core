__version__ = "1.0.0"

from .config import LazyCall, LazyConfig
from .data import build_test_loader, build_train_loader, get_data_info
from .registries import DATASETS, METRICS, MODELS, import_builtin_registrations

import_builtin_registrations()

__all__ = [
    "__version__",
    "DATASETS",
    "METRICS",
    "MODELS",
    "build_test_loader",
    "build_train_loader",
    "get_data_info",
    "LazyCall",
    "LazyConfig",
]
