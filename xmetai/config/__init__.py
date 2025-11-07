# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode, configurable
from .instantiate import instantiate
from .lazy import LazyCall, LazyConfig

__all__ = [
    "CfgNode",
    "configurable",
    "instantiate",
    "LazyCall",
    "LazyConfig",
]


from xmetai.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
