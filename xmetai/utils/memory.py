# -*- coding: utf-8 -*-
"""Lightweight CPU memory accounting helpers for training diagnostics."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import numpy as np
import torch


def bytes_to_mb(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return num_bytes / 1024.0 / 1024.0


def object_cpu_memory_bytes(obj: Any) -> dict[str, int]:
    """Return CPU tensor/ndarray bytes reachable from a Python object."""
    stats = {
        "tensor_cpu_bytes": 0,
        "pinned_tensor_cpu_bytes": 0,
        "numpy_bytes": 0,
    }
    seen_objects: set[int] = set()
    seen_tensor_storages: set[tuple[Any, ...]] = set()
    seen_numpy_buffers: set[tuple[int, int]] = set()
    stack: deque[Any] = deque([obj])

    while stack:
        item = stack.pop()
        item_id = id(item)
        if item_id in seen_objects:
            continue
        seen_objects.add(item_id)

        if isinstance(item, torch.Tensor):
            tensor_bytes = _cpu_tensor_storage_bytes(item, seen_tensor_storages)
            stats["tensor_cpu_bytes"] += tensor_bytes
            if tensor_bytes and _is_pinned_cpu_tensor(item):
                stats["pinned_tensor_cpu_bytes"] += tensor_bytes
            continue

        if isinstance(item, np.ndarray):
            numpy_bytes = _numpy_buffer_bytes(item, seen_numpy_buffers)
            stats["numpy_bytes"] += numpy_bytes
            continue

        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)

    stats["total_cpu_bytes"] = stats["tensor_cpu_bytes"] + stats["numpy_bytes"]
    return stats


def model_cpu_memory_bytes(model: torch.nn.Module) -> dict[str, int]:
    params = _cpu_tensor_iter_bytes(model.parameters())
    buffers = _cpu_tensor_iter_bytes(model.buffers())
    grads = _cpu_tensor_iter_bytes(p.grad for p in model.parameters() if p.grad is not None)
    return {
        "model_param_cpu_bytes": params,
        "model_buffer_cpu_bytes": buffers,
        "model_grad_cpu_bytes": grads,
        "model_cpu_bytes": params + buffers + grads,
    }


def optimizer_cpu_memory_bytes(optimizer: torch.optim.Optimizer) -> int:
    stats = object_cpu_memory_bytes(list(optimizer.state.values()))
    return stats["total_cpu_bytes"]


def process_memory_bytes(include_children: bool = True) -> dict[str, Optional[int]]:
    main = _linux_proc_status_bytes(os.getpid())
    child_rss = None
    if include_children:
        child_values = []
        for child_pid in _linux_descendant_pids(os.getpid()):
            child = _linux_proc_status_bytes(child_pid)
            if child["rss_bytes"] is not None:
                child_values.append(child["rss_bytes"])
        child_rss = sum(child_values) if child_values else 0

    total_rss = None
    if main["rss_bytes"] is not None:
        total_rss = main["rss_bytes"] + (child_rss or 0)

    return {
        "process_rss_bytes": main["rss_bytes"],
        "process_vms_bytes": main["vms_bytes"],
        "child_process_rss_bytes": child_rss,
        "total_process_rss_bytes": total_rss,
    }


def _cpu_tensor_iter_bytes(tensors: Iterable[Optional[torch.Tensor]]) -> int:
    seen_tensor_storages: set[tuple[Any, ...]] = set()
    total = 0
    for tensor in tensors:
        if tensor is not None:
            total += _cpu_tensor_storage_bytes(tensor, seen_tensor_storages)
    return total


def _cpu_tensor_storage_bytes(tensor: torch.Tensor, seen_storages: set[tuple[Any, ...]]) -> int:
    if tensor.device.type != "cpu":
        return 0

    try:
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())
        if key in seen_storages:
            return 0
        seen_storages.add(key)
        return int(storage.nbytes())
    except RuntimeError:
        return int(tensor.numel() * tensor.element_size())


def _numpy_buffer_bytes(array: np.ndarray, seen_buffers: set[tuple[int, int]]) -> int:
    data = array.__array_interface__.get("data")
    if data is None:
        return 0
    key = (int(data[0]), int(array.nbytes))
    if key in seen_buffers:
        return 0
    seen_buffers.add(key)
    return int(array.nbytes)


def _is_pinned_cpu_tensor(tensor: torch.Tensor) -> bool:
    try:
        return bool(tensor.is_pinned())
    except RuntimeError:
        return False


def _linux_proc_status_bytes(pid: int) -> dict[str, Optional[int]]:
    values: dict[str, Optional[int]] = {"rss_bytes": None, "vms_bytes": None}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    values["rss_bytes"] = _status_kb_line_to_bytes(line)
                elif line.startswith("VmSize:"):
                    values["vms_bytes"] = _status_kb_line_to_bytes(line)
    except OSError:
        pass
    return values


def _status_kb_line_to_bytes(line: str) -> int:
    parts = line.split()
    if len(parts) < 2:
        return 0
    return int(parts[1]) * 1024


def _linux_descendant_pids(pid: int) -> list[int]:
    descendants: list[int] = []
    pending = [pid]
    seen = {pid}
    while pending:
        current = pending.pop()
        for child_pid in _linux_child_pids(current):
            if child_pid in seen:
                continue
            seen.add(child_pid)
            descendants.append(child_pid)
            pending.append(child_pid)
    return descendants


def _linux_child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="utf-8") as handle:
            content = handle.read().strip()
    except OSError:
        return []
    if not content:
        return []
    return [int(item) for item in content.split()]
