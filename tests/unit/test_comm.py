"""Object collectives must use a CPU-capable group, independently of CUDA collectives."""

from types import SimpleNamespace

import pytest

from xmetai.utils import comm


@pytest.fixture(autouse=True)
def clear_cached_gloo_group():
    comm._get_global_gloo_group.cache_clear()
    yield
    comm._get_global_gloo_group.cache_clear()


@pytest.mark.parametrize("backend", ["nccl", "gloo"])
def test_default_object_collectives_share_cpu_group(monkeypatch, backend):
    cpu_group = object()
    created = []
    calls = []
    monkeypatch.setattr(comm.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(comm.dist, "group", SimpleNamespace(WORLD=cpu_group))
    monkeypatch.setattr(comm, "get_world_size", lambda: 2)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(comm.dist, "get_rank", lambda group=None: 0)

    def new_group(backend):
        assert backend == "gloo"
        created.append(cpu_group)
        return cpu_group

    def collect(output, data, group):
        assert group is cpu_group
        calls.append(data)
        output[:] = [data, {"rank": 1}]

    def gather(data, output, dst, group):
        assert dst == 0
        collect(output, data, group)

    monkeypatch.setattr(comm.dist, "new_group", new_group)
    monkeypatch.setattr(comm.dist, "all_gather_object", collect)
    monkeypatch.setattr(comm.dist, "gather_object", gather)
    value = {"rank": 0}
    assert comm.all_gather(value) == [value, {"rank": 1}]
    assert comm.gather(value) == [value, {"rank": 1}]
    assert calls == [value, value]
    assert len(created) == (1 if backend == "nccl" else 0)


@pytest.mark.parametrize("operation", ["all_gather", "gather"])
def test_explicit_group_is_respected(monkeypatch, operation):
    explicit_group = object()
    monkeypatch.setattr(comm, "get_world_size", lambda: 2)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(comm.dist, "get_rank", lambda group=None: 0)

    def forbidden():
        raise AssertionError("An explicit group must not create a default group")

    def collect(output, data, group):
        assert group is explicit_group
        output[:] = [data, data]

    def gather(data, output, dst, group):
        collect(output, data, group)

    monkeypatch.setattr(comm, "_get_global_gloo_group", forbidden)
    monkeypatch.setattr(comm.dist, "all_gather_object", collect)
    monkeypatch.setattr(comm.dist, "gather_object", gather)
    assert getattr(comm, operation)("value", group=explicit_group) == ["value", "value"]


@pytest.mark.parametrize("operation", ["all_gather", "gather"])
def test_single_process_does_not_create_group(monkeypatch, operation):
    monkeypatch.setattr(comm, "get_world_size", lambda: 1)

    def forbidden():
        raise AssertionError("Single-process use must not initialize a communication group")

    monkeypatch.setattr(comm, "_get_global_gloo_group", forbidden)
    assert getattr(comm, operation)({"value": 1}) == [{"value": 1}]


def test_gather_non_destination_rank_returns_empty_list(monkeypatch):
    group = object()
    monkeypatch.setattr(comm, "get_world_size", lambda: 2)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(comm.dist, "get_rank", lambda group=None: 1)
    calls = []

    def gather(data, output, dst, group):
        assert output is None
        calls.append((data, dst, group))

    monkeypatch.setattr(comm.dist, "gather_object", gather)
    assert comm.gather("value", dst=0, group=group) == []
    assert calls == [("value", 0, group)]
