"""Exercise the real Dynamo exporter, including the GDN2 custom-op boundary."""

import numpy as np
import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from xmetai.cli.train import do_export  # noqa: E402
from xmetai.layers.gated_deltanet2 import GatedDeltaNet2Mixer  # noqa: E402
from xmetai.layers.mamba_vision import Mamba3SisoOnnxFn, _selective_scan_onnx  # noqa: E402


class ExportModel(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, data):
        raise AssertionError("The training forward must not run during export")

    def onnx_export(self, data, scale):
        return self.linear(data) * scale


@pytest.mark.parametrize("example_batch", [1, 2])
def test_export_selected_forward_and_runtime(tmp_path, example_batch):
    ort = pytest.importorskip("onnxruntime")
    model = ExportModel().eval()
    data = torch.randn(example_batch, 4)
    scale = np.array([2.0], dtype=np.float32)
    path = tmp_path / "model.onnx"
    expected = model.onnx_export(data, torch.from_numpy(scale)).detach().numpy()
    do_export(model, {"input": data, "scale": scale}, save_name=str(path), device="cpu")
    onnx.checker.check_model(str(path))
    assert not path.with_suffix(".onnx.data").exists()
    assert not path.with_suffix(".data").exists()
    assert all(item.data_location != onnx.TensorProto.EXTERNAL for item in onnx.load(path).graph.initializer)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["pred"], {"input": data.numpy(), "scale": scale})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert isinstance(session.get_inputs()[0].shape[0], str)
    for batch_size in (1, 3, 4):
        values = torch.randn(batch_size, 4)
        expected = model.onnx_export(values, torch.from_numpy(scale)).detach().numpy()
        actual = session.run(["pred"], {"input": values.numpy(), "scale": scale})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class TimestampExportModel(nn.Module):
    dtype = torch.float32

    def onnx_export(self, data, step, hour, doy, perturb_std_scale):
        return data * perturb_std_scale + (step + hour + doy)[:, None]


def test_dynamic_batch_timestamps_keep_group_scales_static(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = TimestampExportModel().eval()
    feed = {
        "input": np.ones((1, 3), np.float32),
        "step": np.zeros(1, np.float32),
        "hour": np.zeros(1, np.float32),
        "doy": np.zeros(1, np.float32),
        "perturb_std_scale": np.array([1, 2, 3], np.float32),
    }
    path = tmp_path / "timestamps.onnx"
    do_export(model, feed, save_name=str(path), device="cpu")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[-1].shape == [3]
    for batch_size in (1, 2, 4):
        feed["input"] = np.arange(batch_size * 3, dtype=np.float32).reshape(batch_size, 3)
        for name in ("step", "hour", "doy"):
            feed[name] = np.arange(batch_size, dtype=np.float32)
        expected = model.onnx_export(*(torch.from_numpy(v) for v in feed.values())).numpy()
        actual = session.run(["pred"], feed)[0]
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gdn2_dynamo_preserves_custom_operator(tmp_path, device, dynamo):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("fla")
    model = GatedDeltaNet2Mixer(hidden_size=32, num_heads=2).eval().to(device)
    path = tmp_path / "gdn2.onnx"
    torch.onnx.export(
        model, (torch.randn(1, 8, 32, device=device),), path, dynamo=dynamo, opset_version=18, custom_opsets=None if dynamo else {"xmetai_plugins": 1}
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    nodes = [node for node in graph.graph.node if node.domain == "xmetai_plugins"]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.op_type == "GatedDeltaNet2Fn"
    assert len(node.input) == 7
    assert {entry.domain: entry.version for entry in graph.opset_import}["xmetai_plugins"] == 1
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    assert attrs == {"scale": 0.25, "use_qk_l2norm_in_kernel": 1, "output_final_state": 0, "use_gate_in_kernel": 0, "has_cu_seqlens": 0, "state_v_first": 0}
    # The old autograd.Function.forward returned empty_like(v); a naive Dynamo
    # migration silently prunes q/k/decay/beta/w and their trained parameters.
    producers = {name: item for item in graph.graph.node for name in item.output}
    graph_inputs = {item.name for item in graph.graph.input}
    for name in node.input[:6]:
        pending = [name]
        ancestors = set()
        while pending:
            current = pending.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            if current in producers:
                pending.extend(producers[current].input)
        assert ancestors & graph_inputs, name
        assert any(producers[item].op_type in ("MatMul", "Gemm") for item in ancestors if item in producers), name


def test_external_data_name_and_reexport(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = np.arange(4096, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "forecast.v2.onnx"
    sizes = []
    # Exercise the protobuf size boundary through the real export entry point.
    monkeypatch.setattr(onnx.checker, "MAXIMUM_PROTOBUF", 1)
    for offset in (0, 1):
        model = ExportModel().eval()
        model.linear = nn.Linear(64, 64, bias=False)
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(weights + offset))
        do_export(model, {"input": torch.ones(1, 64), "scale": np.ones(1, dtype=np.float32)}, save_name=str(path), device="cpu")
        stored = onnx.load(path, load_external_data=False)
        external = [item for item in stored.graph.initializer if item.data_location == onnx.TensorProto.EXTERNAL]
        assert len(external) == 1
        assert {entry.key: entry.value for entry in external[0].external_data}["location"] == "forecast.v2.data"
        sizes.append(path.with_suffix(".data").stat().st_size)
        restored = onnx.load(path)
        tensor = next(item for item in restored.graph.initializer if item.name == external[0].name)
        # The optimizer may transpose MatMul weights; values must survive exactly.
        np.testing.assert_array_equal(np.sort(onnx.numpy_helper.to_array(tensor).ravel()), (weights + offset).ravel())
    assert sizes[0] == sizes[1] == weights.nbytes
    assert not path.with_suffix(".onnx.data").exists()


class Mamba3OperatorExport(nn.Module):
    def forward(self, x):
        return Mamba3SisoOnnxFn.apply(x, x, x, x, x, x, x, x, x, x, x, 64, True, True)


class MambaOperatorExport(nn.Module):
    def forward(self, x):
        return _selective_scan_onnx(x, x, x, x, x, delta_softplus=True)


@pytest.mark.parametrize("model,op_type,input_count", [(MambaOperatorExport, "SelectiveScanFn", 8), (Mamba3OperatorExport, "Mamba3SisoFn", 11)])
def test_mamba_exporters_preserve_same_operator_contract(tmp_path, model, op_type, input_count):
    contracts = []
    for dynamo in (False, True):
        path = tmp_path / f"{op_type}_{dynamo}.onnx"
        torch.onnx.export(
            model().eval(), (torch.randn(1, 2, 4),), path, dynamo=dynamo, opset_version=18, custom_opsets=None if dynamo else {"xmetai_plugins": 1}
        )
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        (node,) = [n for n in graph.graph.node if n.domain == "xmetai_plugins"]
        assert node.op_type == op_type
        assert len(node.input) == input_count
        assert {o.domain: o.version for o in graph.opset_import}["xmetai_plugins"] == 1
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        contracts.append(attrs)
        # Every required operand remains connected to the real graph input.
        assert node.input[0] == graph.graph.input[0].name
    assert contracts[0] == contracts[1]
