import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from xmetai.config import LazyConfig
from xmetai.core.base_predictor import BasePredictor
from xmetai.core.checkpoint import FSDPCheckpointer


def test_infer_evaluator_loop_does_not_export_detached_config(tmp_path):
    source = (Path(__file__).resolve().parents[2] / "configs" / "iwc_infer.py").read_text()
    loop = ast.get_source_segment(source, next(node for node in ast.parse(source).body if isinstance(node, ast.For)))
    assert loop is not None
    # Exercise the real infer configuration loop without requiring weather data.
    config_path = tmp_path / "infer.py"
    config_path.write_text("""from omegaconf import OmegaConf
dataloader = OmegaConf.create({"evaluator": [
    {"test_names": ["z500", "tp"], "test_times": [6]},
    {"test_names": "${..0.test_names}", "test_times": "${..0.test_times}"}
]})
eval_times = [6, 12]
test_names = ["z500", "tp"]
""" + loop)
    cfg = LazyConfig.load(str(config_path))
    assert set(cfg.keys()) == {"dataloader"}
    save_path = tmp_path / "saved.yaml"
    LazyConfig.save(cfg, str(save_path))
    restored = LazyConfig.load(str(save_path))
    assert list(restored.dataloader.evaluator[1].test_names) == ["z500", "tp"]
    assert list(restored.dataloader.evaluator[1].test_times) == [6, 12]


def test_summary_buffers_load_placeholders_then_normal_checkpoint(tmp_path):
    buffers = {"const": np.ones((3, 20, 30), np.float32), "channel_weights": np.array([1, 2], np.float32).reshape((2, 1, 1))}
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": {f"module.{name}": torch.from_numpy(value) for name, value in buffers.items()}}, checkpoint)
    buffers["channel_weights"] = np.array([3, 7], np.float32).reshape((2, 1, 1))
    cfg = OmegaConf.create({"buffers": buffers, "model": {"buffers": "${buffers}"}}, flags={"allow_objects": True})
    path = tmp_path / "config.yaml"
    LazyConfig.save(cfg, str(path), array_threshold=1024)
    with patch("torch.load", side_effect=AssertionError("YAML loading must not read checkpoints")):
        restored = LazyConfig.load(str(path))
    for name, expected in buffers.items():
        assert restored.buffers[name].shape == expected.shape
        assert restored.buffers[name].dtype == expected.dtype
    np.testing.assert_array_equal(restored.model.buffers.channel_weights, buffers["channel_weights"])
    assert not np.any(restored.buffers.const)
    model = BasePredictor(in_chans=2, out_chans=2, in_frames=1, out_frames=1, buffers=restored.model.buffers, device="cpu", precision="fp32")
    FSDPCheckpointer(model).load(str(checkpoint))
    torch.testing.assert_close(model.const, torch.from_numpy(buffers["const"]))
