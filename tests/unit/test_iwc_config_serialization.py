from pathlib import Path

from xmetai.config import LazyConfig


def test_infer_evaluator_loop_does_not_export_detached_config(tmp_path):
    source = (Path(__file__).resolve().parents[2] / "configs" / "iwc_infer.py").read_text()
    loop = source[source.index("for _evaluator in dataloader.evaluator:") :]
    # Exercise the real infer configuration loop without requiring weather data.
    config_path = tmp_path / "infer.py"
    config_path.write_text("""from omegaconf import OmegaConf
dataloader = OmegaConf.create({"evaluator": [
    {"test_names": ["z500", "tp"], "test_times": [6]},
    {"test_names": "${..0.test_names}", "test_times": "${..0.test_times}"}
]})
eval_times = [6, 12]
""" + loop)
    cfg = LazyConfig.load(str(config_path))
    assert set(cfg.keys()) == {"dataloader"}
    save_path = tmp_path / "saved.yaml"
    LazyConfig.save(cfg, str(save_path))
    restored = LazyConfig.load(str(save_path))
    assert list(restored.dataloader.evaluator[1].test_names) == ["z500", "tp"]
    assert list(restored.dataloader.evaluator[1].test_times) == [6, 12]
