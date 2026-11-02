"""Public family imports and discovery after the model layout migration."""

from xmetai import MODELS
from xmetai.models.adapter import ForecastAdapter
from xmetai.models.afno import AFNONet
from xmetai.models.dit import DiffusionForecast, FlowMatchingForecast, JiTForecast
from xmetai.models.infer import OnnxInferModel
from xmetai.models.puyun import PuYunModel
from xmetai.models.tianquan import TianQuan
from xmetai.models.tongyun import TongYun
from xmetai.models.tutorials import ViTModel, ViTModelEns
from xmetai.registries import import_builtin_registrations


def test_public_model_families_register_once():
    models = (
        ForecastAdapter,
        AFNONet,
        DiffusionForecast,
        FlowMatchingForecast,
        JiTForecast,
        OnnxInferModel,
        PuYunModel,
        TianQuan,
        TongYun,
        ViTModel,
        ViTModelEns,
    )
    before = dict(MODELS)
    import_builtin_registrations()
    assert dict(MODELS) == before
    for model in models:
        assert MODELS.get(model.__name__) is model
    assert "GraphCastPredictor" not in MODELS
    assert "CorrectionModel" not in MODELS
    assert "GroupVAEPredictor" not in MODELS
    assert "FeatureGroupVAEPredictor" not in MODELS
