import pytest
import torch

from xmetai.core.meta import Meta
from xmetai.models.tutorials.vit_ens import ViTModelEns


@pytest.mark.parametrize("requested_frames, expected_frames", [(None, 1), (2, 2), (8, 4)])
def test_vit_ensemble_weights_match_processed_target_window(requested_frames, expected_frames):
    model = ViTModelEns.__new__(ViTModelEns)
    Meta.__init__(model, device="cpu")
    model.in_frames = 2
    model.out_frames = 1
    model.alpha = 0.5
    data = {"inputs": torch.zeros(1, 2, 1, 2, 2), "targets": torch.zeros(1, 4, 1, 2, 2), "times": torch.tensor([0])}
    if requested_frames is not None:
        data["future_frames"] = requested_frames

    batch = model.process_data(data)

    assert batch["targets"].shape[1] == model.future_frames == expected_frames
    weights = 0.5 ** torch.arange(expected_frames, dtype=torch.float32)
    torch.testing.assert_close(model.rolling_weights, weights / weights.sum())
