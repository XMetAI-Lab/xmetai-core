"""Layout / SlotSpec 的 CPU 单元测试(无需 NATTEN / GPU)。"""

from __future__ import annotations

import pytest
import torch

from xmetai.layers.async_multimodal import (
    HEBEI_7FRAME,
    NULL_MODALITY,
    STATE_MISSING,
    STATE_OBSERVED,
    Availability,
    Layout,
    SlotSpec,
)

OLD_TEMPORAL_MAPPING = (0, 0, 1, 2, 3, 3, 4, 5, 6, 6)
OLD_NOISE_OUTPUT_INDICES = (1, 2, 3, 5, 6, 7, 9)


class TestSlotSpec:
    def test_basic(self):
        s = SlotSpec("noise", 3)
        assert s.modality == "noise" and s.time == 3

    def test_reject_empty_modality(self):
        with pytest.raises(ValueError):
            SlotSpec("", 0)

    def test_reject_null_modality_name(self):
        with pytest.raises(ValueError):
            SlotSpec(NULL_MODALITY, 0)

    def test_reject_non_int_time(self):
        with pytest.raises(TypeError):
            SlotSpec("noise", 1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SlotSpec("noise", True)  # type: ignore[arg-type]

    def test_frozen(self):
        s = SlotSpec("noise", 0)
        with pytest.raises(Exception):
            s.time = 5  # type: ignore[misc]


class TestLayoutSort:
    def test_hebei_7frame_slot_times(self):
        layout = Layout(HEBEI_7FRAME())
        assert layout.seq_len == 10
        assert tuple(s.time for s in layout.slots) == OLD_TEMPORAL_MAPPING

    def test_hebei_7frame_modality_sequence(self):
        mods = [s.modality for s in Layout(HEBEI_7FRAME()).slots]
        assert mods[0] == "init"  # t=0 init 在 noise 前(声明顺序)
        assert mods[4] == "coarse" and mods[5] == "noise"  # t=3 coarse 在 noise 前
        assert mods[8] == "coarse" and mods[9] == "noise"  # t=6

    def test_sort_by_time_decl_order_at_same_time(self):
        layout = Layout([SlotSpec("a", 0), SlotSpec("b", 0), SlotSpec("c", 1)])
        assert [(s.modality, s.time) for s in layout.slots] == [("a", 0), ("b", 0), ("c", 1)]

    def test_nonuniform_time_supported(self):
        layout = Layout([SlotSpec("noise", 0), SlotSpec("noise", 3), SlotSpec("sat", 3), SlotSpec("noise", 12)])
        assert [s.time for s in layout.slots] == [0, 3, 3, 12]


class TestLayoutStructure:
    def test_structure_from_hebei(self):
        layout = Layout(HEBEI_7FRAME())
        assert layout.seq_len == 10
        assert tuple(s.time for s in layout.slots) == OLD_TEMPORAL_MAPPING
        assert layout.num_modalities == 4
        assert layout.modalities == sorted([NULL_MODALITY, "init", "noise", "coarse"])

    def test_reject_empty(self):
        with pytest.raises(ValueError):
            Layout([])

    def test_slot_indices_noise_matches_old_output(self):
        layout = Layout(HEBEI_7FRAME())
        noise_idxs = []
        for t in range(7):
            noise_idxs.extend(layout.slot_indices("noise", t))
        assert tuple(sorted(noise_idxs)) == OLD_NOISE_OUTPUT_INDICES

    def test_accepts_dict_slots(self):
        layout = Layout([{"modality": "coarse", "time": 0}, {"modality": "noise", "time": 0}])
        assert layout.seq_len == 2


class TestAvailability:
    def test_helpers(self):
        a = Availability.all_available(2, 10)
        assert a.batch_size == 2 and a.seq_len == 10 and torch.all(a.flags)
        assert not torch.any(Availability.none_available(2, 10).flags)

    def test_coerce_to_bool(self):
        a = Availability(torch.tensor([[1, 0], [0, 1]]))
        assert a.flags.dtype == torch.bool and a.flags[0].tolist() == [True, False]

    def test_reject_wrong_shape(self):
        with pytest.raises(ValueError):
            Availability(torch.ones(3))


class TestPositionInfo:
    def test_default_time_coords(self):
        layout = Layout(HEBEI_7FRAME())
        tc = layout.default_time_coords(B=2)
        assert tc.shape == (2, 10) and tc.dtype == torch.float32
        assert tc[0].tolist() == list(OLD_TEMPORAL_MAPPING)

    def test_position_info_fields(self):
        layout = Layout(HEBEI_7FRAME())
        tc = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        pi = layout.position_info(tc)
        assert pi.modality_id.shape == (2, 10) and pi.modality_id.dtype == torch.long
        assert pi.time_coord.shape == (2, 10) and torch.equal(pi.time_coord, tc)
        assert pi.state_id.shape == (2, 10) and (pi.state_id == STATE_OBSERVED).all()  # 默认全 observed

    def test_position_info_state_on_unavailable(self):
        # 关键:缺失 slot 的 modality_id 保留原模态(不丢),state_id=missing
        layout = Layout(HEBEI_7FRAME())
        tc = layout.default_time_coords(B=2)
        flags = torch.ones(2, 10, dtype=torch.bool)
        flags[:, [0, 4, 8]] = False  # init@0 / coarse@3 / coarse@6 缺
        pi = layout.position_info(tc, flags)  # 接 Tensor
        # modality_id 保留原(init/coarse 的 id,不是 null_id)
        assert pi.modality_id[0, 0].item() != layout.null_id
        assert pi.modality_id[0, 4].item() != layout.null_id
        assert pi.modality_id[0, 1].item() != layout.null_id  # noise 保留
        # state_id:缺失=missing,有数据=observed
        assert pi.state_id[0, 0].item() == STATE_MISSING
        assert pi.state_id[0, 4].item() == STATE_MISSING
        assert pi.state_id[0, 1].item() == STATE_OBSERVED

    def test_position_info_all_available(self):
        layout = Layout(HEBEI_7FRAME())
        tc = layout.default_time_coords(B=1)
        pi = layout.position_info(tc, torch.ones(1, 10, dtype=torch.bool))
        assert torch.equal(pi.modality_id[0], layout.slot_mod_ids())
        assert (pi.state_id == STATE_OBSERVED).all()

    def test_position_info_wrong_shape_raises(self):
        layout = Layout(HEBEI_7FRAME())
        with pytest.raises(ValueError):
            layout.position_info(torch.zeros(2, 9))
