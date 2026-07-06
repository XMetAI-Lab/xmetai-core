"""Unit tests for Climate PS Score (ClimatePSScore / PS) metric."""

import pytest
import torch

from xmetai import METRICS
from xmetai.metrics.ps import PS, ClimatePSScore

# ─── Test constants ────────────────────────────────────────────────────────────

TEST_NAMES = ["t2m", "prcp"]
TEST_TIMES = [1, 2, 3]
HEIGHT, WIDTH = 12, 13
BSZ = 2
L1, L2 = 0.20, 0.50


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_evaluator(**overrides):
    kwargs = dict(test_names=TEST_NAMES, test_times=TEST_TIMES)
    kwargs.update(overrides)
    return PS(**kwargs)


def _mock_batch(output, target):
    return {"output": output, "target": target}


# ─── _classify_level ──────────────────────────────────────────────────────────


def test_classify_level_normal():
    anom = torch.tensor([0.0, 0.10, 0.19])
    level = ClimatePSScore._classify_level(anom.abs(), L1, L2)
    assert (level == 0).all()


def test_classify_level_l1():
    anom = torch.tensor([0.20, 0.35, 0.499])
    level = ClimatePSScore._classify_level(anom.abs(), L1, L2)
    assert (level == 1).all()


def test_classify_level_l2():
    anom = torch.tensor([0.50, 0.80, 2.0])
    level = ClimatePSScore._classify_level(anom.abs(), L1, L2)
    assert (level == 2).all()


def test_classify_level_boundary_l1():
    anom = torch.tensor([L1])
    assert ClimatePSScore._classify_level(anom, L1, L2).item() == 1


def test_classify_level_boundary_l2():
    anom = torch.tensor([L2])
    assert ClimatePSScore._classify_level(anom, L1, L2).item() == 2


# ─── _ps_formula ──────────────────────────────────────────────────────────────


def test_ps_formula_all_normal_correct():
    """All normal, all sign-match: N0=N, N1=0, N2=0 → PS = 2N/2N*100 = 100."""
    n = 165.0
    assert abs(ClimatePSScore._ps_formula(n0=n, n1=0.0, n2=0.0, n=n) - 100.0) < 1e-6


def test_ps_formula_all_l1_correct():
    """All level-1, all sign-match: N0=N, N1=N, N2=0 → PS = 4N/4N*100 = 100."""
    n = 165.0
    assert abs(ClimatePSScore._ps_formula(n0=n, n1=n, n2=0.0, n=n) - 100.0) < 1e-6


def test_ps_formula_all_l2_correct():
    """All level-2, all sign-match: N0=N, N1=N, N2=N → PS = 8N/8N*100 = 100."""
    n = 165.0
    assert abs(ClimatePSScore._ps_formula(n0=n, n1=n, n2=n, n=n) - 100.0) < 1e-6


def test_ps_formula_all_wrong():
    """All wrong (N0=N1=N2=0): PS = 0."""
    assert abs(ClimatePSScore._ps_formula(n0=0.0, n1=0.0, n2=0.0, n=165.0) - 0.0) < 1e-6


def test_ps_formula_empty_grid():
    """N=0 must return 0 without division error."""
    assert ClimatePSScore._ps_formula(n0=0.0, n1=0.0, n2=0.0, n=0.0) == 0.0


def test_ps_formula_analytical():
    """Hand-computed: N=4, N0=3, N1=1, N2=0.

    numerator   = 2*3 + 2*1 + 4*0 = 8
    denominator = 4   + 3   + 2*1 + 4*0 = 9
    PS = 8/9 * 100 ≈ 88.8889
    """
    ps = ClimatePSScore._ps_formula(n0=3.0, n1=1.0, n2=0.0, n=4.0)
    assert abs(ps - 8.0 / 9.0 * 100.0) < 1e-5


# ─── _accumulate_counts ───────────────────────────────────────────────────────


def test_accumulate_counts_all_correct():
    """Each point is its own mirror: every count is fully consistent."""
    pred = torch.tensor([0.10, 0.30, 0.60, -0.10, -0.30, -0.60])
    obs = pred.clone()
    n0, n1, n2, n = ClimatePSScore._accumulate_counts(pred, obs, L1, L2)
    assert n == 6.0
    assert n0 == 6.0  # all same sign
    assert n1 == 4.0  # |anom| >= 0.20: ±0.30, ±0.60
    assert n2 == 2.0  # |anom| >= 0.50: ±0.60


def test_accumulate_counts_all_wrong_sign():
    pred = torch.tensor([0.10, 0.35])
    obs = -pred.clone()
    n0, n1, n2, n = ClimatePSScore._accumulate_counts(pred, obs, L1, L2)
    assert n0 == 0.0 and n1 == 0.0 and n2 == 0.0


def test_accumulate_counts_mixed_levels():
    """Verify N1/N2 only count when BOTH pred and obs are at sufficient level."""
    # obs level-2, pred level-1 → not counted in N2 (pred is not level-2)
    pred = torch.tensor([0.30])  # level-1
    obs = torch.tensor([0.60])  # level-2
    n0, n1, n2, n = ClimatePSScore._accumulate_counts(pred, obs, L1, L2)
    assert n0 == 1.0  # same sign
    assert n1 == 1.0  # both >= level-1
    assert n2 == 0.0  # pred is not level-2


# ─── Full pipeline ─────────────────────────────────────────────────────────────


def test_process_and_evaluate_perfect():
    """Identical pred and target → PS = 100 at every lead and overall."""
    torch.manual_seed(42)
    evaluator = _make_evaluator()
    evaluator.reset()

    output = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH)
    target = output.clone()

    evaluator.process({}, _mock_batch(output, target))
    results = evaluator.evaluate()

    assert results is not None
    assert "PS" in results
    table = results["PS"]
    assert table["Lead\\PS"][-1] == "Overall"
    for name in TEST_NAMES:
        assert name in table
        for ps_val in table[name]:
            assert abs(ps_val - 100.0) < 1e-4, f"{name}: expected 100, got {ps_val}"


def test_process_and_evaluate_zero():
    """Opposite-sign (normal-range) predictions → PS = 0 everywhere."""
    evaluator = _make_evaluator()
    evaluator.reset()

    output = torch.full((BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH), 0.05)
    target = -output.clone()

    evaluator.process({}, _mock_batch(output, target))
    results = evaluator.evaluate()

    assert results is not None
    table = results["PS"]
    for name in TEST_NAMES:
        for ps_val in table[name]:
            assert abs(ps_val - 0.0) < 1e-6, f"{name}: expected 0, got {ps_val}"


def test_accumulate_across_batches_is_consistent():
    """Splitting the same data into two identical batches gives the same PS."""
    torch.manual_seed(7)
    evaluator = _make_evaluator()

    output = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH)
    target = torch.randn_like(output)

    evaluator.reset()
    evaluator.process({}, _mock_batch(output, target))
    results_one = evaluator.evaluate()

    evaluator.reset()
    evaluator.process({}, _mock_batch(output, target))
    evaluator.process({}, _mock_batch(output, target))
    results_two = evaluator.evaluate()

    for name in TEST_NAMES:
        for v1, v2 in zip(results_one["PS"][name], results_two["PS"][name]):
            assert abs(v1 - v2) < 1e-5, f"{name}: single={v1} double={v2}"


def test_overall_aggregates_across_leads():
    """Overall PS aggregates raw counts across all leads, not the average of per-lead PS."""
    evaluator = _make_evaluator()
    evaluator.reset()

    output = torch.randn(1, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH)
    target = output.clone()

    evaluator.process({}, _mock_batch(output, target))
    results = evaluator.evaluate()

    for name in TEST_NAMES:
        assert abs(results["PS"][name][-1] - 100.0) < 1e-4


# ─── Output structure ──────────────────────────────────────────────────────────


def test_result_structure():
    evaluator = _make_evaluator()
    evaluator.reset()

    output = torch.randn(1, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH)
    evaluator.process({}, _mock_batch(output, output.clone()))
    results = evaluator.evaluate()

    table = results["PS"]
    assert len(table["Lead\\PS"]) == len(TEST_TIMES) + 1
    for name in TEST_NAMES:
        assert len(table[name]) == len(TEST_TIMES) + 1
        assert table["Lead\\PS"][-1] == "Overall"
        assert all(isinstance(v, float) for v in table[name])


# ─── Custom thresholds ─────────────────────────────────────────────────────────


def test_custom_thresholds_all_l1():
    """With L1=0.30 / L2=0.70, values at 0.50 are level-1.
    All same sign, all level-1: PS = (2N+2N)/(N+N+2N) * 100 = 100.
    """
    evaluator = PS(test_names=["var"], test_times=[1], threshold_l1=0.30, threshold_l2=0.70)
    evaluator.reset()

    v = torch.full((1, 1, 1, 4, 4), 0.50)
    evaluator.process({}, _mock_batch(v, v.clone()))
    results = evaluator.evaluate()

    assert abs(results["PS"]["var"][0] - 100.0) < 1e-4


def test_custom_threshold_no_l2():
    """Values below L2 never contribute to N2, so N2=0."""
    evaluator = PS(test_names=["var"], test_times=[1], threshold_l1=0.10, threshold_l2=0.90)
    evaluator.reset()

    v = torch.full((1, 1, 1, 4, 4), 0.50)  # >= L1 but < L2=0.90
    evaluator.process({}, _mock_batch(v, v.clone()))
    results = evaluator.evaluate()

    # All level-1, PS = 100
    assert abs(results["PS"]["var"][0] - 100.0) < 1e-4


# ─── Custom output/target keys ────────────────────────────────────────────────


def test_custom_output_target_keys():
    evaluator = PS(test_names=["var"], test_times=[1], output_key="pred", target_key="obs")
    evaluator.reset()

    v = torch.ones(1, 1, 1, 4, 4) * 0.05
    evaluator.process({}, {"pred": v, "obs": v.clone()})
    results = evaluator.evaluate()

    assert results is not None
    assert "PS" in results


# ─── Error handling ────────────────────────────────────────────────────────────


def test_invalid_threshold_order():
    with pytest.raises(ValueError, match="threshold"):
        PS(test_names=["x"], test_times=[1], threshold_l1=0.50, threshold_l2=0.20)


def test_equal_thresholds_rejected():
    with pytest.raises(ValueError, match="threshold"):
        PS(test_names=["x"], test_times=[1], threshold_l1=0.50, threshold_l2=0.50)


def test_empty_test_names():
    with pytest.raises(ValueError, match="test_names"):
        PS(test_names=[], test_times=[1])


def test_empty_test_times():
    with pytest.raises(ValueError, match="test_times"):
        PS(test_names=["x"], test_times=[])


def test_wrong_output_ndim():
    evaluator = _make_evaluator()
    evaluator.reset()
    bad = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT)  # 4D
    with pytest.raises(ValueError, match="B, T, C, H, W"):
        evaluator.process({}, _mock_batch(bad, bad))


def test_shape_mismatch():
    evaluator = _make_evaluator()
    evaluator.reset()
    output = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT, WIDTH)
    target = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES), HEIGHT + 1, WIDTH)
    with pytest.raises(ValueError, match="shapes must match"):
        evaluator.process({}, _mock_batch(output, target))


def test_lead_time_count_mismatch():
    evaluator = _make_evaluator()
    evaluator.reset()
    output = torch.randn(BSZ, len(TEST_TIMES) + 1, len(TEST_NAMES), HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="lead times"):
        evaluator.process({}, _mock_batch(output, output.clone()))


def test_channel_count_too_few():
    evaluator = _make_evaluator()
    evaluator.reset()
    output = torch.randn(BSZ, len(TEST_TIMES), len(TEST_NAMES) - 1, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="channels"):
        evaluator.process({}, _mock_batch(output, output.clone()))


# ─── Registry ──────────────────────────────────────────────────────────────────


def test_ps_registered():
    assert "PS" in METRICS
    assert METRICS.get("PS") is PS


def test_climate_ps_score_registered():
    assert "ClimatePSScore" in METRICS
    assert METRICS.get("ClimatePSScore") is ClimatePSScore
