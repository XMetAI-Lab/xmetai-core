"""Unit tests for Integrated Pattern Score (IPS) metric."""

import torch

from xmetai import METRICS
from xmetai.metrics.ips import IPS, IntegratedPatternScore

TEST_NAMES = ["t2m", "tp"]
PENTADS = list(range(3, 13))
HEIGHT, WIDTH = 8, 16
BSZ = 2


def _make_evaluator(**overrides):
    kwargs = dict(
        test_names=TEST_NAMES,
        test_times=PENTADS,
        lead_unit="pentad",
    )
    kwargs.update(overrides)
    return IPS(**kwargs)


def _mock_batch(output, target, **extra):
    return {"output": output, "target": target, **extra}


def test_ips_pattern_scores_perfect():
    pred = torch.randn(HEIGHT, WIDTH)
    pcc, as_score, ips = IntegratedPatternScore._pattern_scores(pred, pred.clone(), None)

    assert abs(pcc - 1.0) < 1e-5
    assert abs(as_score - 1.0) < 1e-5
    assert abs(ips - 100.0) < 1e-4


def test_ips_process_and_evaluate():
    torch.manual_seed(0)
    evaluator = _make_evaluator()
    evaluator.reset()

    output = torch.randn(BSZ, len(PENTADS), len(TEST_NAMES), HEIGHT, WIDTH)
    target = output.clone()

    evaluator.process({}, _mock_batch(output, target))
    results = evaluator.evaluate()

    assert results is not None
    assert "IPS" in results
    table = results["IPS"]
    assert table["Pentad\\IPS"][-1] == "Overall"
    for name in TEST_NAMES:
        assert name in table
        assert abs(table[name][-1] - 100.0) < 1e-4
        for ips_val in table[name][:-1]:
            assert abs(ips_val - 100.0) < 1e-4


def test_ips_evaluate_hour_leads():
    test_times = [day * 24 for day in range(11, 61)]
    evaluator = IPS(
        test_names=["var0"],
        test_times=test_times,
        lead_unit="hour",
        start_pentad=3,
        end_pentad=12,
    )
    evaluator.reset()

    output = torch.randn(1, len(test_times), 1, HEIGHT, WIDTH)
    target = output.clone()
    evaluator.process({}, _mock_batch(output, target))
    results = evaluator.evaluate()

    assert results is not None
    assert results["IPS"]["var0"][-1] == 100.0


def test_ips_registered():
    assert "IPS" in METRICS
    assert METRICS.get("IPS") is IPS
