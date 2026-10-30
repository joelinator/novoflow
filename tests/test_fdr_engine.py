import pytest
from novoflow.fdr.engine import FDREngine


def test_fdr_pep_calculation():
    engine = FDREngine(sigmoid_slope=1.0, sigmoid_center=5.0)
    # High score should yield very low PEP (low error probability)
    pep_high = engine.score_to_pep(10.0)
    assert pep_high < 0.01

    # Low score should yield high PEP
    pep_low = engine.score_to_pep(0.0)
    assert pep_low > 0.90


def test_q_value_monotonicity():
    engine = FDREngine()
    scores = [1.0, 3.5, 7.8, 2.1, 9.4, 5.2]
    peps, q_values = engine.compute_q_values_empirical(scores)

    assert len(q_values) == len(scores)
    # Sort pairs by score descending
    pairs = sorted(zip(scores, q_values), key=lambda x: x[0], reverse=True)
    for i in range(len(pairs) - 1):
        # Higher or equal score must have lower or equal q-value
        assert pairs[i][1] <= pairs[i + 1][1] + 1e-6


def test_filter_at_fdr():
    engine = FDREngine(sigmoid_slope=2.0, sigmoid_center=3.0)
    records = [
        {"sequence": "HIGH_CONF_1", "novoflow_score": 10.0},
        {"sequence": "HIGH_CONF_2", "novoflow_score": 9.0},
        {"sequence": "LOW_CONF_1", "novoflow_score": -2.0},
    ]
    filtered = engine.filter_at_fdr(records, fdr_threshold=0.05)
    assert len(filtered) >= 1
    assert any(r["sequence"] == "HIGH_CONF_1" for r in filtered)
    assert all(r["sequence"] != "LOW_CONF_1" for r in filtered)
