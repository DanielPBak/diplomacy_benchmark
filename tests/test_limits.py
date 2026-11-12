import pytest

from dipbench.llm.limits import OutputLimitConfig, resolve_output_limits

# Made-up prices ($ per million output tokens), for the arithmetic only.
PRICES = {"big": 50.0, "small": 5.0, "local": 0.0}
MODELS = {"FRANCE": "big", "ENGLAND": "small", "GERMANY": "local"}


def test_tokens_basis_is_equal_by_default():
    limits = resolve_output_limits(MODELS, OutputLimitConfig(), PRICES.get)
    assert set(limits.values()) == {8192}


def test_cost_basis_gives_equal_dollars():
    cfg = OutputLimitConfig(basis="cost", usd_per_call=0.2)
    limits = resolve_output_limits(MODELS, cfg, PRICES.get)
    assert limits["FRANCE"] == 4000       # $0.20 / $50 per Mtok
    assert limits["ENGLAND"] == 40000     # $0.20 / $5 per Mtok
    assert limits["GERMANY"] == 64000     # free local model: the ceiling


def test_cost_basis_floor_and_ceiling():
    cfg = OutputLimitConfig(basis="cost", usd_per_call=0.01, min_tokens=1024, max_tokens=5000)
    limits = resolve_output_limits(MODELS, cfg, PRICES.get)
    assert limits["FRANCE"] == 1024 and limits["ENGLAND"] == 2000 and limits["GERMANY"] == 5000


def test_overrides_win():
    cfg = OutputLimitConfig(basis="cost", usd_per_call=0.2, overrides={"FRANCE": 1234})
    assert resolve_output_limits(MODELS, cfg, PRICES.get)["FRANCE"] == 1234


def test_cost_basis_needs_budget():
    with pytest.raises(ValueError):
        OutputLimitConfig(basis="cost")
