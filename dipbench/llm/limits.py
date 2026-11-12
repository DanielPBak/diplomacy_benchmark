"""Per-call output limits (thinking included) for each power's model.

`tokens` basis (default): every model gets the same output-token cap, so models are
compared on equal output compute. `cost` basis: every model gets the same dollar budget
per call, so each one's cap is that budget divided by its output price. Per-power
overrides win over both. Agents are told their own cap.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

LIMIT_BASES = ("tokens", "cost")


@dataclass
class OutputLimitConfig:
    basis: str = "tokens"
    max_output_tokens: int = 8192            # tokens basis: the same cap for every model
    usd_per_call: float | None = None        # cost basis: the same $ per call for every model
    overrides: dict[str, int] = field(default_factory=dict)  # power -> tokens
    min_tokens: int = 1024                   # floor for the cost basis
    max_tokens: int = 64000                  # ceiling for the cost basis

    def __post_init__(self):
        if self.basis not in LIMIT_BASES:
            raise ValueError(f"basis must be one of {LIMIT_BASES}, got {self.basis!r}")
        if self.basis == "cost" and not self.usd_per_call:
            raise ValueError("cost basis needs usd_per_call")


def resolve_output_limits(models: dict[str, str], cfg: OutputLimitConfig,
                          output_price_per_mtok: Callable[[str], float]) -> dict[str, int]:
    """Output-token cap per power. `models` maps power -> model id; the price function
    returns a model's output price in $ per million tokens (0 for local models)."""
    limits = {}
    for power, model in models.items():
        if power in cfg.overrides:
            limits[power] = int(cfg.overrides[power])
        elif cfg.basis == "tokens":
            limits[power] = cfg.max_output_tokens
        else:
            price = output_price_per_mtok(model)
            tokens = cfg.max_tokens if price <= 0 else int(cfg.usd_per_call / price * 1_000_000)
            limits[power] = max(cfg.min_tokens, min(cfg.max_tokens, tokens))
    return limits
