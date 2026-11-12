"""Per-model prices in USD per million tokens.

Anthropic first-party API rates as of 2026-10 (input / output / cache read). Cache writes
(5-minute TTL) are billed at 1.25x input. Claude Haiku 5.5 is $0.10 / $0.50 for prompts of
100K tokens or fewer and $0.50 / $2.50 above that. Haiku's cache-read rate isn't in the
cached docs; it's assumed to be 0.1x input like the other models. Re-check against
https://platform.claude.com/docs/en/about-claude/pricing before relying on totals.

Local models default to $0. Give them a price in the run config to compare on cost.
"""
from __future__ import annotations

from dataclasses import dataclass

CACHE_WRITE_MULTIPLIER = 1.25      # 5-minute cache entries
CACHE_WRITE_1H_MULTIPLIER = 2.0    # 1-hour cache entries


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    long_context_threshold: int | None = None  # prompt tokens above which long_* apply
    long_input: float | None = None
    long_output: float | None = None


ANTHROPIC_PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.00, 50.00, 0.25),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, long_context_threshold=100_000,
                              long_input=0.50, long_output=2.50),
}

FREE = Price(0.0, 0.0, 0.0)

_custom: dict[str, Price] = {}


def register_price(model: str, price: Price):
    """Set a price for a model not in the table (e.g. a hosted open-weight model)."""
    _custom[model] = price


def price_for(model: str) -> Price:
    if model in _custom:
        return _custom[model]
    if model in ANTHROPIC_PRICES:
        return ANTHROPIC_PRICES[model]
    return FREE


def output_price_per_mtok(model: str) -> float:
    return price_for(model).output


def call_cost(model: str, input_tokens: int, output_tokens: int,
              cache_read_tokens: int = 0, cache_write_tokens: int = 0, cache_write_1h_tokens: int = 0) -> float:
    """USD cost of one call. `input_tokens` excludes cached tokens, as the Anthropic API reports it;
    `cache_write_1h_tokens` is the part of `cache_write_tokens` written with the 1-hour lifetime."""
    p = price_for(model)
    prompt_total = input_tokens + cache_read_tokens + cache_write_tokens
    in_rate, out_rate = p.input, p.output
    if p.long_context_threshold is not None and prompt_total > p.long_context_threshold:
        in_rate, out_rate = p.long_input, p.long_output
    return (input_tokens * in_rate
            + (cache_write_tokens - cache_write_1h_tokens) * in_rate * CACHE_WRITE_MULTIPLIER
            + cache_write_1h_tokens * in_rate * CACHE_WRITE_1H_MULTIPLIER
            + cache_read_tokens * p.cache_read
            + output_tokens * out_rate) / 1_000_000
