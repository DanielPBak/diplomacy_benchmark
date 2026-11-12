"""Provider-agnostic LLM layer (Anthropic first, OpenAI-compatible endpoints for local models)."""
from __future__ import annotations

from .base import (BudgetExceeded, CallLogger, CallRecord, LLMError, LLMRequest, LLMResult, LLMUnavailable,
                   Provider)


def make_provider(spec: dict, logger: CallLogger | None = None) -> Provider:
    """Build a provider from a config dict: {"provider": "anthropic", "model": ...} or
    {"provider": "openai_compat", "model": ..., "base_url": ..., "api_key_env": ..., "json_mode": ...}."""
    kind = spec.get("provider", "anthropic")
    if kind == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(spec["model"], logger=logger, cache_ttl=spec.get("cache_ttl", "5m"))
    if kind == "claude_cli":
        from .claude_cli import ClaudeCLIProvider
        return ClaudeCLIProvider(spec["model"], logger=logger, billing=spec.get("billing", "subscription"),
                                 cli=spec.get("cli", "claude"), extra_args=spec.get("extra_args"))
    if kind == "openai_compat":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(spec["model"], spec["base_url"], logger=logger,
                                    api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                                    json_mode=spec.get("json_mode", "json_schema"),
                                    enable_thinking=spec.get("enable_thinking"),
                                    thinking_budget=spec.get("thinking_budget"))
    raise ValueError(f"unknown provider {kind!r}")


__all__ = ["BudgetExceeded", "CallLogger", "CallRecord", "LLMError", "LLMRequest", "LLMResult",
           "LLMUnavailable", "Provider", "make_provider"]
