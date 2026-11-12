"""Anthropic Messages API provider. The API key comes from the environment (ANTHROPIC_API_KEY
or an `ant auth login` profile); nothing is read from config files."""
from __future__ import annotations

import time

import anthropic

from .base import CallLogger, CallRecord, LLMRequest, LLMResult, LLMUnavailable, Provider, parse_json
from .pricing import call_cost


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, logger: CallLogger | None = None, client: anthropic.Anthropic | None = None,
                 max_retries: int = 6, timeout_s: float = 600.0, cache_ttl: str = "5m"):
        """`cache_ttl` "1h" keeps cached prefixes alive across long gaps between an agent's turns
        (e.g. tables with slow local models), at 2x the write price instead of 1.25x."""
        super().__init__(model)
        if cache_ttl not in ("5m", "1h"):
            raise ValueError("cache_ttl must be '5m' or '1h'")
        self.cache_control = {"type": "ephemeral"} if cache_ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}
        self.logger = logger
        self.client = client or anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)

    def build_kwargs(self, req: LLMRequest) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": req.schema}}
        if req.effort:
            output_config["effort"] = req.effort
        # Adaptive thinking; "summarized" returns a readable summary of it (billed the same as
        # the default, which returns nothing), kept for analysis and never sent back.
        return {
            "model": self.model,
            "max_tokens": req.max_output_tokens,
            # The system prompt is fixed for the game and the conversation is append-only, so
            # both are cached: each call re-reads the earlier turns at the cache-read rate and
            # pays full price only for the new turn. (Two breakpoints: system + automatic.)
            "system": [{"type": "text", "text": req.system, "cache_control": dict(self.cache_control)}],
            "messages": list(req.history) + [{"role": "user", "content": req.prompt}],
            "cache_control": dict(self.cache_control),
            "output_config": output_config,
            "thinking": {"type": "adaptive", "display": "summarized"},
        }

    def complete(self, req: LLMRequest) -> LLMResult:
        if self.logger:
            self.logger.check_budget()
        record = CallRecord(game_id=req.game_id, power=req.power, phase=req.phase, purpose=req.purpose,
                            provider=self.name, model=self.model)
        text, data, reasoning = "", None, None
        start = time.monotonic()
        try:
            kwargs = self.build_kwargs(req)
            if req.max_output_tokens > 16000:
                with self.client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
            else:
                response = self.client.messages.create(**kwargs)
        except anthropic.APIError as e:
            # Credits exhausted, auth, rate limits or outages that outlasted the SDK's retries.
            # Stop the game (resumable) rather than play on with missing orders.
            record.latency_s = time.monotonic() - start
            record.error = f"{type(e).__name__}: {e}"
            if self.logger:
                self.logger.log(record)
            raise LLMUnavailable(record.error) from e
        record.latency_s = time.monotonic() - start

        usage = response.usage
        record.input_tokens = usage.input_tokens or 0
        record.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        record.cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        breakdown = getattr(usage, "cache_creation", None)
        write_1h = getattr(breakdown, "ephemeral_1h_input_tokens", 0) or 0 if breakdown else 0
        record.output_tokens = usage.output_tokens or 0
        details = getattr(usage, "output_tokens_details", None)
        record.thinking_tokens = getattr(details, "thinking_tokens", None) if details else None
        record.stop_reason = response.stop_reason
        record.cost_usd = call_cost(self.model, record.input_tokens, record.output_tokens,
                                    record.cache_read_tokens, record.cache_write_tokens, write_1h)
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            record.error = f"refusal ({category})"
        elif response.stop_reason == "max_tokens":
            record.error = f"hit max_tokens={req.max_output_tokens}"
        else:
            text = "".join(b.text for b in response.content if b.type == "text")
            reasoning = "\n".join(b.thinking for b in response.content if b.type == "thinking" and b.thinking) or None
            data = parse_json(text)
            if data is None:
                record.error = "response was not a JSON object"

        if self.logger:
            self.logger.log(record)
        return LLMResult(data=data, text=text, record=record, reasoning=reasoning)
