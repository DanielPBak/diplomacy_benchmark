"""Provider for any OpenAI-compatible chat-completions endpoint: local servers (Ollama,
vLLM, llama.cpp, LM Studio) or hosted ones. The API key is read from the environment
variable named in the config (local servers usually accept any value)."""
from __future__ import annotations

import json
import os
import time

import openai

from .base import CallLogger, CallRecord, LLMRequest, LLMResult, LLMUnavailable, Provider, parse_json
from .pricing import call_cost


BUDGET_CUTOFF = "\n\nI'm out of thinking budget, so I'll stop thinking here and give my answer now."


def render_chatml(system: str, history: list[dict], prompt: str) -> str:
    """The conversation in ChatML (Qwen's chat format), ending with an open thinking section.
    Earlier assistant turns are rendered without reasoning, as Qwen's own template does."""
    out = f"<|im_start|>system\n{system}<|im_end|>\n"
    for m in history:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, model: str, base_url: str, logger: CallLogger | None = None,
                 api_key_env: str = "OPENAI_API_KEY", json_mode: str = "json_schema",
                 enable_thinking: bool | None = None, thinking_budget: int | None = None,
                 max_retries: int = 2, timeout_s: float = 600.0):
        super().__init__(model)
        self.logger = logger
        self.json_mode = json_mode  # "json_schema", "json_object", or "none" (prompt only)
        # For models with switchable reasoning (Qwen3 and others), passed to the chat template
        # as chat_template_kwargs; servers that don't support it ignore it. None: model default.
        self.enable_thinking = enable_thinking
        # Hard cap on thinking tokens via budget forcing, for ChatML reasoning models (Qwen3.x):
        # think up to `thinking_budget` tokens through the raw /v1/completions endpoint; if the
        # model hasn't closed its thinking by then, close it with a short note and generate the
        # answer from there. Uses the server's prefix cache like chat requests do.
        self.thinking_budget = thinking_budget
        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "not-needed"),
                                    max_retries=max_retries, timeout=timeout_s)

    def _response_format(self, req: LLMRequest) -> dict | None:
        if self.json_mode == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": "response", "schema": req.schema, "strict": True}}
        if self.json_mode == "json_object":
            return {"type": "json_object"}
        return None

    def _complete_with_budget(self, req: LLMRequest, prompt: str, record: CallRecord):
        """Returns (answer_text, reasoning, finish_reason). Raises openai.APIError on API failure."""
        base = render_chatml(req.system, req.history, prompt)
        think = self.client.completions.create(model=self.model, prompt=base, max_tokens=self.thinking_budget,
                                               stop=["</think>"])
        reasoning = (think.choices[0].text or "").split("</think>")[0]
        cut_off = think.choices[0].finish_reason == "length"
        if cut_off:
            reasoning += BUDGET_CUTOFF
        used = think.usage.completion_tokens if think.usage else 0
        answer = self.client.completions.create(
            model=self.model, prompt=base + reasoning + "\n</think>\n\n",
            max_tokens=max(512, req.max_output_tokens - used), stop=["<|im_end|>"])
        for u in (think.usage, answer.usage):
            if u is not None:
                cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                record.input_tokens += (u.prompt_tokens or 0) - cached
                record.cache_read_tokens += cached
                record.output_tokens += u.completion_tokens or 0
        record.thinking_tokens = used
        return answer.choices[0].text or "", reasoning, answer.choices[0].finish_reason

    def complete(self, req: LLMRequest) -> LLMResult:
        if self.logger:
            self.logger.check_budget()
        record = CallRecord(game_id=req.game_id, power=req.power, phase=req.phase, purpose=req.purpose,
                            provider=self.name, model=self.model)
        text, data, reasoning = "", None, None
        prompt = req.prompt
        if self.json_mode != "json_schema":
            # The server won't enforce the schema, so state it in the prompt.
            prompt += ("\n\nRespond with only a JSON object (no other text) matching this JSON schema:\n"
                       + json.dumps(req.schema))
        kwargs = {"model": self.model, "max_tokens": req.max_output_tokens,
                  "messages": [{"role": "system", "content": req.system}, *req.history,
                               {"role": "user", "content": prompt}]}
        fmt = self._response_format(req)
        if fmt:
            kwargs["response_format"] = fmt
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        start = time.monotonic()
        if self.thinking_budget:
            try:
                text, reasoning, finish = self._complete_with_budget(req, prompt, record)
            except openai.APIError as e:
                record.latency_s = time.monotonic() - start
                record.error = f"{type(e).__name__}: {e}"
                if self.logger:
                    self.logger.log(record)
                raise LLMUnavailable(record.error) from e
            record.latency_s = time.monotonic() - start
            record.stop_reason = finish
            data = parse_json(text)
            if finish == "length":
                record.error = f"hit max_tokens={req.max_output_tokens}"
                data = None
            elif data is None:
                record.error = "response was not a JSON object"
            if self.logger:
                self.logger.log(record)
            return LLMResult(data=data, text=text, record=record, reasoning=reasoning)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            record.latency_s = time.monotonic() - start
            record.error = f"{type(e).__name__}: {e}"
            if self.logger:
                self.logger.log(record)
            raise LLMUnavailable(record.error) from e
        record.latency_s = time.monotonic() - start

        usage = response.usage
        if usage is not None:
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            record.input_tokens = (usage.prompt_tokens or 0) - cached
            record.cache_read_tokens = cached
            record.output_tokens = usage.completion_tokens or 0
            details = getattr(usage, "completion_tokens_details", None)
            record.thinking_tokens = getattr(details, "reasoning_tokens", None) if details else None
        choice = response.choices[0]
        record.stop_reason = choice.finish_reason
        record.cost_usd = call_cost(self.model, record.input_tokens, record.output_tokens,
                                    record.cache_read_tokens)
        if choice.finish_reason == "length":
            record.error = f"hit max_tokens={req.max_output_tokens}"
        else:
            text = choice.message.content or ""
            # Servers return a reasoning model's thinking separately (MLX: "reasoning";
            # vLLM and others: "reasoning_content"). Kept for analysis only.
            reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None)
            data = parse_json(text)
            if data is None:
                record.error = "response was not a JSON object"

        if self.logger:
            self.logger.log(record)
        return LLMResult(data=data, text=text, record=record, reasoning=reasoning)
