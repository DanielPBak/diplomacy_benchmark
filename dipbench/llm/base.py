"""Provider interface, call records and the per-game call logger."""
from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised before a call once the logger's spending cap has been reached."""


class LLMError(RuntimeError):
    """The call failed or returned no usable JSON (refusal, truncation, bad JSON)."""


class LLMUnavailable(RuntimeError):
    """The API itself failed after the SDK's retries (credits exhausted, rate limits,
    outages, network). The game stops instead of continuing with missing orders, and can be
    resumed from its last checkpoint."""


@dataclass
class LLMRequest:
    system: str                 # stable per agent per game, so it can be cached
    prompt: str                 # this turn's user message (what's new + the task)
    schema: dict                # JSON schema the response must match
    max_output_tokens: int      # thinking included
    effort: str | None = None   # low | medium | high | xhigh | max (Anthropic); ignored elsewhere
    # Logging metadata.
    game_id: str = ""
    power: str = ""
    phase: str = ""
    purpose: str = ""           # e.g. "negotiate_r0", "orders", "orders_retry1"
    # Earlier turns of this agent's conversation: alternating {"role": "user"|"assistant",
    # "content": str}, starting with "user". `prompt` is appended as the next user turn.
    history: list[dict] = field(default_factory=list)


@dataclass
class CallRecord:
    game_id: str
    power: str
    phase: str
    purpose: str
    provider: str
    model: str
    input_tokens: int = 0           # uncached prompt tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0          # includes thinking
    thinking_tokens: int | None = None  # when the provider reports it separately
    latency_s: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    error: str | None = None
    # Cost as reported by the provider, when it differs from what counts toward budgets (e.g.
    # claude -p on a subscription: reported API-equivalent cost, counted as $0).
    reported_cost_usd: float | None = None
    ts: float = field(default_factory=time.time)


@dataclass
class LLMResult:
    data: dict | None               # parsed JSON, None on failure
    text: str
    record: CallRecord
    # The model's reasoning or thinking summary, when the provider returns one. Logged for
    # analysis only, and never sent back to the model.
    reasoning: str | None = None


class Provider(ABC):
    name = "provider"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResult:
        """Make one call. Model-side failures (refusal, truncation, bad JSON) are returned with
        `record.error` set; they are part of what the benchmark measures. API failures raise
        LLMUnavailable, and the spending cap raises BudgetExceeded."""


class CallLogger:
    """Thread-safe log of every LLM call in a game, with an optional spending cap."""

    def __init__(self, max_usd: float | None = None, path: str | Path | None = None, resume: bool = False):
        """With resume=True, calls already in `path` (from an interrupted run) are loaded, so
        their cost counts toward `max_usd` and the summary. Otherwise `path` starts empty."""
        self.max_usd = max_usd
        self.path = Path(path) if path else None
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            if resume:
                self.records = [CallRecord(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
            else:
                self.path.unlink()

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self.records)

    def check_budget(self):
        if self.max_usd is not None and self.total_usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.total_usd:.4f} of ${self.max_usd:.2f} cap")

    def log(self, record: CallRecord):
        with self._lock:
            self.records.append(record)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(asdict(record)) + "\n")

    def summary(self) -> dict:
        with self._lock:
            rows = list(self.records)
        out = {"calls": len(rows), "errors": sum(1 for r in rows if r.error),
               "input_tokens": sum(r.input_tokens for r in rows),
               "cache_read_tokens": sum(r.cache_read_tokens for r in rows),
               "cache_write_tokens": sum(r.cache_write_tokens for r in rows),
               "output_tokens": sum(r.output_tokens for r in rows),
               "thinking_tokens": sum(r.thinking_tokens or 0 for r in rows),
               "cost_usd": sum(r.cost_usd for r in rows),
               "by_purpose": {}, "by_power": {}}
        for r in rows:
            kind = r.purpose.split("_")[0]
            for key, bucket in ((kind, out["by_purpose"]), (r.power, out["by_power"])):
                b = bucket.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                b["calls"] += 1
                b["input_tokens"] += r.input_tokens + r.cache_read_tokens + r.cache_write_tokens
                b["output_tokens"] += r.output_tokens
                b["cost_usd"] += r.cost_usd
        return out


_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def parse_json(text: str) -> dict | None:
    """Parse a JSON object, tolerating surrounding prose, a ```json fence, or inline
    <think>...</think> reasoning (e.g. Qwen3) from local models."""
    text = _THINK_BLOCK.sub("", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
