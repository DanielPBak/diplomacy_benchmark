"""Provider that runs agents through the Claude Code CLI (`claude -p`), so frontier models can
be used through a Claude subscription as well as through API credits.

Each call is one headless `claude -p` run:
- `--system-prompt` replaces Claude Code's own system prompt.
- `--tools ""` disables every tool.
- `--safe-mode` and `--strict-mcp-config` ignore your CLAUDE.md, hooks, plugins and MCP servers.
- `--json-schema` enforces the reply format.
- `--effort` and `--model` are passed through, and the output cap is set with
  CLAUDE_CODE_MAX_OUTPUT_TOKENS.

A headless run has no multi-turn history, so the agent's earlier turns are sent as a transcript
in one user message, with each turn as its own content block (via --input-format stream-json). The
prompt cache can only reuse a prefix that ends at a block boundary, so separate blocks let every
call reuse the previous call's transcript; one growing text block would be re-written every call.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

from .base import CallLogger, CallRecord, LLMRequest, LLMResult, LLMUnavailable, Provider, parse_json

BILLING = ("subscription", "api")


TRANSCRIPT_HEADER = ("YOUR CONVERSATION IN THIS GAME, oldest first. The last \"YOU WERE SENT\" block is "
                     "the current turn: reply to it.")


def transcript_blocks(history: list[dict], prompt: str) -> list[str]:
    """One text block per turn, the newest last. A block's text never changes once sent (the
    newest turn is labelled like every other), so each call's cached prefix matches the next."""
    msgs = list(history) + [{"role": "user", "content": prompt}]
    blocks = []
    for i, m in enumerate(msgs):
        label = "=== YOU WERE SENT ===" if m["role"] == "user" else "=== YOU REPLIED ==="
        blocks.append((TRANSCRIPT_HEADER + "\n\n" if i == 0 else "") + f"{label}\n{m['content']}")
    return blocks


def render_transcript(history: list[dict], prompt: str) -> str:
    return "\n\n".join(transcript_blocks(history, prompt))


def stream_json_input(history: list[dict], prompt: str) -> str:
    content = [{"type": "text", "text": b} for b in transcript_blocks(history, prompt)]
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def result_event(stdout: str) -> dict | None:
    """The final `result` event from stream-json output."""
    found = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                found = ev
    return found


class ClaudeCLIProvider(Provider):
    name = "claude_cli"

    def __init__(self, model: str, logger: CallLogger | None = None, billing: str = "subscription",
                 cli: str | list[str] = "claude", extra_args: list[str] | None = None, timeout_s: float = 900.0):
        """`billing`: "subscription" counts calls as $0 toward spending caps (the CLI's reported
        cost is still logged as reported_cost_usd); "api" counts the reported cost."""
        super().__init__(model)
        if billing not in BILLING:
            raise ValueError(f"billing must be one of {BILLING}")
        self.logger = logger
        self.billing = billing
        self.cli = [cli] if isinstance(cli, str) else list(cli)
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        # Claude Code tells the model its working directory, so agents run from an empty,
        # neutrally named folder instead of the project (whose name gives the benchmark away).
        # Claude Code still adds the model's own name, the date, a token-budget note and the
        # account email, which API seats don't get.
        self.workdir = tempfile.mkdtemp(prefix="game_")

    def build_command(self, req: LLMRequest) -> list[str]:
        cmd = self.cli + ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                          "--model", self.model,
                          "--system-prompt", req.system, "--tools", "", "--safe-mode", "--strict-mcp-config",
                          "--no-session-persistence", "--json-schema", json.dumps(req.schema)]
        if req.effort:
            cmd += ["--effort", req.effort]
        return cmd + self.extra_args

    def complete(self, req: LLMRequest) -> LLMResult:
        if self.logger:
            self.logger.check_budget()
        record = CallRecord(game_id=req.game_id, power=req.power, phase=req.phase, purpose=req.purpose,
                            provider=self.name, model=self.model)
        env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(req.max_output_tokens)}
        start = time.monotonic()
        try:
            proc = subprocess.run(self.build_command(req), input=stream_json_input(req.history, req.prompt),
                                  capture_output=True, text=True, timeout=self.timeout_s, env=env,
                                  cwd=self.workdir)
            out = result_event(proc.stdout)
        except (subprocess.TimeoutExpired, OSError) as e:
            out, proc = None, None
            record.error = f"{type(e).__name__}: {e}"[:500]
        record.latency_s = time.monotonic() - start

        # No usable result, or an API-side failure (auth, rate limit, overload): stop the game
        # so it can be resumed, as with the other providers.
        if out is None or (out.get("is_error") and out.get("api_error_status")):
            if record.error is None:
                detail = (out or {}).get("result") or (proc.stderr if proc else "")
                record.error = f"claude CLI failed (exit {proc.returncode if proc else '?'}): {str(detail)[:400]}"
            if self.logger:
                self.logger.log(record)
            raise LLMUnavailable(record.error)

        usage = out.get("usage") or {}
        record.input_tokens = usage.get("input_tokens", 0) or 0
        record.cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
        record.cache_write_tokens = usage.get("cache_creation_input_tokens", 0) or 0
        record.output_tokens = usage.get("output_tokens", 0) or 0
        record.thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        record.stop_reason = out.get("subtype")
        record.reported_cost_usd = float(out.get("total_cost_usd") or 0.0)
        record.cost_usd = record.reported_cost_usd if self.billing == "api" else 0.0

        text = out.get("result") or ""
        data = out.get("structured_output")
        if not isinstance(data, dict):
            data = parse_json(text)
        if out.get("is_error"):
            record.error = f"claude CLI error: {out.get('subtype')}: {text[:300]}"
            data = None
        elif data is None:
            record.error = "response was not a JSON object"
        else:
            text = json.dumps(data)  # what goes back into the agent's memory
        if self.logger:
            self.logger.log(record)
        return LLMResult(data=data, text=text, record=record)
