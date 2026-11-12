"""LLM-backed agent with a persistent memory.

Each power plays the whole game as one append-only conversation: every turn sends only what
is new since its last turn, and its own earlier replies stay in the conversation. Earlier
private reasoning is not carried over, for every model alike. Each order submission also
writes a short diary entry. If the conversation outgrows the memory window, only the
conversation of the oldest turns (messages and replies) is removed: it is replaced by the
complete orders-and-results record of those phases plus the diary entries written then, so
the board history is never lost. Trims are logged. The memory rules are stated in the
system prompt.
"""
from __future__ import annotations

from ..config import GameConfig
from ..llm.base import LLMRequest, Provider
from ..state import PhaseContext
from ..tokens import count_tokens
from . import prompts
from .base import Agent, OrderSubmission, OutgoingMessage

NO_REPLY = "(no usable reply)"


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, power: str, scoring: str, provider: Provider, config: GameConfig,
                 output_limit: int = 4096, effort: str | None = "low", identity: str | None = None,
                 memory_window: int = 200_000, diary_max_chars: int = 800, parse_retries: int = 1,
                 trim_to: float = 0.7, thinking_hint: str | None = None, **kwargs):
        """`memory_window` is in model-neutral tokens (dipbench.tokens). Every seat in a game
        should get the same window, sized for the smallest-context model at the table."""
        if scoring not in prompts.SCORING_RULES:
            raise ValueError(f"scoring must be one of {sorted(prompts.SCORING_RULES)}, got {scoring!r}")
        super().__init__(power, scoring, identity=identity or provider.model)
        self.provider = provider
        self.config = config
        self.output_limit = output_limit
        self.effort = effort
        self.memory_window = memory_window
        # Once over the window, trim down to this fraction of it. Any trim changes the start
        # of the conversation and so invalidates the provider's cache; trimming in one big
        # step (not one turn per call) keeps those full cache rewrites rare.
        self.trim_to = trim_to
        # Optional per-model instruction appended to the system prompt, for models whose
        # thinking length can't be set any other way (e.g. local reasoning models).
        self.thinking_hint = thinking_hint
        self.diary_max_chars = diary_max_chars
        self.parse_retries = parse_retries
        self.turns: list[dict] = []        # {"phase", "user", "assistant", "board_upto"}, oldest first
        self.board_log: list[dict] = []    # every completed phase shown so far (never trimmed)
        self.diary: list[list[str]] = []   # [phase, entry], never trimmed
        self.trimmed_through = 0           # number of oldest turns replaced by the digest
        self.trims: list[dict] = []        # log of each trim
        self.seen_history = 0              # completed phases already shown
        self.seen_messages = 0             # visible messages already shown
        self.seen_rejected = 0             # own undelivered messages already reported
        # Index of the turn that last showed the board / legal orders / votes, and what was shown,
        # so each is sent only when new (and again if trimming removed it).
        self.shown: dict[str, list] = {}   # key -> [turn_index, value]
        self._system: str | None = None
        self.errors: list[str] = []

    # ---------- checkpoint state ----------

    def get_state(self) -> dict:
        return {"turns": self.turns, "board_log": self.board_log, "diary": self.diary,
                "trimmed_through": self.trimmed_through,
                "trims": self.trims, "seen_history": self.seen_history, "seen_messages": self.seen_messages,
                "shown": self.shown, "seen_rejected": self.seen_rejected, "errors": self.errors}

    def set_state(self, state: dict):
        self.turns = state.get("turns", [])
        self.board_log = state.get("board_log", [])
        self.diary = state.get("diary", [])
        self.trimmed_through = state.get("trimmed_through", 0)
        self.trims = state.get("trims", [])
        self.seen_history = state.get("seen_history", 0)
        self.seen_messages = state.get("seen_messages", 0)
        self.shown = state.get("shown", {})
        self.seen_rejected = state.get("seen_rejected", 0)
        self.errors = state.get("errors", [])

    # ---------- memory ----------

    def _system_prompt(self, ctx: PhaseContext) -> str:
        # Built once: disclosed identities are fixed for the game, so the prompt stays cacheable.
        if self._system is None:
            self._system = prompts.system_prompt(self.power, self.scoring, self.config, self.output_limit,
                                                 ctx.opponent_identities, self.memory_window, self.diary_max_chars)
            if self.thinking_hint:
                self._system += "\n\nTHINKING: " + self.thinking_hint
        return self._system

    def _messages(self) -> list[dict]:
        """The conversation as sent: a digest of trimmed turns (if any), then the kept turns."""
        out = []
        if self.trimmed_through:
            gone = self.turns[:self.trimmed_through]
            gone_phases = {t["phase"] for t in gone}
            entries = [(p, e) for p, e in self.diary if p in gone_phases]
            board = self.board_log[:gone[-1]["board_upto"]]
            out.append({"role": "user", "content": prompts.memory_digest(entries, board, gone[0]["phase"],
                                                                        gone[-1]["phase"])})
            out.append({"role": "assistant", "content": "Understood."})
        for t in self.turns[self.trimmed_through:]:
            out.append({"role": "user", "content": t["user"]})
            out.append({"role": "assistant", "content": t["assistant"]})
        return out

    def _fit(self, system: str, prompt: str, phase: str):
        """If system + memory + new turn + reply exceed the window, trim the oldest turns until
        they fit in `trim_to` of it."""
        def size():
            return (count_tokens(system) + sum(count_tokens(m["content"]) for m in self._messages())
                    + count_tokens(prompt) + self.output_limit)
        dropped = 0
        if size() > self.memory_window:
            target = int(self.memory_window * self.trim_to)
            while size() > target and self.trimmed_through < len(self.turns):
                self.trimmed_through += 1
                dropped += 1
        if dropped:
            self.trims.append({"phase": phase, "dropped_turns": dropped, "total_trimmed": self.trimmed_through,
                               "over_window": size() > self.memory_window})
            if size() > self.memory_window:
                # Everything trimmable is gone; the board record and diary are kept even so.
                self.errors.append(f"{phase}: memory over window ({size():,} > {self.memory_window:,} tokens)")

    # ---------- calls ----------

    def _turn(self, ctx: PhaseContext, build, schema: dict, purpose: str) -> dict | None:
        """One conversational turn (plus a corrective turn if the reply isn't valid JSON).
        `build(show)` renders the prompt; it is re-rendered if trimming removed something
        (the board, legal orders, votes or shares) that the first rendering pointed back to."""
        system = self._system_prompt(ctx)
        show = self._what_to_show(ctx)
        prompt = build(show)
        before = self.trimmed_through
        self._fit(system, prompt, ctx.phase)
        if self.trimmed_through != before:
            # Keep everything this turn already shows, and add whatever the trim removed.
            again = self._what_to_show(ctx)
            for key, flag in (("board", "show_board"), ("legal", "show_legal"), ("votes", "show_votes"),
                              ("shares", "show_shares")):
                if show[flag] and key in self.shown:
                    self.shown[key][0] = len(self.turns)
            prompt = build({k: show[k] or again[k] for k in show})
        for attempt in range(self.parse_retries + 1):
            if attempt:
                self._fit(system, prompt, ctx.phase)
            req = LLMRequest(system=system, prompt=prompt, schema=schema, history=self._messages(),
                             max_output_tokens=self.output_limit, effort=self.effort,
                             game_id=ctx.game_id, power=self.power, phase=ctx.phase,
                             purpose=purpose if attempt == 0 else f"{purpose}_parse{attempt}")
            result = self.provider.complete(req)
            # "reasoning" is stored for analysis only: _messages() never sends it back.
            self.turns.append({"phase": ctx.phase, "user": prompt, "assistant": result.text or NO_REPLY,
                               "board_upto": len(self.board_log), "reasoning": result.reasoning})
            if result.data is not None:
                return result.data
            self.errors.append(f"{ctx.phase} {purpose}: {result.record.error}")
            prompt = (f"Your last reply could not be used ({result.record.error}). Reply again with a single "
                      "JSON object matching the requested schema.")
        return None

    def _needs(self, key: str, value) -> bool:
        """True if `value` (e.g. this phase's board) isn't visible in the kept conversation yet."""
        last = self.shown.get(key)
        return last is None or last[1] != value or last[0] < self.trimmed_through

    def _mark(self, key: str, value):
        self.shown[key] = [len(self.turns), value]  # the turn about to be appended

    def _what_to_show(self, ctx: PhaseContext) -> dict:
        votes = None if ctx.draw_votes is None else sorted(ctx.draw_votes.items())
        votes_key = [ctx.votes_held, ctx.my_draw_vote, votes]
        # Draw shares depend only on who owns how many centers.
        shares_key = sorted([p, len(cs), bool(ctx.units[p])] for p, cs in ctx.centers.items())
        show = {"show_board": self._needs("board", ctx.phase), "show_legal": self._needs("legal", ctx.phase),
                "show_votes": self._needs("votes", votes_key), "show_shares": self._needs("shares", shares_key)}
        for key, value, flag in (("board", ctx.phase, "show_board"), ("legal", ctx.phase, "show_legal"),
                                 ("votes", votes_key, "show_votes"), ("shares", shares_key, "show_shares")):
            if show[flag]:
                self._mark(key, value)
        return show

    def _new_since_last_turn(self, ctx: PhaseContext):
        new_history = ctx.history[self.seen_history:]
        # Own messages are already in the agent's replies, so only others' messages are new.
        new_messages = [m for m in ctx.messages[self.seen_messages:] if m.sender != self.power]
        undelivered = ctx.rejected_messages[self.seen_rejected:]
        first = not self.turns
        self.board_log.extend(new_history)
        self.seen_history, self.seen_messages = len(ctx.history), len(ctx.messages)
        self.seen_rejected = len(ctx.rejected_messages)
        return new_history, new_messages, first, undelivered

    def negotiate(self, ctx: PhaseContext) -> list[OutgoingMessage]:
        new_history, new_messages, first, undelivered = self._new_since_last_turn(ctx)
        build = lambda show: prompts.negotiate_prompt(ctx, new_history, new_messages, first, **show,  # noqa: E731
                                                      scoring=self.scoring, undelivered=undelivered)
        data = self._turn(ctx, build, prompts.TURN_SCHEMA, f"negotiate_r{ctx.negotiation_round}")
        if not data:
            return []
        out = []
        for m in data.get("messages") or []:
            if isinstance(m, dict) and isinstance(m.get("text"), str) and m.get("recipients"):
                out.append(OutgoingMessage(list(m["recipients"]), m["text"]))
        return out

    def orders(self, ctx: PhaseContext) -> OrderSubmission:
        new_history, new_messages, first, undelivered = self._new_since_last_turn(ctx)
        build = lambda show: prompts.orders_prompt(ctx, new_history, new_messages, first,  # noqa: E731
                                                   diary_max_chars=self.diary_max_chars, **show,
                                                   scoring=self.scoring, undelivered=undelivered)
        data = self._turn(ctx, build, prompts.TURN_SCHEMA, "orders" if not ctx.feedback else "orders_retry")
        if not data:
            return OrderSubmission([], None)  # the loop rejects it and retries with feedback
        diary = data.get("diary")
        if isinstance(diary, str) and diary.strip():
            entry = diary.strip()[: self.diary_max_chars]
            if self.diary and self.diary[-1][0] == ctx.phase:
                self.diary[-1][1] = entry  # a resubmission replaces this phase's entry
            else:
                self.diary.append([ctx.phase, entry])
        vote = data.get("draw_vote")
        orders = [o for o in data.get("orders") or [] if isinstance(o, str)]
        return OrderSubmission(orders, vote if isinstance(vote, bool) else None)
