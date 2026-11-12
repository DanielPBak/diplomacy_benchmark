"""Game loop: unmodified engine + 7 pluggable agents, full press or gunboat."""
from __future__ import annotations

import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from diplomacy import Game
from diplomacy.engine.message import Message as EngineMessage
from diplomacy.utils import strings
from diplomacy.utils.export import from_saved_game_format, to_saved_game_format

from .agents.base import Agent
from .config import POWERS, GameConfig
from .tokens import CHARS_PER_TOKEN, count_tokens
from .state import build_context, center_counts, current_year, order_location, survivors

log = logging.getLogger(__name__)

OUTCOMES = ("solo", "draw_vote", "year_cap", "engine_draw")
CHECKPOINT_VERSION = 1


class PressViolation(RuntimeError):
    """Raised if anything tries to create a message in a gunboat game."""


@dataclass
class GameResult:
    game_id: str
    press_mode: str
    outcome: str                 # one of OUTCOMES
    winners: list[str]           # solo winner, or the draw participants
    final_centers: dict[str, int]
    final_phase: str
    final_year: int
    survivors: list[str]
    max_year: int
    announced_max_year: int | None = None  # the end year agents were told, if different
    # Center counts and survivors at each year close (after Winter), plus the final year.
    # Used by analysis for growth curves and elimination timing.
    year_centers: dict[int, dict[str, int]] = field(default_factory=dict)
    year_survivors: dict[int, list[str]] = field(default_factory=dict)
    n_messages: int = 0
    n_rejected_messages: int = 0  # see rejected_press.jsonl for reasons
    press_tokens_sent: dict[str, int] = field(default_factory=dict)  # model-neutral tokens
    identity_disclosure: str = "hidden"
    identities_true: dict[str, str] = field(default_factory=dict)
    identities_shown: dict[str, str] | None = None  # what agents saw; None when hidden
    invalid_order_attempts: dict[str, int] = field(default_factory=dict)
    resumes: int = 0  # times this game was resumed from a checkpoint

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_order(order: str) -> str:
    return " ".join(str(order).upper().split())


def validate_orders(legal: dict[str, list[str]], phase_type: str, delta: int,
                    orders: list[str]) -> tuple[list[str], list[str]]:
    """Split submitted orders into (valid, errors). Never raises."""
    legal_set = {o for opts in legal.values() for o in opts}
    valid, errors, used_locs = [], [], set()
    for raw in orders:
        order = normalize_order(raw)
        if order not in legal_set:
            errors.append(f"Illegal order: {order!r}")
            continue
        loc = order_location(order)
        key = loc.split("/")[0] if loc else f"WAIVE{len(valid)}"
        if key in used_locs:
            errors.append(f"More than one order for {loc}: {order!r}")
            continue
        if phase_type == "A" and len(valid) >= abs(delta):
            errors.append(f"Too many adjustment orders; you may give {abs(delta)}: {order!r}")
            continue
        used_locs.add(key)
        valid.append(order)
    return valid, errors


class GameRunner:
    def __init__(self, config: GameConfig, agents: dict[str, Agent], checkpoint_path: str | Path | None = None):
        """With `checkpoint_path`, the full game state is saved there at the start of every
        phase, so an interrupted game can be continued with GameRunner.resume()."""
        assert set(agents) == set(POWERS), "need exactly one agent per power"
        self.config = config
        self.agents = agents
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.resumes = 0
        self.game = Game(game_id=config.game_id)
        if config.is_gunboat:
            self.game.add_rule("NO_PRESS")
        else:
            self.game.remove_rule("NO_PRESS")
            if config.is_public_press:
                self.game.add_rule("PUBLIC_PRESS")
        self.identities_true = {p: agents[p].identity for p in POWERS}
        self.identities_shown = shown_identities(self.identities_true, config.identity_disclosure,
                                                 random.Random(f"identities-{config.seed}"))
        self.press_log: list[dict] = []
        self.rejected_press: list[dict] = []
        # Press tokens spent per (power, budget window).
        self.press_spent: dict[tuple, int] = {}
        self.vote_log: list[dict] = []
        self.order_log: list[dict] = []
        self.year_centers: dict[int, dict[str, int]] = {}
        self.year_survivors: dict[int, list[str]] = {}
        self.invalid_attempts = {p: 0 for p in POWERS}
        # Ballots from the most recent draw vote (shown to agents when ballots are public).
        self.last_ballots: dict[str, bool] = {}  # empty until the first vote
        self.last_year_closed = self.game.map.first_year - 1
        self.pool = ThreadPoolExecutor(max_workers=config.max_workers)

    # ---------- checkpoints ----------

    def checkpoint_state(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "engine": to_saved_game_format(self.game),
            "rules": sorted(self.game.rules),  # the engine export drops some rules
            "phase": self.game.get_current_phase(),
            "last_year_closed": self.last_year_closed,
            "identities_true": self.identities_true,
            "identities_shown": self.identities_shown,
            "press_log": self.press_log,
            "rejected_press": self.rejected_press,
            "press_spent": [[list(k), v] for k, v in self.press_spent.items()],
            "vote_log": self.vote_log,
            "order_log": self.order_log,
            "year_centers": self.year_centers,
            "year_survivors": self.year_survivors,
            "invalid_attempts": self.invalid_attempts,
            "last_ballots": self.last_ballots,
            "agents": {p: a.get_state() for p, a in self.agents.items()},
            "resumes": self.resumes,
        }

    def save_checkpoint(self):
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.checkpoint_state()))
        os.replace(tmp, self.checkpoint_path)  # atomic: a crash never leaves half a checkpoint

    @classmethod
    def resume(cls, checkpoint_path: str | Path, agents: dict[str, Agent],
               max_workers: int | None = None) -> "GameRunner":
        """Rebuild a game from its last checkpoint. The current phase restarts from its
        beginning; the agents must be the same models as before (checked by identity).
        `max_workers` (concurrent agent calls) may be changed: it doesn't affect the game."""
        state = json.loads(Path(checkpoint_path).read_text())
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version {state.get('version')}")
        config = GameConfig(**state["config"])
        if max_workers is not None:
            config.max_workers = max_workers
        runner = cls(config, agents, checkpoint_path)
        mismatched = {p: (state["identities_true"][p], runner.identities_true[p]) for p in POWERS
                      if state["identities_true"][p] != runner.identities_true[p]}
        if mismatched:
            raise ValueError(f"agents differ from the checkpointed game: {mismatched}")
        game = from_saved_game_format(state["engine"])
        for rule in set(game.rules) - set(state["rules"]):
            game.remove_rule(rule)
        for rule in set(state["rules"]) - set(game.rules):
            game.add_rule(rule)
        assert game.get_current_phase() == state["phase"]
        runner.game = game
        runner.last_year_closed = state["last_year_closed"]
        runner.identities_shown = state["identities_shown"]
        runner.press_log = state["press_log"]
        runner.rejected_press = state["rejected_press"]
        runner.press_spent = {tuple(k): v for k, v in state["press_spent"]}
        runner.vote_log = state["vote_log"]
        runner.order_log = state["order_log"]
        runner.year_centers = {int(y): c for y, c in state["year_centers"].items()}
        runner.year_survivors = {int(y): s for y, s in state["year_survivors"].items()}
        runner.invalid_attempts = state["invalid_attempts"]
        runner.last_ballots = state["last_ballots"]
        for p, a in agents.items():
            a.set_state(state["agents"].get(p, {}))
        runner.resumes = state["resumes"] + 1
        return runner

    # ---------- helpers ----------

    def _parallel(self, fn, powers):
        return dict(zip(powers, self.pool.map(fn, powers)))

    def _ctx(self, power, negotiation_round=None):
        return build_context(self.game, power, self.config, self.last_ballots, self.press_log,
                             shown_identities=self.identities_shown, vote_phase=self.is_vote_phase(),
                             negotiation_round=negotiation_round,
                             press_budget=self._budget_view(power, negotiation_round or 0),
                             rejected_press=self.rejected_press)

    def _budget_window(self, power: str, rnd: int) -> tuple:
        scope = self.config.press_budget_scope
        phase = self.game.current_short_phase
        return (power,) if scope == "game" else (power, phase) if scope == "phase" else (power, phase, rnd)

    def _budget_remaining(self, power: str, rnd: int) -> int | None:
        if self.config.press_token_budget is None:
            return None
        return self.config.press_token_budget - self.press_spent.get(self._budget_window(power, rnd), 0)

    def _budget_view(self, power: str, rnd: int) -> dict | None:
        if self.config.press_token_budget is None:
            return None
        return {"limit": self.config.press_token_budget, "remaining": self._budget_remaining(power, rnd),
                "scope": self.config.press_budget_scope,
                "unit": f"tokens of message text, 1 token = {CHARS_PER_TOKEN} characters"}

    def is_vote_phase(self) -> bool:
        """Draw votes are taken in movement and Winter phases, never in retreats."""
        return self.config.votes_count(current_year(self.game)) and self.game.phase_type in ("M", "A")

    def _send(self, sender: str, recipients: list[str], text: str, rnd: int):
        if self.config.is_gunboat:
            raise PressViolation(f"{sender} tried to send a message in a gunboat game")
        if self.config.is_public_press and recipients != ["GLOBAL"]:
            raise PressViolation(f"{sender} tried to send a private message in a public-press game")
        phase = self.game.current_short_phase
        # The engine's Message has a single recipient, so a group message is stored as one
        # copy per recipient. The press log keeps it as one message with its recipient list.
        for recipient in recipients:
            self.game.add_message(EngineMessage(sender=sender, recipient=recipient, message=text, phase=phase))
        entry = {"id": len(self.press_log), "phase": phase, "round": rnd, "sender": sender,
                 "recipients": list(recipients), "text": text, "tokens": count_tokens(text)}
        self.press_log.append(entry)
        if self.checkpoint_path:
            # Live view for watching a game in progress. press.jsonl (written at the end) is
            # the record; this file may also hold messages from a phase that was later re-run.
            with open(self.checkpoint_path.parent / "press_live.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _reject(self, sender: str, rnd: int, recipients, text: str, reason: str):
        self.rejected_press.append({"phase": self.game.current_short_phase, "round": rnd, "sender": sender,
                                    "recipients": list(recipients), "text": text, "reason": reason})

    def _route(self, sender: str, msg, alive: list[str], index: int, rnd: int):
        """Validate one outgoing message and send it, logging anything rejected or changed."""
        cfg = self.config
        text = str(msg.text)
        asked = list(dict.fromkeys(str(r).upper() for r in msg.recipients))  # dedupe, keep order
        if cfg.max_messages_per_round is not None and index >= cfg.max_messages_per_round:
            return self._reject(sender, rnd, asked, text, f"over max_messages_per_round={cfg.max_messages_per_round}")
        if "GLOBAL" in asked:
            recipients = ["GLOBAL"]
        else:
            recipients = [r for r in asked if r in alive and r != sender]
            bad = [r for r in asked if r not in recipients]
            if bad:
                self._reject(sender, rnd, bad, text, "not a surviving other power")
            if not recipients:
                return
        if cfg.is_public_press and recipients != ["GLOBAL"]:
            return self._reject(sender, rnd, recipients, text, "public press: only GLOBAL messages are allowed")
        if not cfg.allow_group_messages and len(recipients) > 1:
            return self._reject(sender, rnd, recipients, text,
                                "group messages are off: send to one power or to GLOBAL")
        if cfg.max_message_chars is not None and len(text) > cfg.max_message_chars:
            self._reject(sender, rnd, recipients, text, f"truncated to max_message_chars={cfg.max_message_chars}")
            text = text[: cfg.max_message_chars]
        cost = count_tokens(text)
        remaining = self._budget_remaining(sender, rnd)
        if remaining is not None and cost > remaining:
            return self._reject(sender, rnd, recipients, text,
                                f"over press token budget: costs {cost}, {remaining} left this {cfg.press_budget_scope}")
        window = self._budget_window(sender, rnd)
        self.press_spent[window] = self.press_spent.get(window, 0) + cost
        self._send(sender, recipients, text, rnd)

    # ---------- phase steps ----------

    def negotiate(self):
        alive = survivors(self.game)
        for rnd in range(self.config.negotiation_rounds):
            outgoing = self._parallel(
                lambda p: self.agents[p].negotiate(self._ctx(p, negotiation_round=rnd)), alive)
            # Messages in a round are simultaneous: delivered only after every power has spoken.
            for sender in alive:
                for index, msg in enumerate(outgoing[sender] or []):
                    self._route(sender, msg, alive, index, rnd)

    def _collect_orders(self, power: str) -> tuple[list[str], bool | None, list[dict]]:
        """Runs in a worker thread, so log rows are returned, not appended (keeps the log order fixed)."""
        ctx = self._ctx(power)
        valid: list[str] = []
        vote = None
        rows: list[dict] = []
        for attempt in range(self.config.order_retries + 1):
            sub = self.agents[power].orders(ctx)
            valid, errors = validate_orders(ctx.legal_orders, ctx.phase_type, ctx.adjustment_delta, sub.orders or [])
            vote = sub.draw_vote if isinstance(sub.draw_vote, bool) else None
            if not ctx.vote_phase:
                vote = None  # retreat phase: any vote sent is ignored
            elif vote is None:
                errors.append("draw_vote is required: true to accept a draw among all survivors, false otherwise")
            rows.append({"phase": ctx.phase, "power": power, "attempt": attempt,
                         "submitted": list(sub.orders or []), "draw_vote": sub.draw_vote,
                         "valid": valid, "errors": errors})
            if not errors:
                return valid, vote, rows
            self.invalid_attempts[power] += 1
            ctx.feedback = errors
        # Out of retries: keep the legal subset. The engine holds unordered units, disbands
        # unordered retreats and civil-disorders missing disbands. A missing vote counts as no.
        return valid, (bool(vote) if ctx.vote_phase else None), rows

    def submit_orders(self) -> dict[str, bool] | None:
        """Collect this phase's submissions. In a vote phase every survivor submits, even with
        nothing to order (no units, or no builds/disbands), and the ballots are returned."""
        orderable = [p for p in POWERS if self.game.get_orderable_locations(p)]
        vote_phase = self.is_vote_phase()
        callers = survivors(self.game) if vote_phase else orderable
        ballots = {}
        for power, (orders, vote, rows) in self._parallel(self._collect_orders, callers).items():
            self.order_log.extend(rows)
            if power in orderable:
                self.game.set_orders(power, orders)
            if vote_phase:
                ballots[power] = vote
        return ballots if vote_phase else None

    def check_draw(self, ballots: dict[str, bool]) -> bool:
        """Unanimity over all survivors' ballots, decided by the engine's own vote fields."""
        alive = survivors(self.game)
        for p in alive:
            self.game.get_power(p).vote = strings.YES if ballots[p] else strings.NO
        passed = self.game.has_draw_vote()
        self.game.clear_vote()
        self.last_ballots.update(ballots)
        # All ballots are logged. Agents see others' votes only when ballots are public.
        self.vote_log.append({"phase": self.game.get_current_phase(), "survivors": alive,
                              "ballots": {p: ballots[p] for p in alive}, "passed": passed,
                              "centers": {p: len(self.game.get_power(p).centers) for p in alive}})
        return passed

    # ---------- main loop ----------

    def run(self) -> GameResult:
        game, cfg = self.game, self.config
        outcome = None
        try:
            while not game.is_game_done:
                # Everything up to here is saved, so an interruption anywhere in this phase
                # (spending cap, API failure, crash) loses only this phase's calls.
                self.save_checkpoint()
                year = current_year(game)
                # Reaching Spring of year Y closes year Y-1 (including its Winter).
                if year - 1 > self.last_year_closed:
                    self.last_year_closed = year - 1
                    if self.last_year_closed >= game.map.first_year:
                        self.year_centers[self.last_year_closed] = center_counts(game)
                        self.year_survivors[self.last_year_closed] = survivors(game)
                        if self.last_year_closed >= cfg.max_year:
                            game.draw(winners=survivors(game))
                            outcome = "year_cap"
                            break

                # Full and public press negotiate in every phase: movement, retreats and Winter.
                if not cfg.is_gunboat:
                    self.negotiate()
                ballots = self.submit_orders()
                # A unanimous draw ends the game before this phase's orders are adjudicated.
                if ballots is not None and self.check_draw(ballots):
                    game.draw(winners=survivors(game))
                    outcome = "draw_vote"
                    break
                game.process()
            if outcome is None:
                # Engine-ended game (solo): the final year counts as closed.
                self.year_centers[current_year_from_outcome(game)] = center_counts(game)
                self.year_survivors[current_year_from_outcome(game)] = survivors(game)
        finally:
            # Agents that wait on a person (HumanAgent) block a worker thread until they reply;
            # closing them releases it so an interrupted game can shut down.
            for agent in self.agents.values():
                if hasattr(agent, "close"):
                    agent.close()
            self.pool.shutdown(wait=True)

        if outcome is None:
            # Engine finished the game itself: an 18-center solo, or its 100-year draw.
            winners = game.outcome[1:]
            outcome = "solo" if len(winners) == 1 and center_counts(game)[winners[0]] >= game.win else "engine_draw"
        if cfg.is_gunboat and (self.press_log or count_messages(game)):
            raise PressViolation("messages exist in a gunboat game")
        if cfg.is_public_press and any(m["recipients"] != ["GLOBAL"] for m in self.press_log):
            raise PressViolation("private messages exist in a public-press game")

        return GameResult(
            game_id=cfg.game_id, press_mode=cfg.press_mode, outcome=outcome,
            winners=list(game.outcome[1:]), final_centers=center_counts(game),
            final_phase=game.outcome[0] if game.outcome else game.get_current_phase(),
            final_year=current_year_from_outcome(game), survivors=survivors(game),
            max_year=cfg.max_year, announced_max_year=cfg.announced_max_year,
            year_centers=self.year_centers, year_survivors=self.year_survivors,
            n_messages=len(self.press_log), n_rejected_messages=len(self.rejected_press),
            press_tokens_sent={p: sum(m["tokens"] for m in self.press_log if m["sender"] == p) for p in POWERS},
            identity_disclosure=cfg.identity_disclosure, identities_true=self.identities_true,
            identities_shown=self.identities_shown,
            invalid_order_attempts=self.invalid_attempts, resumes=self.resumes,
        )

    def save(self, out_dir: str | Path, result: GameResult):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "game.json", "w") as f:
            json.dump(to_saved_game_format(self.game), f)
        with open(out / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        # Each agent's final memory: LLM conversation, diary and memory trims (for analysis of
        # why agents did what they did; the checkpoint holding them is removed after the game).
        with open(out / "agents.json", "w") as f:
            json.dump({p: {"identity": a.identity, **a.get_state()} for p, a in self.agents.items()}, f)
        _write_jsonl(out / "votes.jsonl", self.vote_log)
        _write_jsonl(out / "orders.jsonl", self.order_log)
        if not self.config.is_gunboat:
            _write_jsonl(out / "press.jsonl", self.press_log)
            _write_jsonl(out / "rejected_press.jsonl", self.rejected_press)


def shown_identities(true: dict[str, str], mode: str, rng: random.Random) -> dict[str, str] | None:
    """Labels agents see for the other powers. false_labels shuffles the true labels to
    maximize mismatches; when one label fills more than half the seats, some must stay true."""
    if mode == "hidden":
        return None
    if mode == "revealed":
        return dict(true)
    powers = list(true)
    best, best_wrong = None, -1
    for _ in range(500):
        labels = [true[p] for p in powers]
        rng.shuffle(labels)
        wrong = sum(label != true[p] for p, label in zip(powers, labels))
        if wrong > best_wrong:
            best, best_wrong = dict(zip(powers, labels)), wrong
    return best


def count_messages(game: Game) -> int:
    """All messages the engine holds, current phase and history."""
    return len(game.messages) + sum(len(game.message_history[p]) for p in game.message_history.keys())


def current_year_from_outcome(game: Game) -> int:
    phase = game.outcome[0] if game.outcome else game.get_current_phase()
    return int(phase[1:5])


def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_game(config: GameConfig, agents: dict[str, Agent], out_dir: str | Path | None = None,
             resume: bool = False) -> GameResult:
    """Play a game, saving outputs to `out_dir`. With an `out_dir`, a checkpoint is kept
    there during play; resume=True continues from it if one exists (the config then comes
    from the checkpoint). The checkpoint is removed once the game finishes."""
    checkpoint = Path(out_dir) / "checkpoint.json" if out_dir is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        runner = GameRunner.resume(checkpoint, agents)
    else:
        runner = GameRunner(config, agents, checkpoint)
    result = runner.run()
    if out_dir is not None:
        runner.save(out_dir, result)
        checkpoint.unlink(missing_ok=True)
    return result
