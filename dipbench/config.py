"""Game-level configuration shared by the loop, agents and runner."""
from __future__ import annotations

from dataclasses import dataclass, field

# full: private and broadcast press. public: broadcast only. gunboat: no press at all.
PRESS_MODES = ("full", "public", "gunboat")
PRESS_BUDGET_SCOPES = ("round", "phase", "game")
# What agents are told about which model plays each other power.
IDENTITY_MODES = ("hidden", "revealed", "false_labels")

POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")


@dataclass
class GameConfig:
    press_mode: str = "full"
    # Negotiation rounds per phase (movement, retreat and Winter). Ignored in gunboat.
    negotiation_rounds: int = 5
    # Press token budget per power: the message text it may send per `press_budget_scope`
    # (default: each negotiation round; unused budget doesn't carry over), counted
    # model-neutrally (dipbench.tokens: 1 token per 4 characters). A group or GLOBAL
    # message costs its length once. Agents are told the budget and what's left; a message
    # that doesn't fit is rejected and logged. None means unlimited. 300 tokens is about
    # 200 words: two or three short messages per round (5 rounds per phase by default).
    press_token_budget: int | None = 300
    press_budget_scope: str = "round"
    # Full press only: allow messages to a chosen group of powers. False = private one-to-one
    # messages and GLOBAL broadcasts only.
    allow_group_messages: bool = True
    # Optional caps, off by default. Anything over a cap is rejected (or, for length,
    # truncated) and logged with a reason, never dropped silently.
    max_messages_per_round: int | None = None
    max_message_chars: int | None = None
    # Last game-year played. The game ends as a "year_cap" draw after this year's Winter.
    max_year: int = 1920
    # The end year agents are told. None = max_year. Set it later than max_year to see how
    # agents play when they believe the game is long, while the run actually stops at max_year.
    announced_max_year: int | None = None
    # Every order submission, in every season, carries a required draw vote. The game ends
    # as a draw when all survivors' current votes are yes. Votes count from this year on.
    draw_votes: bool = True
    draw_vote_from_year: int = 1901
    # Public ballots (default): every agent sees all survivors' current votes. Secret: an
    # agent sees only its own vote.
    secret_ballots: bool = False
    # hidden: agents see only power names. revealed: each other power's true model label.
    # false_labels: other powers' labels, shuffled so they are (as far as possible) wrong.
    identity_disclosure: str = "hidden"
    # Retries after the first order submission is rejected.
    order_retries: int = 2
    max_workers: int = 7
    seed: int = 0
    game_id: str = "game"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.press_mode not in PRESS_MODES:
            raise ValueError(f"press_mode must be one of {PRESS_MODES}, got {self.press_mode!r}")
        if self.press_budget_scope not in PRESS_BUDGET_SCOPES:
            raise ValueError(f"press_budget_scope must be one of {PRESS_BUDGET_SCOPES}, "
                             f"got {self.press_budget_scope!r}")
        if self.announced_max_year is not None and self.announced_max_year < self.max_year:
            raise ValueError("announced_max_year can't be earlier than max_year")
        if self.announced_max_year is not None and self.announced_max_year > 1999:
            raise ValueError("announced_max_year must be 1999 or earlier (the engine draws at 100 years)")
        if self.identity_disclosure not in IDENTITY_MODES:
            raise ValueError(f"identity_disclosure must be one of {IDENTITY_MODES}, "
                             f"got {self.identity_disclosure!r}")

    @property
    def told_max_year(self) -> int:
        return self.announced_max_year or self.max_year

    @property
    def is_gunboat(self) -> bool:
        return self.press_mode == "gunboat"

    @property
    def is_public_press(self) -> bool:
        return self.press_mode == "public"

    def votes_count(self, year: int) -> bool:
        return self.draw_votes and year >= self.draw_vote_from_year
