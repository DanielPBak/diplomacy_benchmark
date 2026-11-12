"""Agent interface. One agent instance plays one power for one game."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..state import PhaseContext


@dataclass
class OutgoingMessage:
    """A message to one power, a group of powers, or everyone (["GLOBAL"]). Every
    recipient sees the full recipient list. A single name may be passed as a string."""
    recipients: list[str] | str
    text: str

    def __post_init__(self):
        if isinstance(self.recipients, str):
            self.recipients = [self.recipients]


@dataclass
class OrderSubmission:
    """One phase's submission. In movement and Winter phases (`ctx.vote_phase`) every
    survivor submits, even with no orders, and `draw_vote` is required: True accepts a
    draw shared by all current survivors. Retreat phases take no vote. Ballots are public
    unless `GameConfig.secret_ballots` is set."""
    orders: list[str] = field(default_factory=list)
    draw_vote: bool | None = None


class Agent(ABC):
    name = "agent"

    def __init__(self, power: str, scoring: str = "dss", identity: str | None = None, **kwargs):
        self.power = power
        self.scoring = scoring
        # Label other agents may be shown, e.g. "Claude Haiku 5.5 (Anthropic)".
        self.identity = identity or self.name

    def get_state(self) -> dict:
        """JSON-serializable state saved in checkpoints (e.g. notes, RNG state)."""
        return {}

    def set_state(self, state: dict):
        """Restore what get_state() returned, when resuming a game."""

    def negotiate(self, ctx: PhaseContext) -> list[OutgoingMessage]:
        """Full and public press, every phase. Never called in gunboat. In public press,
        only GLOBAL messages are delivered."""
        return []

    @abstractmethod
    def orders(self, ctx: PhaseContext) -> OrderSubmission:
        """Return this phase's orders plus a draw vote. `ctx.feedback` is set on a retry."""
