"""Zero-cost scripted agent for smoke tests."""
from __future__ import annotations

import random

from ..state import PhaseContext
from .base import Agent, OrderSubmission, OutgoingMessage


class RandomAgent(Agent):
    name = "random"

    def __init__(self, power: str, scoring: str = "dss", seed: int = 0,
                 draw_prob: float = 0.0, invalid_prob: float = 0.0, chatty: bool = True, **kwargs):
        super().__init__(power, scoring, **kwargs)
        self.rng = random.Random(f"{seed}-{power}")
        self.draw_prob = draw_prob
        self.invalid_prob = invalid_prob
        self.chatty = chatty

    def get_state(self) -> dict:
        version, internal, gauss = self.rng.getstate()
        return {"rng": [version, list(internal), gauss]}

    def set_state(self, state: dict):
        if "rng" in state:
            version, internal, gauss = state["rng"]
            self.rng.setstate((version, tuple(internal), gauss))

    def negotiate(self, ctx: PhaseContext) -> list[OutgoingMessage]:
        if not self.chatty:
            return []
        others = [p for p, units in ctx.units.items() if p != self.power and (units or ctx.centers[p])]
        if not others:
            return []
        if self.rng.random() < 0.3:
            targets = ["GLOBAL"]
        else:
            targets = self.rng.sample(others, self.rng.randint(1, min(3, len(others))))
        return [OutgoingMessage(targets, f"{self.power} to {'+'.join(targets)}, round {ctx.negotiation_round}: hello")]

    def orders(self, ctx: PhaseContext) -> OrderSubmission:
        vote = self.rng.random() < self.draw_prob
        # Exercise the retry path: on the first attempt only, send a bogus order and no vote.
        if not ctx.feedback and self.rng.random() < self.invalid_prob:
            return OrderSubmission(["A XXX - YYY"], None)
        locs = list(ctx.legal_orders)
        if ctx.phase_type == "A":
            self.rng.shuffle(locs)
            locs = locs[:abs(ctx.adjustment_delta)]
        return OrderSubmission([self.rng.choice(ctx.legal_orders[loc]) for loc in locs], vote)
