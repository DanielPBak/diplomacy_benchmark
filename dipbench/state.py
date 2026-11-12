"""Read-only views of the engine state handed to agents.

Agents never get the engine `Game` object, only these plain-data snapshots, so a
power can't see other powers' private press or ballots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from diplomacy import Game

from .config import GameConfig


def survivors(game: Game) -> list[str]:
    return [name for name, power in game.powers.items() if not power.is_eliminated()]


def center_counts(game: Game) -> dict[str, int]:
    return {name: len(power.centers) for name, power in game.powers.items()}


def current_year(game: Game) -> int:
    return int(game.get_current_phase()[1:5])


def order_location(order: str) -> str | None:
    """Location an order is attached to ("A PAR - BUR" -> "PAR"); None for WAIVE."""
    parts = order.split()
    if len(parts) < 2 or parts[0] == "WAIVE":
        return None
    return parts[1]


@dataclass
class Message:
    phase: str
    round: int
    sender: str
    recipients: list[str]  # one or more power names, or ["GLOBAL"]
    text: str


@dataclass
class PhaseContext:
    game_id: str
    power: str
    phase: str           # short phase, e.g. "S1901M"
    phase_type: str      # "M", "R" or "A"
    year: int
    press_mode: str
    units: dict[str, list[str]]
    centers: dict[str, list[str]]
    legal_orders: dict[str, list[str]]  # this power's orderable locations only
    # Movement/retreat: number of units. Adjustment: builds (>0) or disbands (<0).
    adjustment_delta: int
    history: list[dict]                 # previous phases: orders, results, centers
    supply_centers: list[str] = field(default_factory=list)  # all 34 on the map
    messages: list[Message] = field(default_factory=list)  # always empty in gunboat
    rejected_messages: list[dict] = field(default_factory=list)  # this power's undelivered messages
    vote_phase: bool = False          # True in movement/Winter: draw_vote is required
    votes_held: bool = False          # has any draw vote happened yet?
    my_draw_vote: bool = False        # this power's ballot in the most recent vote
    draw_votes: dict[str, bool] | None = None  # survivors' most recent ballots; None if secret
    # Press token budget (full/public press): {"limit", "remaining", "scope", "unit"};
    # None when unlimited or gunboat.
    press_budget: dict | None = None
    # Model label shown for each other power; None when identities are hidden.
    opponent_identities: dict[str, str] | None = None
    negotiation_round: int | None = None
    feedback: list[str] = field(default_factory=list)       # errors from a rejected attempt
    config: GameConfig | None = None


def legal_orders_for(game: Game, power: str) -> dict[str, list[str]]:
    # The engine builds these from sets; sort so seeded games and prompts are reproducible.
    possible = game.get_all_possible_orders()
    return {loc: sorted(possible[loc]) for loc in sorted(game.get_orderable_locations(power)) if possible.get(loc)}


def adjustment_delta(game: Game, power: str) -> int:
    p = game.get_power(power)
    if game.phase_type == "A":
        return len(p.centers) - len(p.units)
    if game.phase_type == "R":
        return len(p.retreats)
    return len(p.units)


def build_history(game: Game) -> list[dict]:
    out = []
    for phase in game.state_history.keys():
        state = game.state_history[phase]
        out.append({
            "phase": str(phase),
            "orders": dict(game.order_history.get(phase, {})),
            "results": {unit: [str(r) for r in res]
                        for unit, res in game.result_history.get(phase, {}).items()},
            "centers": state["centers"],
        })
    return out


def visible_messages(press_log: list[dict], power: str) -> list[Message]:
    """Messages this power sent or received, in order, with full recipient lists."""
    return [Message(phase=m["phase"], round=m["round"], sender=m["sender"],
                    recipients=list(m["recipients"]), text=m["text"])
            for m in press_log
            if m["sender"] == power or m["recipients"] == ["GLOBAL"] or power in m["recipients"]]


def build_context(game: Game, power: str, config: GameConfig, last_ballots: dict[str, bool],
                  press_log: list[dict], shown_identities: dict[str, str] | None = None,
                  vote_phase: bool = False, negotiation_round: int | None = None,
                  press_budget: dict | None = None, rejected_press: list[dict] | None = None) -> PhaseContext:
    return PhaseContext(
        game_id=config.game_id,
        power=power,
        phase=game.get_current_phase(),
        phase_type=game.phase_type,
        year=current_year(game),
        press_mode=config.press_mode,
        units={n: list(p.units) for n, p in game.powers.items()},
        centers={n: list(p.centers) for n, p in game.powers.items()},
        legal_orders=legal_orders_for(game, power),
        adjustment_delta=adjustment_delta(game, power),
        history=build_history(game),
        supply_centers=sorted(game.map.scs),
        messages=[] if config.is_gunboat else visible_messages(press_log, power),
        press_budget=None if config.is_gunboat else press_budget,
        rejected_messages=[r for r in (rejected_press or []) if r["sender"] == power],
        vote_phase=vote_phase,
        votes_held=bool(last_ballots),
        my_draw_vote=last_ballots.get(power, False),
        draw_votes=None if config.secret_ballots else {p: last_ballots.get(p, False) for p in survivors(game)},
        opponent_identities=(None if shown_identities is None
                             else {p: label for p, label in shown_identities.items() if p != power}),
        negotiation_round=negotiation_round,
        config=config,
    )
