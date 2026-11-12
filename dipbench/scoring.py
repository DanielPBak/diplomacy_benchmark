"""Scoring systems, all computed from a finished game's final state.

Every scorer takes a `GameResult` (or its `to_dict()` form, as saved in result.json) and
returns a score for all seven powers. A solo always scores 1 for the winner and 0 for
everyone else in every system. Draw outcomes (`draw_vote`,
`year_cap`, `engine_draw`) are scored identically; `year_cap` is only flagged separately
in analysis.
"""
from __future__ import annotations

from .config import POWERS

TOTAL_CENTERS = 34
SYSTEMS = ("dss", "sos", "linear", "linear_norm")
# Systems an agent can be told it is playing to.
CONDITIONS = ("dss", "sos", "linear")
DRAW_OUTCOMES = ("draw_vote", "year_cap", "engine_draw")


def _get(result, key):
    return result[key] if isinstance(result, dict) else getattr(result, key)


def _parts(result) -> tuple[str, list[str], dict[str, int]]:
    outcome = _get(result, "outcome")
    if outcome != "solo" and outcome not in DRAW_OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    winners = list(_get(result, "winners"))
    if outcome == "solo" and len(winners) != 1:
        raise ValueError(f"solo needs exactly one winner, got {winners}")
    centers = {p: int(_get(result, "final_centers").get(p, 0)) for p in POWERS}
    return outcome, winners, centers


def _solo(winners) -> dict[str, float]:
    return {p: (1.0 if p in winners else 0.0) for p in POWERS}


def dss(result) -> dict[str, float]:
    """Draw-size scoring. Solo: 1 to the winner. Draw: 1/n to each of the n participants."""
    outcome, winners, _ = _parts(result)
    if outcome == "solo":
        return _solo(winners)
    share = 1.0 / len(winners)
    return {p: (share if p in winners else 0.0) for p in POWERS}


def sos(result) -> dict[str, float]:
    """Sum-of-squares. Solo: 1 to the winner. Draw: SC_i^2 / sum of SC_j^2 over participants."""
    outcome, winners, centers = _parts(result)
    if outcome == "solo":
        return _solo(winners)
    denom = sum(centers[p] ** 2 for p in winners)
    return {p: (centers[p] ** 2 / denom if p in winners and denom else 0.0) for p in POWERS}


def linear(result) -> dict[str, float]:
    """Solo: 1 to the winner. Otherwise SC_i / 34."""
    outcome, winners, centers = _parts(result)
    if outcome == "solo":
        return _solo(winners)
    return {p: centers[p] / TOTAL_CENTERS for p in POWERS}


def linear_norm(result) -> dict[str, float]:
    """Solo: 1 to the winner. Otherwise SC_i / sum of survivors' SCs (sums to 1)."""
    outcome, winners, centers = _parts(result)
    if outcome == "solo":
        return _solo(winners)
    alive = set(_get(result, "survivors"))
    denom = sum(centers[p] for p in alive)
    return {p: (centers[p] / denom if p in alive and denom else 0.0) for p in POWERS}


SCORERS = {"dss": dss, "sos": sos, "linear": linear, "linear_norm": linear_norm}


def score_all(result) -> dict[str, dict[str, float]]:
    return {name: fn(result) for name, fn in SCORERS.items()}
