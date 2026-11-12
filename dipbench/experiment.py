"""Config-driven experiments: which models sit where, for how many games.

A config is a JSON file:

{
  "name": "haiku_vs_qwen",
  "models": {
    "haiku": {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "medium"},
    "qwen8": {"provider": "openai_compat", "model": "/path/to/qwen3-8b-8bit", "base_url": "http://127.0.0.1:8080/v1",
              "json_mode": "none", "identity": "Qwen3-8B (8-bit MLX, thinking)", "context_window": 34000}
  },
  "lineup": ["haiku", "qwen8", "qwen8", "qwen8", "qwen8", "qwen8", "qwen8"],   # or entries like
                                        # {"model": "haiku", "scoring": "sos"}; scoring follows the model
  "n_games": 14,
  "assignment": "balanced",             # balanced | random | fixed
  "games": [[...7 entries...], ...],    # instead of lineup/n_games: explicit tables, in power order
  "scoring": "dss",                     # default scoring for entries that don't set one
  "game": {...GameConfig fields...},
  "agent": {"output_limit": 8192, "memory_window": "auto"},
  "seed": 1,
  "max_usd": 5
}

Seat assignment:
- balanced: shuffle the lineup, then rotate it one seat per game for 7 games, then reshuffle.
  Every lineup slot plays every power once per block of 7 games, so power strength cancels
  out quickly; the analysis can still model score as model + power.
- random: an independent shuffle per game.
- fixed: the lineup in power order (Austria, England, France, Germany, Italy, Russia, Turkey).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields
from pathlib import Path

from .config import POWERS, GameConfig
from .scoring import CONDITIONS

ASSIGNMENTS = ("balanced", "random", "fixed")
DEFAULT_CONTEXT = {"anthropic": 200_000, "claude_cli": 200_000}
# Model-specific defaults that beat the provider default. Claude Haiku 5.5 costs 5x per token for
# prompts over 100K tokens, so its window stays under that (with margin for neutral-token drift).
MODEL_CONTEXT = {"claude-haiku-5-5": 95_000, "haiku": 95_000}


@dataclass
class Seat:
    model: str      # key into config["models"]
    scoring: str


@dataclass
class PlannedGame:
    index: int
    seed: int
    seats: dict[str, Seat]  # power -> seat

    @property
    def game_id(self) -> str:
        return f"game_{self.index:03d}"


def _entry(e, default_scoring: str, models: dict) -> Seat:
    seat = Seat(model=e, scoring=default_scoring) if isinstance(e, str) else Seat(e["model"], e.get("scoring", default_scoring))
    if seat.model not in models:
        raise ValueError(f"lineup names unknown model {seat.model!r}; defined: {sorted(models)}")
    if seat.scoring not in CONDITIONS:
        raise ValueError(f"unknown scoring {seat.scoring!r}; use one of {CONDITIONS}")
    return seat


def plan_games(cfg: dict) -> list[PlannedGame]:
    models = cfg["models"]
    default_scoring = cfg.get("scoring", "dss")
    base_seed = int(cfg.get("seed", 1))
    if "games" in cfg:
        tables = []
        for g in cfg["games"]:
            if isinstance(g, dict):
                g = [g[p] for p in POWERS]
            if len(g) != len(POWERS):
                raise ValueError(f"each game needs {len(POWERS)} seats, got {len(g)}")
            tables.append([_entry(e, default_scoring, models) for e in g])
    else:
        lineup = [_entry(e, default_scoring, models) for e in cfg["lineup"]]
        if len(lineup) != len(POWERS):
            raise ValueError(f"lineup needs {len(POWERS)} entries, got {len(lineup)}")
        n = int(cfg.get("n_games", 1))
        mode = cfg.get("assignment", "balanced")
        if mode not in ASSIGNMENTS:
            raise ValueError(f"assignment must be one of {ASSIGNMENTS}")
        rng = random.Random(f"assignment-{base_seed}")
        tables, order = [], list(range(len(POWERS)))
        for i in range(n):
            if mode == "fixed":
                perm = order
            elif mode == "random":
                perm = rng.sample(order, len(order))
            else:
                if i % len(POWERS) == 0:
                    block = rng.sample(order, len(order))
                k = i % len(POWERS)
                perm = block[-k:] + block[:-k] if k else block
            tables.append([lineup[j] for j in perm])
    return [PlannedGame(i, base_seed + i, dict(zip(POWERS, t))) for i, t in enumerate(tables)]


def game_config(cfg: dict, planned: PlannedGame) -> GameConfig:
    allowed = {f.name for f in fields(GameConfig)} - {"seed", "game_id"}
    extra = set(cfg.get("game", {})) - allowed
    if extra:
        raise ValueError(f"unknown game settings: {sorted(extra)}")
    return GameConfig(**cfg.get("game", {}), seed=planned.seed, game_id=f"{cfg['name']}/{planned.game_id}")


def memory_window(cfg: dict, planned: PlannedGame) -> int:
    """The same window for every seat: the smallest context of any model at this table."""
    setting = cfg.get("agent", {}).get("memory_window", "auto")
    if setting != "auto":
        return int(setting)
    windows = []
    for seat in planned.seats.values():
        spec = cfg["models"][seat.model]
        if spec.get("provider") == "human":
            continue  # people have no context window
        ctx = spec.get("context_window", MODEL_CONTEXT.get(spec.get("model"),
                                                           DEFAULT_CONTEXT.get(spec.get("provider", "anthropic"))))
        if ctx is None:
            raise ValueError(f"model {seat.model!r} needs a context_window for memory_window 'auto'")
        windows.append(int(ctx))
    return min(windows) if windows else DEFAULT_CONTEXT["anthropic"]


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    for key in ("name", "models"):
        if key not in cfg:
            raise ValueError(f"config needs {key!r}")
    if ("games" in cfg) == ("lineup" in cfg):
        raise ValueError("config needs exactly one of 'lineup' or 'games'")
    plan_games(cfg)  # validate
    return cfg
