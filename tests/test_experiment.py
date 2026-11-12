"""Experiment configs: seat planning, memory windows, and the runner end to end (fake models)."""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from dipbench.config import POWERS
from dipbench.experiment import load_config, memory_window, plan_games
from test_llm_agent import FakeProvider

MODELS = {"a": {"provider": "anthropic", "model": "claude-haiku-5-5"},
          "q": {"provider": "openai_compat", "model": "qwen", "base_url": "http://x", "context_window": 34000}}


def cfg(**kw):
    return {"name": "t", "models": MODELS, "lineup": ["a"] + ["q"] * 6, "n_games": 7, **kw}


def test_balanced_puts_each_slot_in_each_power_once_per_block():
    plan = plan_games(cfg(n_games=14, assignment="balanced"))
    for block in (plan[:7], plan[7:]):
        a_seats = [next(p for p, s in g.seats.items() if s.model == "a") for g in block]
        assert sorted(a_seats) == sorted(POWERS)
    assert [g.seed for g in plan] == list(range(1, 15))


def test_random_and_fixed_assignment():
    plan = plan_games(cfg(n_games=200, assignment="random", seed=3))
    counts = Counter(next(p for p, s in g.seats.items() if s.model == "a") for g in plan)
    assert set(counts) == set(POWERS)
    fixed = plan_games(cfg(n_games=3, assignment="fixed"))
    assert all(g.seats["AUSTRIA"].model == "a" for g in fixed)
    assert plan_games(cfg(assignment="random", seed=3))[0].seats == plan_games(cfg(assignment="random", seed=3))[0].seats


def test_scoring_follows_the_model_and_explicit_games():
    c = cfg(lineup=[{"model": "a", "scoring": "sos"}] + ["q"] * 6)
    for g in plan_games(c):
        for s in g.seats.values():
            assert s.scoring == ("sos" if s.model == "a" else "dss")
    explicit = {"name": "t", "models": MODELS, "games": [["q"] * 7, {p: "a" for p in POWERS}]}
    plan = plan_games(explicit)
    assert [set(s.model for s in g.seats.values()) for g in plan] == [{"q"}, {"a"}]


def test_memory_window_is_smallest_at_table():
    plan = plan_games(cfg(assignment="fixed"))
    assert memory_window(cfg(), plan[0]) == 34000
    all_a = {"name": "t", "models": MODELS, "lineup": ["a"] * 7}
    assert memory_window(all_a, plan_games(all_a)[0]) == 95_000  # Haiku stays under its 100K price tier
    sonnet = {"name": "t", "models": {"s": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}, "lineup": ["s"] * 7}
    assert memory_window(sonnet, plan_games(sonnet)[0]) == 200_000


def test_bad_configs_rejected(tmp_path):
    for bad in (cfg(lineup=["a"] * 6), cfg(lineup=["zzz"] + ["q"] * 6), cfg(assignment="nope"),
                {"name": "t", "models": MODELS}):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, KeyError)):
            load_config(p)


def test_runner_runs_skips_and_resumes(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run
    monkeypatch.setattr(run, "make_provider", lambda spec, logger=None: FakeProvider(model=spec["model"], logger=logger))
    c = cfg(n_games=2, game={"press_mode": "full", "max_year": 1901, "negotiation_rounds": 1}, max_usd=1.0)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run.py", str(path), "--out", str(out)])
    run.main()
    for i in (0, 1):
        g = out / f"game_{i:03d}"
        assert (g / "result.json").exists() and (g / "agents.json").exists() and not (g / "checkpoint.json").exists()
        seats = json.loads((g / "seats.json").read_text())
        assert seats["memory_window"] == 34000 and len(seats["seats"]) == 7
    spent = run.spent_so_far(out)
    run.main()  # second run skips both games and spends nothing
    assert run.spent_so_far(out) == spent
