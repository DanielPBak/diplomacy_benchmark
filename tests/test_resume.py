"""Checkpoints and resuming interrupted games."""
import json

import pytest

from dipbench.agents import LLMAgent, RandomAgent
from dipbench.config import POWERS, GameConfig
from dipbench.llm import CallLogger, LLMUnavailable
from dipbench.loop import GameRunner, run_game
from test_llm_agent import FakeProvider


class Crash(RuntimeError):
    pass


class CrashOnce(RandomAgent):
    """Raises the first time it is asked for orders in `crash_phase` (one crash per game)."""
    crashed = False

    def __init__(self, *a, crash_phase="F1902M", **kw):
        super().__init__(*a, **kw)
        self.crash_phase = crash_phase

    def orders(self, ctx):
        if ctx.phase == self.crash_phase and not CrashOnce.crashed and self.power == "ITALY":
            CrashOnce.crashed = True
            raise Crash("simulated failure")
        return super().orders(ctx)


def agents(cls=RandomAgent, **kw):
    return {p: cls(p, seed=5, draw_prob=0.05, **kw) for p in POWERS}


@pytest.mark.parametrize("press_mode", ["full", "public", "gunboat"])
def test_resumed_game_matches_uninterrupted_game(press_mode, tmp_path):
    cfg = GameConfig(press_mode=press_mode, max_year=1904, negotiation_rounds=2)
    straight = GameRunner(cfg, agents())
    expected = straight.run()

    CrashOnce.crashed = False
    out = tmp_path / "game"
    with pytest.raises(Crash):
        run_game(cfg, agents(CrashOnce), out_dir=out)
    checkpoint = json.loads((out / "checkpoint.json").read_text())
    assert checkpoint["phase"] == "F1902M"  # saved at the start of the failed phase

    resumed = run_game(cfg, agents(CrashOnce), out_dir=out, resume=True)
    assert resumed.resumes == 1
    assert not (out / "checkpoint.json").exists() and (out / "result.json").exists()
    assert set(json.loads((out / "agents.json").read_text())) == set(POWERS)
    for field in ("outcome", "winners", "final_centers", "final_phase", "year_centers", "n_messages",
                  "press_tokens_sent"):
        assert getattr(resumed, field) == getattr(expected, field), field
    saved_votes = [json.loads(l) for l in (out / "votes.jsonl").read_text().splitlines()]
    assert saved_votes == straight.vote_log
    saved_orders = [json.loads(l) for l in (out / "orders.jsonl").read_text().splitlines()]
    assert saved_orders == straight.order_log  # the aborted phase's partial orders are gone
    if press_mode != "gunboat":
        saved_press = [json.loads(l) for l in (out / "press.jsonl").read_text().splitlines()]
        assert saved_press == straight.press_log
        assert (out / "press_live.jsonl").exists()


def test_resume_restores_engine_rules(tmp_path):
    CrashOnce.crashed = False
    cfg = GameConfig(press_mode="public", max_year=1903)
    out = tmp_path / "g"
    with pytest.raises(Crash):
        run_game(cfg, agents(CrashOnce), out_dir=out)
    runner = GameRunner.resume(out / "checkpoint.json", agents())
    assert "PUBLIC_PRESS" in runner.game.rules and "NO_PRESS" not in runner.game.rules
    assert set(runner.game.rules) == set(GameRunner(cfg, agents()).game.rules)


def test_resume_rejects_different_models(tmp_path):
    CrashOnce.crashed = False
    out = tmp_path / "g"
    with pytest.raises(Crash):
        run_game(GameConfig(press_mode="gunboat", max_year=1903), agents(CrashOnce), out_dir=out)
    swapped = agents()
    swapped["FRANCE"] = RandomAgent("FRANCE", seed=5, identity="some-other-model")
    with pytest.raises(ValueError, match="agents differ"):
        GameRunner.resume(out / "checkpoint.json", swapped)


class FlakyProvider(FakeProvider):
    """Raises LLMUnavailable (e.g. credits exhausted) on the Nth call made through it."""
    def __init__(self, fail_at, **kw):
        super().__init__(**kw)
        self.fail_at, self.calls = fail_at, 0

    def complete(self, req):
        self.calls += 1
        if self.calls == self.fail_at:
            raise LLMUnavailable("BadRequestError: Your credit balance is too low")
        return super().complete(req)


def test_api_failure_stops_game_and_llm_game_resumes(tmp_path):
    out = tmp_path / "llm"
    cfg = GameConfig(press_mode="full", max_year=1902, negotiation_rounds=1)
    logger = CallLogger(path=out / "tokens.jsonl")
    first = {p: LLMAgent(p, "dss", FlakyProvider(fail_at=7 if p == "RUSSIA" else 10**9, logger=logger), cfg)
             for p in POWERS}
    with pytest.raises(LLMUnavailable):
        run_game(cfg, first, out_dir=out)
    state = json.loads((out / "checkpoint.json").read_text())
    assert state["agents"]["FRANCE"]["turns"] and state["agents"]["FRANCE"]["diary"]  # memory is checkpointed
    calls_before = len(logger.records)

    logger2 = CallLogger(path=out / "tokens.jsonl", resume=True)
    assert len(logger2.records) == calls_before  # earlier spending still counts
    second = {p: LLMAgent(p, "dss", FakeProvider(logger=logger2), cfg) for p in POWERS}
    result = run_game(cfg, second, out_dir=out, resume=True)
    assert result.outcome == "year_cap" and result.resumes == 1
    # The resumed agent continues its conversation instead of starting over.
    first_req = second["FRANCE"].provider.requests[0]
    assert len(first_req.history) == 2 * len(state["agents"]["FRANCE"]["turns"]) and "The game begins" not in first_req.prompt
    assert len(logger2.records) > calls_before


def test_fresh_logger_clears_old_token_log(tmp_path):
    path = tmp_path / "tokens.jsonl"
    path.write_text('{"x": 1}\n')
    CallLogger(path=path)
    assert not path.exists()
