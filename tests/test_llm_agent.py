"""LLM agent, prompts and LLM layer, exercised with a scripted provider at zero cost."""
import json
import re

import pytest

from dipbench.agents import LLMAgent
from dipbench.agents import prompts
from dipbench.config import POWERS, GameConfig
from dipbench.llm import BudgetExceeded, CallLogger, LLMRequest, LLMResult, Provider
from dipbench.llm.base import CallRecord, parse_json
from dipbench.llm.pricing import call_cost
from dipbench.loop import GameRunner


class FakeProvider(Provider):
    """Plays the first legal order for every unit and broadcasts one short message per round."""
    name = "fake"

    def __init__(self, model="fake-model", logger=None, vote=False, garbage_first=False):
        super().__init__(model)
        self.logger = logger
        self.vote = vote
        self.garbage_first = garbage_first
        self.requests: list[LLMRequest] = []

    def complete(self, req):
        if self.logger:
            self.logger.check_budget()
        self.requests.append(req)
        record = CallRecord(req.game_id, req.power, req.phase, req.purpose, self.name, self.model,
                            input_tokens=len(req.prompt) // 4, output_tokens=50, cost_usd=0.001)
        if self.logger:
            self.logger.log(record)
        if self.garbage_first and "_parse" not in req.purpose:
            record.error = "response was not a JSON object"
            return LLMResult(None, "not json", record)
        if req.purpose.startswith("negotiate"):
            data = {"messages": [{"recipients": ["GLOBAL"], "text": f"{req.power} says hi in {req.purpose} {req.phase}"}]}
        else:
            header = "YOUR ORDERABLE UNITS AND LEGAL ORDERS THIS PHASE:\n"
            # On a corrective retry the legal list is in an earlier turn, as for a real model.
            source = next(t for t in [req.prompt] + [m["content"] for m in reversed(req.history)] if header in t)
            block = source.split(header, 1)[1].split("\n\n", 1)[0]
            orders = [line.split(": ", 1)[1].split(" | ")[0] for line in block.splitlines() if ": " in line]
            data = {"orders": orders, "draw_vote": self.vote, "diary": f"diary {req.power} {req.phase} " + "x" * 200}
        return LLMResult(data, json.dumps(data), record)


def llm_agents(cfg, logger, **kw):
    return {p: LLMAgent(p, "dss", FakeProvider(logger=logger, **kw), cfg, identity=f"model-{p}") for p in POWERS}


# --- system prompts -------------------------------------------------------------------

def test_system_prompt_states_scoring_rule_with_worked_example():
    cfg = GameConfig()
    for scoring, needle in (("dss", "1/3 = 0.333"), ("sos", "256/436 = 0.587"), ("linear", "10/34 = 0.294")):
        text = prompts.system_prompt("FRANCE", scoring, cfg, 8192, None)
        assert needle in text and "You are playing FRANCE" in text


def test_system_prompt_per_press_mode():
    gunboat = prompts.system_prompt("FRANCE", "dss", GameConfig(press_mode="gunboat"), 8192, None)
    assert "no power can send or receive messages" in gunboat and "PRESS BUDGET" not in gunboat
    public = prompts.system_prompt("FRANCE", "dss", GameConfig(press_mode="public"), 8192, None)
    assert "only allowed recipient is GLOBAL" in public and "300 tokens" in public
    full = prompts.system_prompt("FRANCE", "dss", GameConfig(press_mode="full"), 8192, None)
    assert "group of powers" in full and "per round" in full


def test_system_prompt_votes_limits_and_identities():
    cfg = GameConfig(secret_ballots=True, max_year=1912)
    text = prompts.system_prompt("FRANCE", "sos", cfg, 4096, {"ENGLAND": "Model X"})
    assert "Ballots are secret" in text and "4096 output tokens" in text
    assert "- ENGLAND: Model X" in text and "after Winter 1912" in text
    hidden = prompts.system_prompt("FRANCE", "sos", GameConfig(), 4096, None)
    assert "not told which AI model" in hidden and "Ballots are public" in hidden


# --- full games through the loop ------------------------------------------------------

@pytest.mark.parametrize("press_mode", ["full", "public", "gunboat"])
def test_llm_agents_play_a_game(press_mode):
    logger = CallLogger()
    cfg = GameConfig(press_mode=press_mode, max_year=1902, negotiation_rounds=2, identity_disclosure="revealed")
    runner = GameRunner(cfg, llm_agents(cfg, logger))
    result = runner.run()
    assert result.outcome == "year_cap"
    purposes = {r.purpose.split("_")[0] for r in logger.records}
    if press_mode == "gunboat":
        assert purposes == {"orders"} and result.n_messages == 0
    else:
        assert purposes == {"negotiate", "orders"} and result.n_messages > 0
    # The system prompt is identical for every call by a power (cacheable).
    reqs = runner.agents["FRANCE"].provider.requests
    assert len({r.system for r in reqs}) == 1
    assert "- ENGLAND: model-ENGLAND" in reqs[0].system and "model-FRANCE" not in reqs[0].system
    # Diary entries are recorded once per phase submitted.
    assert [p for p, _ in runner.agents["FRANCE"].diary][:2] == ["S1901M", "F1901M"]


def test_prompt_shows_press_budget_and_ballots():
    logger = CallLogger()
    cfg = GameConfig(press_mode="full", max_year=1901, negotiation_rounds=1)
    runner = GameRunner(cfg, llm_agents(cfg, logger))
    runner.run()
    reqs = runner.agents["ITALY"].provider.requests
    neg = next(r for r in reqs if r.purpose.startswith("negotiate"))
    assert "300 of 300 press tokens left this round" in neg.prompt
    later_orders = [r for r in reqs if r.purpose == "orders" and r.phase == "F1901M"]
    fall_first = next(r for r in reqs if r.phase == "F1901M")
    assert later_orders and "Most recent draw votes:" in fall_first.prompt  # shown when they changed
    # Earlier press is still in the conversation, with its round number.
    assert any("[S1901M round 1]" in m["content"] for m in later_orders[0].history if m["role"] == "user")


def test_unanimous_llm_draw_vote_ends_game():
    logger = CallLogger()
    cfg = GameConfig(press_mode="gunboat", max_year=1910)
    result = GameRunner(cfg, llm_agents(cfg, logger, vote=True)).run()
    assert result.outcome == "draw_vote" and result.final_phase == "S1901M"


def test_unparseable_response_is_retried_and_logged():
    logger = CallLogger()
    cfg = GameConfig(press_mode="gunboat", max_year=1901)
    runner = GameRunner(cfg, llm_agents(cfg, logger, garbage_first=True))
    runner.run()
    purposes = [r.purpose for r in logger.records if r.power == "FRANCE" and r.phase == "S1901M"]
    assert purposes == ["orders", "orders_parse1"]


def test_budget_cap_stops_the_game():
    logger = CallLogger(max_usd=0.01)
    cfg = GameConfig(press_mode="gunboat", max_year=1905)
    with pytest.raises(BudgetExceeded):
        GameRunner(cfg, llm_agents(cfg, logger)).run()
    assert logger.total_usd < 0.01 + 7 * 0.001  # at most one in-flight call per power past the cap


# --- LLM layer ------------------------------------------------------------------------

def test_call_cost():
    # Opus 5.5: $4 in / $20 out / $0.20 cache read per Mtok; cache write 1.25x input.
    assert call_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert call_cost("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)
    assert call_cost("claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.2)
    assert call_cost("claude-opus-5-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(5.0)
    # Haiku 5.5 long-context tier above 100K prompt tokens.
    assert call_cost("claude-haiku-5-5", 50_000, 1000) == pytest.approx((50_000 * 0.10 + 1000 * 0.50) / 1e6)
    assert call_cost("claude-haiku-5-5", 150_000, 1000) == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1e6)
    assert call_cost("some-local-model", 10**6, 10**6) == 0.0


def test_anthropic_request_shape():
    from dipbench.llm.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider("claude-haiku-5-5", client=object())
    kw = provider.build_kwargs(LLMRequest(system="S", prompt="P", schema=prompts.ORDERS_SCHEMA,
                                          max_output_tokens=8192, effort="low"))
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": prompts.ORDERS_SCHEMA}, "effort": "low"}
    assert kw["max_tokens"] == 8192 and kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["messages"] == [{"role": "user", "content": "P"}]


def test_parse_json_tolerates_fences():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json") is None and parse_json("[1, 2]") is None
    qwen = '<think>\nMaybe {"orders": ["A PAR H"]}? No, better to move.\n</think>\n{"orders": ["A PAR - BUR"]}'
    assert parse_json(qwen) == {"orders": ["A PAR - BUR"]}


def test_logger_summary_and_file(tmp_path):
    logger = CallLogger(path=tmp_path / "tokens.jsonl")
    logger.log(CallRecord("g", "FRANCE", "S1901M", "negotiate_r0", "fake", "m", input_tokens=10,
                          output_tokens=5, cost_usd=0.5))
    logger.log(CallRecord("g", "FRANCE", "S1901M", "orders", "fake", "m", input_tokens=20, output_tokens=5,
                          cost_usd=0.25))
    s = logger.summary()
    assert s["calls"] == 2 and s["cost_usd"] == 0.75
    assert s["by_purpose"]["negotiate"]["calls"] == 1 and s["by_power"]["FRANCE"]["input_tokens"] == 30
    rows = [json.loads(l) for l in (tmp_path / "tokens.jsonl").read_text().splitlines()]
    assert [r["purpose"] for r in rows] == ["negotiate_r0", "orders"]
    assert set(rows[0]) >= {"model", "phase", "power", "purpose", "input_tokens", "output_tokens",
                            "thinking_tokens", "latency_s", "cost_usd"}


def test_openai_compat_request_shape():
    from dipbench.llm.openai_compat import OpenAICompatProvider

    class Capture:
        def __init__(self):
            self.kwargs = None
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.kwargs = kw
            raise RuntimeError("stop")

    p = OpenAICompatProvider("qwen", "http://127.0.0.1:1/v1", json_mode="none", enable_thinking=False)
    p.client = Capture()
    with pytest.raises(RuntimeError):
        p.complete(LLMRequest(system="S", prompt="P", schema={"type": "object"}, max_output_tokens=100))
    kw = p.client.kwargs
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "response_format" not in kw
    assert kw["messages"][1]["content"].startswith("P") and '{"type": "object"}' in kw["messages"][1]["content"]



# --- persistent memory ----------------------------------------------------------------

def test_system_prompt_explains_memory():
    text = prompts.system_prompt("FRANCE", "dss", GameConfig(), 8192, None, memory_window=36000, diary_max_chars=500)
    assert "one continuous conversation" in text and "CANNOT see your own earlier private reasoning" in text
    assert "36,000 tokens" in text and "at most 500 characters" in text
    assert "full\nrecord of every phase's orders and results is always kept" in text


def memory_game(window=200_000, press_mode="full", max_year=1902):
    logger = CallLogger()
    cfg = GameConfig(press_mode=press_mode, max_year=max_year, negotiation_rounds=2)
    agents = {p: LLMAgent(p, "dss", FakeProvider(logger=logger), cfg, output_limit=1000, memory_window=window)
              for p in POWERS}
    runner = GameRunner(cfg, agents)
    runner.run()
    return runner, agents


def test_conversation_is_append_only_and_complete():
    runner, agents = memory_game()
    reqs = agents["FRANCE"].provider.requests
    for a, b in zip(reqs, reqs[1:]):
        sent_a = a.history + [{"role": "user", "content": a.prompt}]
        assert b.history[:len(sent_a)] == sent_a  # each call extends the previous one
        assert [m["role"] for m in b.history] == ["user", "assistant"] * (len(b.history) // 2)
    # Every message France received is delivered in exactly one of its user turns; its own
    # messages aren't echoed back (they're already in its replies).
    from dipbench.state import visible_messages
    user_turns = [r.prompt for r in reqs]
    for m in visible_messages(runner.press_log, "FRANCE"):
        tag = f"[{m.phase} round {m.round + 1}] {m.sender} -> "
        hits = [t for t in user_turns if tag in t and m.text in t]
        assert len(hits) == (0 if m.sender == "FRANCE" else 1), (tag, m.text)


def test_memory_window_trims_oldest_turns_and_keeps_diary():
    runner, agents = memory_game(window=8000, max_year=1903)
    fr = agents["FRANCE"]
    assert fr.trims and fr.trimmed_through > 0
    last = fr.provider.requests[-1]
    from dipbench.tokens import count_tokens
    size = (count_tokens(last.system) + sum(count_tokens(m["content"]) for m in last.history)
            + count_tokens(last.prompt) + fr.output_limit)
    assert size <= 8000
    digest = last.history[0]["content"]
    assert digest.startswith("[MEMORY] The conversation in your turns from S1901M")
    assert "diary FRANCE S1901M" in digest  # the trimmed period's negotiations survive through the diary
    # The board record is never trimmed: every completed phase France was shown is in the
    # digest or in a kept turn, with every power's orders.
    shown = "\n".join([digest] + [m["content"] for m in last.history[1:]] + [last.prompt])
    for entry in fr.board_log:
        assert f"{entry['phase']}:\n" in shown, entry["phase"]
        for power, orders in entry["orders"].items():
            for o in orders:
                assert o in shown, (entry["phase"], o)
    assert "S1901M:\n" in digest and "ENGLAND:" in digest


def test_memory_survives_checkpoint_round_trip():
    runner, agents = memory_game(max_year=1901)
    state = agents["ITALY"].get_state()
    fresh = LLMAgent("ITALY", "dss", FakeProvider(), runner.config)
    fresh.set_state(json.loads(json.dumps(state)))
    assert fresh._messages() == agents["ITALY"]._messages() and fresh.diary == agents["ITALY"].diary



def test_board_legal_orders_and_votes_sent_once_per_phase():
    runner, agents = memory_game(max_year=1902)
    for power, agent in agents.items():
        reqs = agent.provider.requests
        by_phase = {}
        for r in reqs:
            by_phase.setdefault(r.phase, []).append(r.prompt)
        for phase, turns in by_phase.items():
            assert sum("BOARD NOW:" in t for t in turns) == 1, (power, phase)
            assert sum("LEGAL ORDERS THIS PHASE:" in t for t in turns) == 1, (power, phase)
            assert turns[0].count("BOARD NOW:") == 1  # on the power's first turn of the phase
        votes_turns = [t for t in (r.prompt for r in reqs) if "draw votes" in t.lower()]
        assert len(votes_turns) <= len(by_phase)  # never more than once per phase


def test_conversation_growth_per_turn_is_small():
    from dipbench.tokens import count_tokens
    runner, agents = memory_game(max_year=1902)
    reqs = agents["FRANCE"].provider.requests
    later_rounds = [r for r in reqs if r.purpose == "negotiate_r1"]
    # A second negotiation round carries only the new messages and the task.
    assert later_rounds and all("BOARD NOW:" not in r.prompt and "LEGAL ORDERS" not in r.prompt for r in later_rounds)
    assert max(count_tokens(r.prompt) for r in later_rounds) < 400


def test_board_and_legal_orders_reshown_if_trimmed_away():
    runner, agents = memory_game(window=4500, max_year=1901)
    fr = agents["FRANCE"]
    assert fr.trims
    for r in fr.provider.requests:
        visible = "\n".join(m["content"] for m in r.history) + r.prompt
        assert f"Current phase: " in r.prompt
        if r.purpose.startswith("orders"):
            assert "LEGAL ORDERS THIS PHASE:" in visible, r.phase



def test_reasoning_is_logged_but_never_sent_back():
    class Thinker(FakeProvider):
        def complete(self, req):
            r = super().complete(req)
            r.reasoning = f"SECRET-REASONING {req.power} {req.purpose} {req.phase}"
            return r

    cfg = GameConfig(press_mode="full", max_year=1901, negotiation_rounds=2)
    agents = {p: LLMAgent(p, "dss", Thinker(), cfg) for p in POWERS}
    GameRunner(cfg, agents).run()
    fr = agents["FRANCE"]
    assert all(t["reasoning"].startswith("SECRET-REASONING") for t in fr.turns)
    for r in fr.provider.requests:
        assert "SECRET-REASONING" not in r.prompt + "".join(m["content"] for m in r.history)



def test_board_shows_current_draw_shares_under_own_scoring():
    logger = CallLogger()
    cfg = GameConfig(press_mode="gunboat", max_year=1901)
    agents = {p: LLMAgent(p, "sos" if p == "RUSSIA" else "dss", FakeProvider(logger=logger), cfg) for p in POWERS}
    GameRunner(cfg, agents).run()
    ru = agents["RUSSIA"].provider.requests[0].prompt
    # Start: Russia 4 centers, others 3 -> SoS 16/70 and 9/70.
    assert "IF A DRAW PASSED NOW (Sum-of-Squares" in ru and "You (RUSSIA) would score 0.229" in ru
    assert "FRANCE 0.129" in ru
    fr = agents["FRANCE"].provider.requests[0].prompt
    assert "(Draw-Size Scoring" in fr and "You (FRANCE) would score 0.143" in fr


def test_system_prompt_states_timing_rules():
    text = prompts.system_prompt("FRANCE", "sos", GameConfig(), 8192, None)
    assert "change owner only at the end of each Fall turn" in text
    assert "before that\nphase's orders are carried out" in text



def test_draw_shares_shown_only_when_ownership_changes():
    runner, agents = memory_game(max_year=1903)
    for power, agent in agents.items():
        reqs = agent.provider.requests
        shown = [r.phase for r in reqs if "IF A DRAW PASSED NOW" in r.prompt]
        assert shown[0] == "S1901M"  # first turn of the game
        # At most one showing per change of center ownership, which happens after Fall at most.
        changes = 1 + sum(1 for y in (1901, 1902) if runner.year_centers.get(y) != runner.year_centers.get(y - 1))
        assert len(shown) <= changes + 1, (power, shown)
        assert len(shown) < len({r.phase for r in reqs})  # not every phase



def test_trimming_is_rare_so_caches_survive():
    runner, agents = memory_game(window=8000, max_year=1904)
    fr = agents["FRANCE"]
    reqs = fr.provider.requests
    assert fr.trims
    # Most calls extend the previous call exactly (cacheable); only trim events break that.
    extends = sum(1 for a, b in zip(reqs, reqs[1:])
                  if b.history[:len(a.history) + 1] == a.history + [{"role": "user", "content": a.prompt}])
    assert len(fr.trims) <= 0.25 * len(reqs) and extends >= len(reqs) - 1 - len(fr.trims)



def test_map_reference_matches_engine():
    from diplomacy import Game
    from dipbench.agents.map_reference import map_reference
    text = map_reference()
    m = Game().map
    lines = {l.split(" ", 1)[0]: l for l in text.splitlines()[3:]}
    assert len(lines) == 76 and all(c in text.splitlines()[1] for c in m.scs)  # 75 provinces + impassable SWI
    assert lines["PAR"].endswith("army: BRE, BUR, GAS, PIC") and "FRANCE home SC" in lines["PAR"]
    assert "fleet from BUL/EC: BLA, CON, RUM" in lines["BUL"] and "(impassable)" in lines["SWI"]
    # Army moves are symmetric and agree with the adjudicator.
    for prov, line in lines.items():
        if "army: " in line:
            for nb in line.split("army: ")[1].split(";")[0].split(", "):
                assert m.abuts("A", prov, "-", nb) and m.abuts("A", nb, "-", prov), (prov, nb)
    assert "MAP (standard board" in prompts.system_prompt("FRANCE", "dss", GameConfig(), 8192, None)



def test_cache_ttl_option_and_pricing():
    from dipbench.llm.anthropic_provider import AnthropicProvider
    kw = AnthropicProvider("claude-haiku-5-5", client=object(), cache_ttl="1h").build_kwargs(
        LLMRequest(system="S", prompt="P", schema=prompts.TURN_SCHEMA, max_output_tokens=10))
    assert kw["cache_control"] == {"type": "ephemeral", "ttl": "1h"} == kw["system"][0]["cache_control"]
    # 1-hour writes cost 2x input, 5-minute writes 1.25x (Sonnet 5.5: $2/Mtok input).
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=10**6, cache_write_1h_tokens=10**6) == pytest.approx(4.0)
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=10**6) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        AnthropicProvider("claude-haiku-5-5", client=object(), cache_ttl="2h")



def test_announced_end_year_differs_from_real_cap():
    cfg = GameConfig(press_mode="gunboat", max_year=1901, announced_max_year=1999, draw_votes=False)
    logger = CallLogger()
    agents = {p: LLMAgent(p, "dss", FakeProvider(logger=logger), cfg) for p in POWERS}
    result = GameRunner(cfg, agents).run()
    system = agents["FRANCE"].provider.requests[0].system
    assert "after Winter 1999" in system and "1901" not in system.split("GAME END")[1].split("TIMING")[0]
    assert result.outcome == "year_cap" and result.final_phase == "S1902M"
    assert result.max_year == 1901 and result.announced_max_year == 1999
    with pytest.raises(ValueError):
        GameConfig(max_year=1910, announced_max_year=1905)



def test_system_prompt_explains_how_moves_resolve():
    text = prompts.system_prompt("FRANCE", "sos", GameConfig(), 4096, None)
    assert "HOW MOVES RESOLVE" in text and "Equal strengths bounce" in text and "dislodged" in text
    assert "support is cut" in text.lower()



def test_system_prompt_says_zero_sum_and_it_is():
    from dipbench.scoring import dss, linear, sos
    for rule in ("dss", "sos", "linear"):
        text = prompts.system_prompt("FRANCE", rule, GameConfig(), 4096, None)
        assert "ZERO-SUM: the game and its scoring are zero-sum" in text and "any score another power gains" not in text
    # The claim is true for the draw-based rules: scores always sum to 1.
    from test_scoring import make_result
    for r in (make_result("draw_vote", {"FRANCE": 16, "ENGLAND": 12, "GERMANY": 6}, ["ENGLAND", "FRANCE", "GERMANY"]),
              make_result("solo", {"FRANCE": 18, "ENGLAND": 16}, ["FRANCE"], final_year=1905)):
        assert sum(dss(r).values()) == pytest.approx(1) and sum(sos(r).values()) == pytest.approx(1)
        assert sum(linear(r).values()) <= 1 + 1e-9


# --- claude -p provider ---------------------------------------------------------------

def test_claude_cli_provider_flags_transcript_and_billing(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from dipbench.llm.claude_cli import ClaudeCLIProvider
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    fake = [sys.executable, str(Path(__file__).with_name("fake_claude.py"))]
    logger = CallLogger()
    p = ClaudeCLIProvider("sonnet", logger=logger, cli=fake)
    req = LLMRequest(system="SYS", prompt="NEW", schema=prompts.TURN_SCHEMA, max_output_tokens=2048, effort="high",
                     history=[{"role": "user", "content": "OLD1"}, {"role": "assistant", "content": "R1"}])
    result = p.complete(req)
    call = json.loads(log.read_text().splitlines()[0])
    a = call["args"]
    assert a[a.index("--model") + 1] == "sonnet" and a[a.index("--effort") + 1] == "high"
    assert a[a.index("--system-prompt") + 1] == "SYS" and a[a.index("--tools") + 1] == ""
    assert "--safe-mode" in a and "--no-session-persistence" in a and call["max_out"] == "2048"
    assert "diplomacy" not in call["cwd"].lower() and Path(call["cwd"]).name.startswith("game_")
    assert call["prompt"].index("OLD1") < call["prompt"].index("R1") < call["prompt"].index("NEW")
    assert call["blocks"] == 3  # one block per earlier turn + the new turn, so the cache can reuse prefixes
    assert a[a.index("--input-format") + 1] == "stream-json"
    assert result.data == {"messages": [], "orders": [], "draw_vote": False, "diary": ""}
    r = logger.records[0]
    assert r.cost_usd == 0.0 and r.reported_cost_usd == 0.0123  # subscription: counts as $0
    assert (r.cache_read_tokens, r.cache_write_tokens, r.thinking_tokens) == (100, 50, 7)
    api = ClaudeCLIProvider("sonnet", logger=CallLogger(), cli=fake, billing="api")
    api.complete(req)
    assert api.logger.records[0].cost_usd == 0.0123


def test_claude_cli_failure_stops_game(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from dipbench.llm import LLMUnavailable
    from dipbench.llm.claude_cli import ClaudeCLIProvider
    monkeypatch.setenv("FAKE_CLAUDE_FAIL", "1")
    p = ClaudeCLIProvider("sonnet", cli=[sys.executable, str(Path(__file__).with_name("fake_claude.py"))])
    with pytest.raises(LLMUnavailable):
        p.complete(LLMRequest(system="S", prompt="P", schema=prompts.TURN_SCHEMA, max_output_tokens=10))


def test_claude_cli_agents_play_a_game(tmp_path):
    import sys
    from pathlib import Path
    from dipbench.llm import make_provider
    fake = [sys.executable, str(Path(__file__).with_name("fake_claude.py"))]
    cfg = GameConfig(press_mode="full", max_year=1901, negotiation_rounds=1)
    logger = CallLogger()
    agents = {p: LLMAgent(p, "sos", make_provider({"provider": "claude_cli", "model": "haiku", "cli": fake}, logger), cfg)
              for p in POWERS}
    result = GameRunner(cfg, agents).run()
    assert result.outcome == "year_cap" and not [r for r in logger.records if r.error]



def test_claude_cli_blocks_never_change_once_sent():
    from dipbench.llm.claude_cli import transcript_blocks
    h = []
    previous = transcript_blocks(h, "T1")
    for i, reply in enumerate(["R1", "R2", "R3"]):
        h = h + [{"role": "user", "content": f"T{i + 1}"}, {"role": "assistant", "content": reply}]
        current = transcript_blocks(h, f"T{i + 2}")
        assert current[:len(previous)] == previous  # every earlier block identical: cacheable prefix
        previous = current



def test_thinking_budget_forces_an_answer():
    from types import SimpleNamespace as NS
    from dipbench.llm.openai_compat import BUDGET_CUTOFF, OpenAICompatProvider, render_chatml

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kw):
            self.calls.append(kw)
            usage = NS(prompt_tokens=100, completion_tokens=kw["max_tokens"] if len(self.calls) == 1 else 20,
                       prompt_tokens_details=NS(cached_tokens=60))
            if len(self.calls) == 1:  # thinking: runs out of budget
                return NS(choices=[NS(text="Let me think about Belgium and", finish_reason="length")], usage=usage)
            return NS(choices=[NS(text='{"messages": [], "orders": ["A PAR H"], "draw_vote": false, "diary": "d"}',
                                  finish_reason="stop")], usage=usage)

    p = OpenAICompatProvider("qwen", "http://127.0.0.1:1/v1", json_mode="none", thinking_budget=300)
    fake = FakeCompletions()
    p.client = NS(completions=fake)
    r = p.complete(LLMRequest(system="SYS", prompt="P", schema=prompts.TURN_SCHEMA, max_output_tokens=4096,
                              history=[{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]))
    first, second = fake.calls
    assert first["max_tokens"] == 300 and first["stop"] == ["</think>"]
    assert first["prompt"].startswith("<|im_start|>system\nSYS<|im_end|>") and first["prompt"].endswith("<think>\n")
    assert "<|im_start|>assistant\nA1<|im_end|>" in first["prompt"]
    assert second["prompt"].endswith("Belgium and" + BUDGET_CUTOFF + "\n</think>\n\n")
    assert r.data["orders"] == ["A PAR H"] and r.record.thinking_tokens == 300 and r.reasoning.endswith(BUDGET_CUTOFF)
    assert r.record.output_tokens == 320 and r.record.cache_read_tokens == 120
