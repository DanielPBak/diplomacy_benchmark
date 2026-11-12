"""Zero-cost smoke tests of the game loop with scripted random agents."""
import json

import pytest

from dipbench.agents import OutgoingMessage, RandomAgent
from dipbench.config import POWERS, GameConfig
from dipbench.loop import GameRunner, PressViolation, count_messages, run_game, validate_orders


def random_agents(seed, **kw):
    return {p: RandomAgent(p, seed=seed, **kw) for p in POWERS}


def run_until_retreat(make_config, make_agents, seeds=range(40)):
    """Random games rarely retreat early; try seeds until one has a retreat phase."""
    for seed in seeds:
        runner = GameRunner(make_config(), make_agents(seed))
        result = runner.run()
        if any(str(ph).endswith("R") for ph in runner.game.state_history.keys()):
            return runner, result
    raise AssertionError("no seed produced a retreat phase")


@pytest.mark.parametrize("press_mode", ["full", "public", "gunboat"])
def test_game_runs_to_year_cap(press_mode, tmp_path):
    cfg = GameConfig(press_mode=press_mode, max_year=1904, draw_votes=False, negotiation_rounds=2)
    result = run_game(cfg, random_agents(1), out_dir=tmp_path)
    assert result.outcome == "year_cap"
    assert result.final_year == 1905  # drawn at the start of S1905M
    assert sorted(result.winners) == sorted(result.survivors)
    assert sum(result.final_centers.values()) <= 34
    assert set(result.year_centers) == {1901, 1902, 1903, 1904}
    saved = json.loads((tmp_path / "game.json").read_text())
    assert saved["phases"] and ("NO_PRESS" in saved["rules"]) == (press_mode == "gunboat")
    assert (tmp_path / "press.jsonl").exists() == (press_mode != "gunboat")
    assert ("PUBLIC_PRESS" in saved["rules"]) == (press_mode == "public")


def test_gunboat_creates_no_messages_and_never_negotiates():
    class Spy(RandomAgent):
        def negotiate(self, ctx):
            raise AssertionError("negotiate called in gunboat")

        def orders(self, ctx):
            assert ctx.messages == [] and ctx.press_mode == "gunboat"
            return super().orders(ctx)

    cfg = GameConfig(press_mode="gunboat", max_year=1903)
    runner = GameRunner(cfg, {p: Spy(p, seed=2, draw_prob=0.0) for p in POWERS})
    result = runner.run()
    assert result.n_messages == 0
    assert count_messages(runner.game) == 0
    assert runner.press_log == []
    # Votes still happen in gunboat, checked in every phase of every season.
    seasons = {v["phase"][0] for v in runner.vote_log}
    assert {"S", "F", "W"} <= seasons
    assert all(isinstance(r["draw_vote"], bool) for r in runner.order_log)


def test_gunboat_send_is_blocked():
    runner = GameRunner(GameConfig(press_mode="gunboat"), random_agents(0))
    with pytest.raises(PressViolation):
        runner._send("FRANCE", "GLOBAL", "hi", 0)


def test_full_press_messages_are_private_and_in_every_phase():
    runner, _ = run_until_retreat(
        lambda: GameConfig(press_mode="full", max_year=1904, draw_votes=False, negotiation_rounds=3),
        random_agents)
    fanout = sum(1 if m["recipients"] == ["GLOBAL"] else len(m["recipients"]) for m in runner.press_log)
    assert runner.press_log and count_messages(runner.game) == fanout
    played = {str(p) for p in runner.game.state_history.keys()} - {runner.game.outcome[0]}
    press_phases = {m["phase"] for m in runner.press_log}
    assert press_phases >= played  # movement, retreat and Winter phases all have press
    assert {p[-1] for p in played} == {"M", "R", "A"}
    from dipbench.state import visible_messages
    for m in visible_messages(runner.press_log, "FRANCE"):
        assert m.recipients == ["GLOBAL"] or "FRANCE" in m.recipients or m.sender == "FRANCE"


@pytest.mark.parametrize("secret", [False, True])
def test_ballot_visibility(secret):
    seen = []

    class Watcher(RandomAgent):
        def orders(self, ctx):
            seen.append((ctx.power, ctx.my_draw_vote, ctx.draw_votes))
            sub = super().orders(ctx)
            sub.draw_vote = ctx.power == "FRANCE"
            return sub

    cfg = GameConfig(press_mode="gunboat", max_year=1901, secret_ballots=secret)
    run_game(cfg, {p: Watcher(p, seed=9) for p in POWERS})
    later = [s for s in seen if s[2] is not None or secret][7:]  # after the first phase's votes
    assert later
    for power, mine, votes in later:
        assert mine == (power == "FRANCE")
        if secret:
            assert votes is None
        else:
            assert votes == {p: p == "FRANCE" for p in POWERS}


def test_ballots_public_by_default():
    assert GameConfig().secret_ballots is False


def test_unanimous_vote_ends_game_immediately():
    runner = GameRunner(GameConfig(press_mode="gunboat", max_year=1910), random_agents(4, draw_prob=1.0))
    result = runner.run()
    assert result.outcome == "draw_vote"
    assert result.final_phase == "S1901M" and sorted(result.winners) == sorted(result.survivors)
    assert [str(p) for p in runner.game.state_history.keys()] == ["S1901M"]  # never adjudicated


def test_votes_count_from_configured_year():
    cfg = GameConfig(press_mode="gunboat", max_year=1910, draw_vote_from_year=1903)
    result = run_game(cfg, random_agents(4, draw_prob=1.0))
    assert result.outcome == "draw_vote" and result.final_phase == "S1903M"


@pytest.mark.parametrize("press_mode", ["full", "gunboat"])
def test_draw_can_pass_in_any_season(press_mode):
    class FallDrawer(RandomAgent):
        def orders(self, ctx):
            sub = super().orders(ctx)
            sub.draw_vote = ctx.phase.startswith("F")
            return sub

    result = run_game(GameConfig(press_mode=press_mode, max_year=1910),
                      {p: FallDrawer(p, seed=8) for p in POWERS})
    assert result.outcome == "draw_vote" and result.final_phase == "F1901M"


def test_one_no_vote_blocks_draw():
    agents = random_agents(5, draw_prob=1.0)
    agents["TURKEY"] = RandomAgent("TURKEY", seed=5, draw_prob=0.0)
    result = run_game(GameConfig(press_mode="gunboat", max_year=1904), agents)
    assert result.outcome == "year_cap"


def test_no_draw_votes_in_retreat_phases():
    class RetreatYes(RandomAgent):
        def orders(self, ctx):
            sub = super().orders(ctx)
            sub.draw_vote = ctx.phase_type == "R"  # yes only where votes don't count
            return sub

    runner, result = run_until_retreat(lambda: GameConfig(press_mode="gunboat", max_year=1904),
                                       lambda seed: {p: RetreatYes(p, seed=seed) for p in POWERS})
    assert result.outcome == "year_cap"
    assert any(r["phase"].endswith("R") for r in runner.order_log)
    assert {v["phase"][-1] for v in runner.vote_log} == {"M", "A"}


def test_every_survivor_votes_in_movement_and_winter_even_without_orders():
    runner = GameRunner(GameConfig(press_mode="gunboat", max_year=1904), random_agents(0))
    runner.run()
    for v in runner.vote_log:
        callers = {r["power"] for r in runner.order_log if r["phase"] == v["phase"]}
        assert callers == set(v["survivors"]) == set(v["ballots"])
    # Winter calls include powers with no builds or disbands (no legal orders at all).
    winter = [r for r in runner.order_log if r["phase"].endswith("A")]
    assert any(r["submitted"] == [] for r in winter)


def test_power_with_centers_but_no_units_can_vote_draw():
    agents = random_agents(0, draw_prob=1.0)
    runner = GameRunner(GameConfig(press_mode="gunboat", max_year=1910), agents)
    runner.game.set_units("FRANCE", [], reset=True)  # keeps its 3 centers, so it survives
    assert not runner.game.get_orderable_locations("FRANCE")
    result = runner.run()
    assert result.outcome == "draw_vote" and result.final_phase == "S1901M"
    assert "FRANCE" in result.winners and runner.vote_log[0]["ballots"]["FRANCE"] is True


def test_invalid_orders_retry_then_succeed():
    cfg = GameConfig(press_mode="gunboat", max_year=1901)
    runner = GameRunner(cfg, random_agents(6, invalid_prob=1.0))
    runner.run()
    first = [r for r in runner.order_log if r["attempt"] == 0]
    retries = [r for r in runner.order_log if r["attempt"] == 1]
    assert all(r["errors"] for r in first)
    assert all(any("draw_vote is required" in e for e in r["errors"])
               for r in first if not r["phase"].endswith("R"))
    assert retries and all(not r["errors"] for r in retries)


def test_validate_orders():
    legal = {"PAR": ["A PAR H", "A PAR - BUR"], "BRE": ["F BRE H"]}
    valid, errors = validate_orders(legal, "M", 2, ["a par - bur", "A PAR H", "A MUN H"])
    assert valid == ["A PAR - BUR"] and len(errors) == 2
    legal_a = {"PAR": ["A PAR B", "WAIVE"], "MAR": ["A MAR B", "WAIVE"]}
    valid, errors = validate_orders(legal_a, "A", 1, ["A PAR B", "A MAR B"])
    assert valid == ["A PAR B"] and len(errors) == 1


def test_long_random_games_terminate():
    for seed in range(3):
        cfg = GameConfig(press_mode="gunboat", max_year=1930, seed=seed)
        result = run_game(cfg, random_agents(seed))
        assert result.outcome in ("solo", "year_cap")


def test_seeded_games_reproducible_across_hash_seeds():
    import os
    import subprocess
    import sys
    code = ("from dipbench.agents import RandomAgent; from dipbench.config import POWERS, GameConfig;"
            "from dipbench.loop import run_game;"
            "r = run_game(GameConfig(press_mode='full', max_year=1903, negotiation_rounds=1),"
            " {p: RandomAgent(p, seed=11) for p in POWERS}); print(r.final_phase, r.final_centers)")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": "."}).stdout
            for h in ("1", "2")}
    assert len(outs) == 1


def test_eliminated_powers_do_not_negotiate_or_receive_press():
    called = []

    class Recorder(RandomAgent):
        def negotiate(self, ctx):
            called.append(self.power)
            others = [p for p in POWERS if p != self.power]
            return [OutgoingMessage(p, "hi") for p in others] + [OutgoingMessage(others, "group")]

    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=2)
    runner = GameRunner(cfg, {p: Recorder(p, seed=1) for p in POWERS})
    runner.game.set_units("ITALY", [], reset=True)
    runner.game.set_centers("ITALY", [], reset=True)
    assert runner.game.get_power("ITALY").is_eliminated()
    runner.run()
    assert called and "ITALY" not in called
    assert all(m["sender"] != "ITALY" and "ITALY" not in m["recipients"] for m in runner.press_log)
    # The group message still reaches the other five; the Italy part is logged as rejected.
    groups = [m for m in runner.press_log if m["text"] == "group"]
    assert groups and all(len(m["recipients"]) == 5 for m in groups)
    assert all(r["recipients"] == ["ITALY"] for r in runner.rejected_press)
    assert "ITALY" not in {r["power"] for r in runner.order_log}


# --- public press ---------------------------------------------------------------------

def test_public_press_delivers_only_broadcasts():
    class Mixed(RandomAgent):
        def negotiate(self, ctx):
            other = next(p for p in POWERS if p != self.power)
            return [OutgoingMessage(other, "psst"), OutgoingMessage("GLOBAL", "hello all")]

    cfg = GameConfig(press_mode="public", max_year=1901, draw_votes=False, negotiation_rounds=2)
    runner = GameRunner(cfg, {p: Mixed(p, seed=1) for p in POWERS})
    result = runner.run()
    assert runner.press_log and all(m["recipients"] == ["GLOBAL"] for m in runner.press_log)
    assert all(m["text"] == "hello all" for m in runner.press_log)
    assert result.n_rejected_messages == result.n_messages  # one private attempt per broadcast
    assert all("public press" in r["reason"] for r in runner.rejected_press)
    with pytest.raises(PressViolation):
        runner._send("FRANCE", "ENGLAND", "psst", 0)


# --- identity disclosure --------------------------------------------------------------

LABELS = {"AUSTRIA": "Claude Opus", "ENGLAND": "Claude Sonnet", "FRANCE": "Claude Haiku",
          "GERMANY": "GPT", "ITALY": "Gemini", "RUSSIA": "Llama", "TURKEY": "Qwen"}


def identity_run(mode, seed=0):
    seen = {}

    class Peek(RandomAgent):
        def orders(self, ctx):
            seen[ctx.power] = ctx.opponent_identities
            return super().orders(ctx)

    cfg = GameConfig(press_mode="gunboat", max_year=1901, identity_disclosure=mode, seed=seed)
    result = run_game(cfg, {p: Peek(p, seed=1, identity=LABELS[p]) for p in POWERS})
    return seen, result


def test_identities_hidden_by_default():
    assert GameConfig().identity_disclosure == "hidden"
    seen, result = identity_run("hidden")
    assert all(v is None for v in seen.values())
    assert result.identities_shown is None and result.identities_true == LABELS


def test_identities_revealed_exclude_self():
    seen, result = identity_run("revealed")
    for power, shown in seen.items():
        assert shown == {p: l for p, l in LABELS.items() if p != power}
    assert result.identities_shown == LABELS


def test_false_labels_are_a_wrong_shuffle_of_true_labels():
    seen, result = identity_run("false_labels", seed=3)
    shown = result.identities_shown
    assert sorted(shown.values()) == sorted(LABELS.values())
    assert all(shown[p] != LABELS[p] for p in POWERS)  # all distinct labels: full derangement
    for power, view in seen.items():
        assert view == {p: l for p, l in shown.items() if p != power}
    assert identity_run("false_labels", seed=3)[1].identities_shown == shown  # reproducible


def test_false_labels_with_majority_label_minimize_true_labels():
    from dipbench.loop import shown_identities
    import random
    true = {p: ("Claude" if i < 5 else f"other{i}") for i, p in enumerate(POWERS)}
    shown = shown_identities(true, "false_labels", random.Random(0))
    assert sum(shown[p] != true[p] for p in POWERS) == 4  # only 2 non-Claude labels to swap in


# --- multi-recipient messages and caps ------------------------------------------------

def test_group_message_visible_to_recipients_with_full_list():
    class Group(RandomAgent):
        def negotiate(self, ctx):
            if self.power == "FRANCE" and ctx.phase == "S1901M":
                return [OutgoingMessage(["ENGLAND", "GERMANY"], "western pact?")]
            return []

    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=1)
    runner = GameRunner(cfg, {p: Group(p, seed=1) for p in POWERS})
    runner.run()
    from dipbench.state import visible_messages
    for power in ("FRANCE", "ENGLAND", "GERMANY"):
        seen = [m for m in visible_messages(runner.press_log, power) if m.text == "western pact?"]
        assert len(seen) == 1 and seen[0].recipients == ["ENGLAND", "GERMANY"] and seen[0].round == 0
    assert not [m for m in visible_messages(runner.press_log, "ITALY") if m.text == "western pact?"]
    assert len(runner.press_log) == 1  # one message, not one per recipient


def test_no_message_count_or_length_cap_by_default():
    class Spammer(RandomAgent):
        def negotiate(self, ctx):
            return [OutgoingMessage("GLOBAL", "x" * 5000) for _ in range(20)] if self.power == "FRANCE" else []

    assert GameConfig().max_messages_per_round is None and GameConfig().max_message_chars is None
    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=1,
                     press_token_budget=None)
    runner = GameRunner(cfg, {p: Spammer(p, seed=1) for p in POWERS})
    runner.run()
    per_phase = {}
    for m in runner.press_log:
        per_phase[m["phase"]] = per_phase.get(m["phase"], 0) + 1
    assert set(per_phase.values()) == {20}
    assert all(len(m["text"]) == 5000 for m in runner.press_log) and runner.rejected_press == []


def test_caps_when_set_are_logged_not_silent():
    class Spammer(RandomAgent):
        def negotiate(self, ctx):
            return [OutgoingMessage("GLOBAL", "y" * 50) for _ in range(5)] if self.power == "FRANCE" else []

    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=1,
                     max_messages_per_round=3, max_message_chars=10, press_token_budget=None)
    runner = GameRunner(cfg, {p: Spammer(p, seed=1) for p in POWERS})
    result = runner.run()
    phases = {m["phase"] for m in runner.press_log}
    assert len(runner.press_log) == 3 * len(phases) and all(len(m["text"]) == 10 for m in runner.press_log)
    reasons = [r["reason"] for r in runner.rejected_press]
    assert reasons.count("over max_messages_per_round=3") == 2 * len(phases)
    assert reasons.count("truncated to max_message_chars=10") == 3 * len(phases)
    assert result.n_rejected_messages == len(runner.rejected_press)


# --- press token budget ---------------------------------------------------------------

def test_count_tokens_is_model_neutral():
    from dipbench.tokens import count_tokens
    assert count_tokens("") == 0 and count_tokens("abcd") == 1 and count_tokens("abcde") == 2


class Budgeted(RandomAgent):
    """FRANCE sends a 100-token message to a group every round and records its budget view."""
    views = []

    def negotiate(self, ctx):
        if self.power != "FRANCE":
            return []
        Budgeted.views.append((ctx.phase, ctx.negotiation_round, dict(ctx.press_budget)))
        return [OutgoingMessage(["ENGLAND", "GERMANY"], "z" * 400)]


def budget_run(**cfg_kw):
    Budgeted.views = []
    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=3, **cfg_kw)
    runner = GameRunner(cfg, {p: Budgeted(p, seed=1) for p in POWERS})
    return runner, runner.run()


def test_phase_budget_shared_across_rounds_and_shown_to_agent():
    runner, result = budget_run(press_token_budget=250, press_budget_scope="phase")
    s1901 = [v for v in Budgeted.views if v[0] == "S1901M"]
    assert [v[2]["remaining"] for v in s1901] == [250, 150, 50]  # group message costs 100 once
    assert s1901[0][2]["limit"] == 250 and s1901[0][2]["scope"] == "phase"
    assert "4 characters" in s1901[0][2]["unit"]
    sent = [m for m in runner.press_log if m["phase"] == "S1901M"]
    assert len(sent) == 2  # third round's message doesn't fit
    assert any("over press token budget" in r["reason"] for r in runner.rejected_press)
    assert result.press_tokens_sent["FRANCE"] == sum(m["tokens"] for m in runner.press_log)


def test_round_and_game_budget_scopes():
    runner, _ = budget_run(press_token_budget=100, press_budget_scope="round")
    assert all(v[2]["remaining"] == 100 for v in Budgeted.views)
    assert not runner.rejected_press
    runner, _ = budget_run(press_token_budget=250, press_budget_scope="game")
    assert len(runner.press_log) == 2  # whole game, not per phase


def test_default_budget_is_set_and_absent_in_gunboat():
    assert GameConfig().press_token_budget == 300 and GameConfig().press_budget_scope == "round"
    seen = []

    class Peek(RandomAgent):
        def orders(self, ctx):
            seen.append(ctx.press_budget)
            return super().orders(ctx)

    run_game(GameConfig(press_mode="gunboat", max_year=1901), {p: Peek(p, seed=1) for p in POWERS})
    assert seen and all(b is None for b in seen)



def test_group_messages_can_be_turned_off():
    class Group(RandomAgent):
        def negotiate(self, ctx):
            if self.power == "FRANCE":
                return [OutgoingMessage(["ENGLAND", "GERMANY"], "pact"), OutgoingMessage("ENGLAND", "dm"),
                        OutgoingMessage("GLOBAL", "hi all")]
            return []

    cfg = GameConfig(press_mode="full", max_year=1901, draw_votes=False, negotiation_rounds=1,
                     allow_group_messages=False)
    runner = GameRunner(cfg, {p: Group(p, seed=1) for p in POWERS})
    runner.run()
    assert {m["text"] for m in runner.press_log} == {"dm", "hi all"}
    assert all("group messages are off" in r["reason"] for r in runner.rejected_press)
