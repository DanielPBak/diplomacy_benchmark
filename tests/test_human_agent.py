"""The human seat, driven over HTTP by a scripted 'person'."""
import json
import threading
import time
import urllib.request

from dipbench.agents import HumanAgent, RandomAgent
from dipbench.agents.human_agent import HumanQuit, HumanSession
from dipbench.config import POWERS, GameConfig
from dipbench.loop import GameRunner


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def play_as_person(url, stop, seen):
    """Answer every turn: one message to England per round, first legal order per unit, vote no."""
    while not stop.is_set():
        try:
            state = json.loads(get(url + "state"))
        except OSError:
            return  # the game ended and closed the page server
        view = state["view"]
        if state["waiting"] and view["seq"] not in seen:
            seen.append(view["seq"])
            if view["kind"] == "negotiate":
                reply = {"messages": [{"recipients": ["ENGLAND"], "text": f"hello from round {view['seq']}"}]}
            else:
                reply = {"orders": [v[0] for v in view["legal_orders"].values()], "draw_vote": False}
            assert post(url + "submit", {"seq": view["seq"], "reply": reply})["error"] is None
        time.sleep(0.02)


def test_human_seat_plays_a_year_through_the_web_page():
    cfg = GameConfig(press_mode="full", max_year=1901, negotiation_rounds=1, press_token_budget=300)
    human = HumanAgent("FRANCE", "dss", cfg, port=0)
    agents = {p: RandomAgent(p, seed=3) for p in POWERS if p != "FRANCE"} | {"FRANCE": human}
    url = human.session.url
    assert b"Diplomacy Seat" in get(url)
    stop, seen = threading.Event(), []
    person = threading.Thread(target=play_as_person, args=(url, stop, seen), daemon=True)
    person.start()
    runner = GameRunner(cfg, agents)
    result = runner.run()
    stop.set()
    assert result.outcome == "year_cap"
    # 3 phases (S, F, W), each with 1 negotiation round and an order submission.
    assert len(seen) == 6
    sent = [m for m in runner.press_log if m["sender"] == "FRANCE"]
    assert len(sent) == 3 and all(m["recipients"] == ["ENGLAND"] for m in sent)
    france_orders = [r for r in runner.order_log if r["power"] == "FRANCE"]
    assert all(not r["errors"] for r in france_orders)


def test_view_has_map_rules_and_no_model_only_sections():
    cfg = GameConfig(press_mode="full", max_year=1901, negotiation_rounds=1)
    human = HumanAgent("ITALY", "sos", cfg, port=0)
    captured = {}

    def ask(view):
        captured.update(view)
        return {"orders": [], "draw_vote": False}
    human.session.ask = ask
    agents = {p: RandomAgent(p, seed=1) for p in POWERS if p != "ITALY"} | {"ITALY": human}
    GameRunner(cfg, agents).run()
    assert captured["map_svg"].startswith("<svg")
    assert "TIMING RULES" in captured["rules"]
    assert "REPLY FORMAT" not in captured["rules"] and "YOUR MEMORY" not in captured["rules"]
    assert "would score" in captured["draw_shares"]


def test_waiting_has_no_timeout_but_close_releases_it():
    session = HumanSession("GERMANY", port=0)
    out = {}

    def wait():
        try:
            session.ask({"task_label": "test"})
        except HumanQuit:
            out["quit"] = True
    t = threading.Thread(target=wait)
    t.start()
    time.sleep(1.5)
    assert t.is_alive()  # still waiting: no timeout
    session.close()
    t.join(2)
    assert out.get("quit")


def test_stale_submission_is_refused():
    session = HumanSession("TURKEY", port=0)
    try:
        assert session.submit(5, {}) is not None  # nothing is waiting
    finally:
        session.close()
