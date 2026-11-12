"""A human player, through a small local web page.

Each time it is the human's turn (a negotiation round or an order submission), the agent
publishes a view of the game to its page and blocks until the human submits. There is no
timeout: the whole game waits, however long the human takes.

The page is served on 127.0.0.1 only. Open the URL printed at the start of the game.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from diplomacy import Game
from diplomacy.engine.renderer import Renderer

from ..config import POWERS
from ..state import PhaseContext
from . import prompts
from .base import Agent, OrderSubmission, OutgoingMessage

PAGE = Path(__file__).with_name("human_page.html")


class HumanQuit(Exception):
    """Raised in the waiting game thread when the session is closed (e.g. Ctrl-C)."""


class HumanSession:
    """The web page's state: the request waiting for the human, and their reply."""

    def __init__(self, power: str, host: str = "127.0.0.1", port: int = 8765):
        self.power = power
        self._lock = threading.Lock()
        self._replied = threading.Event()
        self._closed = False
        self.view: dict | None = None    # the latest request (kept while waiting for others)
        self.waiting = False              # True while a request awaits the human's reply
        self.seq = 0
        self._reply: dict | None = None
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.url = f"http://{host}:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def ask(self, view: dict) -> dict:
        """Publish `view` and block until the human replies. No timeout."""
        with self._lock:
            self.seq += 1
            self.view = {**view, "seq": self.seq}
            self.waiting = True
            self._reply = None
            self._replied.clear()
        print(f"[{self.power}] your turn: {view['task_label']} -> {self.url}", flush=True)
        while not self._replied.wait(0.5):
            if self._closed:
                raise HumanQuit(f"{self.power}: session closed while waiting for the human")
        with self._lock:
            self.waiting = False
            return self._reply or {}

    def submit(self, seq: int, reply: dict) -> str | None:
        """Accept a reply for request `seq`; returns an error message, or None."""
        with self._lock:
            if not self.waiting or seq != self.seq:
                return "this turn is no longer open; reload the page"
            self._reply = reply
            self._replied.set()
        return None

    def close(self):
        self._closed = True
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        session = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the game's console readable
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif self.path.startswith("/state"):
                    with session._lock:
                        state = {"waiting": session.waiting, "view": session.view}
                    self._send(200, json.dumps(state).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/submit":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    error = session.submit(int(body["seq"]), body.get("reply") or {})
                except (ValueError, KeyError, TypeError) as e:
                    error = f"bad request: {e}"
                self._send(200 if error is None else 409, json.dumps({"error": error}).encode(),
                           "application/json")

        return Handler


def render_map(ctx: PhaseContext) -> str:
    """SVG of the current board, drawn by the engine's renderer."""
    game = Game()
    for power in POWERS:
        game.set_units(power, [u.lstrip("*") for u in ctx.units.get(power, [])], reset=True)
        game.set_centers(power, ctx.centers.get(power, []), reset=True)
    svg = Renderer(game).render(incl_orders=False, incl_abbrev=True)
    return svg[svg.find("<svg"):]  # drop the XML prolog and DOCTYPE so it can be inlined


class HumanAgent(Agent):
    name = "human"

    def __init__(self, power: str, scoring: str = "dss", config=None, port: int = 8765,
                 identity: str | None = None, session: HumanSession | None = None, **kwargs):
        super().__init__(power, scoring, identity=identity or "human")
        self.config = config
        self.session = session or HumanSession(power, port=port)
        self._rules: str | None = None

    def close(self):
        self.session.close()

    # ---------- views ----------

    def _rules_text(self, ctx: PhaseContext) -> str:
        if self._rules is None:
            # The same rules the AI players get, minus the parts about model output and memory.
            text = prompts.system_prompt(self.power, self.scoring, self.config, 0, ctx.opponent_identities)
            skip = ("LIMITS:", "REPLY FORMAT:", "YOUR MEMORY:")
            self._rules = "\n\n".join(p for p in text.split("\n\n") if not p.startswith(skip))
        return self._rules

    def _view(self, ctx: PhaseContext, kind: str) -> dict:
        cfg = self.config
        alive = sorted(p for p in POWERS if ctx.units.get(p) or ctx.centers.get(p))
        if kind == "negotiate":
            label = (f"{ctx.phase} negotiation round {ctx.negotiation_round + 1} of "
                     f"{cfg.negotiation_rounds}")
        else:
            label = f"{ctx.phase} orders" + (" (resubmit: fix the errors shown)" if ctx.feedback else "")
        return {
            "kind": kind, "task_label": label, "power": self.power, "scoring": self.scoring,
            "phase": ctx.phase, "phase_type": ctx.phase_type, "year": ctx.year,
            "press_mode": ctx.press_mode, "allow_group": cfg.allow_group_messages,
            "recipients": [p for p in alive if p != self.power],
            "press_budget": ctx.press_budget,
            "units": ctx.units, "centers": ctx.centers,
            "unowned": sorted(set(ctx.supply_centers) - {c for cs in ctx.centers.values() for c in cs}),
            "history": ctx.history,
            "messages": [m.__dict__ for m in ctx.messages],
            "rejected": ctx.rejected_messages,
            "legal_orders": ctx.legal_orders, "adjustment_delta": ctx.adjustment_delta,
            "vote_phase": ctx.vote_phase, "votes_held": ctx.votes_held,
            "my_draw_vote": ctx.my_draw_vote, "draw_votes": ctx.draw_votes,
            "draw_shares": prompts._draw_shares(ctx, self.scoring),
            "secret_ballots": cfg.secret_ballots,
            "opponent_identities": ctx.opponent_identities,
            "feedback": ctx.feedback,
            "rules": self._rules_text(ctx),
            "map_svg": render_map(ctx),
        }

    # ---------- turns ----------

    def negotiate(self, ctx: PhaseContext) -> list[OutgoingMessage]:
        reply = self.session.ask(self._view(ctx, "negotiate"))
        out = []
        for m in reply.get("messages") or []:
            if isinstance(m, dict) and isinstance(m.get("text"), str) and m["text"].strip() and m.get("recipients"):
                out.append(OutgoingMessage(list(m["recipients"]), m["text"].strip()))
        return out

    def orders(self, ctx: PhaseContext) -> OrderSubmission:
        reply = self.session.ask(self._view(ctx, "orders"))
        orders = [o for o in reply.get("orders") or [] if isinstance(o, str) and o]
        vote = reply.get("draw_vote")
        return OrderSubmission(orders, vote if isinstance(vote, bool) else None)

