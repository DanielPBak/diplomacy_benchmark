"""Prompt text for LLM agents.

The system prompt is fixed for a power for a whole game (rules, scoring rule, press mode,
budgets, disclosed identities), so providers can cache it. Each call's prompt carries the
board, history, press and the task for that call.
"""
from __future__ import annotations

from .. import scoring as scorers
from ..config import GameConfig
from ..state import PhaseContext
from ..tokens import CHARS_PER_TOKEN
from .map_reference import map_reference

SCORING_RULES = {
    "dss": """\
SCORING: Draw-Size Scoring (DSS).
- A solo victory (18 or more supply centers) scores 1 for the winner and 0 for everyone else.
- A draw scores 1/N for each of the N powers in the draw, whatever their center counts. Eliminated powers score 0.
- Centers don't matter in a draw; only how many powers share it.
Worked example: the game ends in a 3-way draw between you (6 centers), England (16) and Turkey (12). Each of you scores 1/3 = 0.333.
If it had ended in a 2-way draw instead, each would have scored 1/2 = 0.5. Eliminating a draw partner raises your score; so does a solo.""",
    "sos": """\
SCORING: Sum-of-Squares (SoS).
- A solo victory (18 or more supply centers) scores 1 for the winner and 0 for everyone else.
- In a draw, your score is (your centers)^2 divided by the sum of (centers)^2 over all powers in the draw. Eliminated powers score 0.
Worked example: a 3-way draw between you (16 centers), England (12) and Turkey (6). The sum of squares is 256 + 144 + 36 = 436.
You score 256/436 = 0.587, England 144/436 = 0.330, Turkey 36/436 = 0.083. Every center you hold at the end matters a lot,
and centers taken from your rivals matter twice: they raise your square and lower theirs.""",
    "linear": """\
SCORING: Linear (center count).
- A solo victory (18 or more supply centers) scores 1 for the winner and 0 for everyone else.
- Otherwise, when the game ends, your score is (your supply centers) / 34. Eliminated powers score 0.
Worked example: the game ends in a draw and you hold 10 centers: you score 10/34 = 0.294. With 12 centers you would score
12/34 = 0.353. How many powers share the draw doesn't affect your score; only your own center count at the end does.""",
}

SCORING_NAMES = {"dss": "Draw-Size Scoring", "sos": "Sum-of-Squares", "linear": "Linear"}

ZERO_SUM = {
    "dss": "ZERO-SUM: the game and its scoring are zero-sum; the seven powers' scores always add up to 1.",
    "sos": "ZERO-SUM: the game and its scoring are zero-sum; the seven powers' scores always add up to 1.",
    "linear": ("ZERO-SUM: the game and its scoring are zero-sum; the seven powers' scores add up to 1, less any "
               "unowned centers."),
}

MOVEMENT_RULES = """\
HOW MOVES RESOLVE (standard Diplomacy rules; all orders are revealed and resolved at the same time):
- Every unit has strength 1, plus 1 for each valid support it gets. A unit can support a move or a hold into a
  province it could itself move to.
- A move succeeds if its strength is greater than the strength of the unit holding the target province (or of every
  other unit moving there). Equal strengths bounce: nobody moves.
- A unit that is beaten while holding is dislodged and must retreat to an empty adjacent province or be disbanded.
- A support is cut if the supporting unit is attacked from any province other than the one it is supporting into.
- You cannot dislodge your own units. Units belonging to other powers can be attacked at any time; agreements are
  not enforced by the rules.
- Growing beyond the unowned centers means taking supply centers that other powers own; to take a defended center
  you need more strength than its defender."""

ORDER_SYNTAX = """\
ORDER FORMAT (copy orders exactly from your list of legal orders):
- Hold: "A PAR H". Move: "A PAR - BUR". Support a hold: "A MAR S A PAR". Support a move: "A MAR S A PAR - BUR".
- Convoy: "F ENG C A LON - BRE", with the army ordered "A LON - BRE VIA".
- Retreat: "A MUN R BOH". Disband during retreats: "A MUN D".
- Winter builds: "A PAR B", "F STP/SC B". Winter disband: "A PAR D". Decline a build: "WAIVE".
Units without a valid order hold. Unordered retreating units are disbanded. Missing Winter disbands are chosen for you."""


def _press_rules(cfg: GameConfig) -> str:
    rounds = cfg.negotiation_rounds
    timing = (f"Every phase (Spring and Fall moves, retreats, Winter adjustments) has {rounds} negotiation "
              f"round{'s' if rounds != 1 else ''} before orders. In each round every surviving power writes at the "
              "same time; messages are delivered only after everyone has written, so a reply takes one round. "
              "Messages sent in the last round are read when everyone writes orders. Nothing anyone says is binding.")
    budget = ""
    if cfg.press_token_budget is not None:
        budget = (f"\nPRESS BUDGET: you may send up to {cfg.press_token_budget} tokens of message text per "
                  f"{cfg.press_budget_scope} (1 token = {CHARS_PER_TOKEN} characters; a message to several powers "
                  "or to everyone costs its length once). A message that doesn't fit your remaining budget is "
                  "rejected and not delivered. Unused budget doesn't carry over.")
    if cfg.press_mode == "full":
        group = ("a group of powers (every recipient sees the full recipient list), " if cfg.allow_group_messages
                 else "")
        no_group = "" if cfg.allow_group_messages else "Group messages are not allowed. "
        return (f"COMMUNICATION: full press. You may message one power privately, {group}or GLOBAL (all surviving "
                f"powers). {no_group}Only the sender and recipients ever see a message. " + timing + budget)
    if cfg.press_mode == "public":
        return ("COMMUNICATION: public press. Every message is public: the only allowed recipient is GLOBAL, and every "
                "surviving power reads every message. Private and group messages are rejected. " + timing + budget)
    return ("COMMUNICATION: none. This is a gunboat game: no power can send or receive messages of any kind, and there "
            "is no negotiation. You can only infer other powers' intentions from their moves and from draw votes.")


def _vote_rules(cfg: GameConfig) -> str:
    if not cfg.draw_votes:
        return "DRAWS: there is no draw voting in this game."
    visibility = ("Ballots are public: you will see every surviving power's most recent vote."
                  if not cfg.secret_ballots else
                  "Ballots are secret: you only see your own vote, and other powers never see yours.")
    start = (f" Votes count from {cfg.draw_vote_from_year} on." if cfg.draw_vote_from_year > 1901 else "")
    return ("DRAW VOTES: every order submission in a movement or Winter phase includes a draw vote (true or false). "
            "If every surviving power votes true in the same phase, the game ends immediately as a draw among all "
            "surviving powers, before that phase's orders are carried out. Retreat phases have no vote. "
            + visibility + start)


def memory_rules(memory_window: int, diary_max_chars: int) -> str:
    return f"""\
YOUR MEMORY: this whole game is one continuous conversation with you. Each time it is your turn, you are sent only
what is new since your last turn (results of phases that just finished, new messages to you, the current board and
your task), and you can scroll back through every earlier turn: what you were sent and every reply you wrote.
You CANNOT see your own earlier private reasoning or thinking, only your replies. So anything you will need later
(plans, deals, promises made to you or by you, suspicions) must be written down in your replies.
Every order submission has a "diary" field (at most {diary_max_chars} characters): a short record of this phase,
covering agreements, promises, betrayals, and your plans. Your memory holds about {memory_window:,} tokens. If the
game grows past that, the conversation in your oldest turns (the messages and your replies) is removed. The full
record of every phase's orders and results is always kept, and so are your diary entries, which replace those turns.
Keep diary entries informative: for old negotiations, they may be all you remember."""


def system_prompt(power: str, scoring: str, cfg: GameConfig, output_limit: int,
                  opponent_identities: dict[str, str] | None, memory_window: int = 200_000,
                  diary_max_chars: int = 800) -> str:
    if opponent_identities is None:
        identities = "You are not told which AI model or person plays each other power."
    else:
        lines = "\n".join(f"- {p}: {label}" for p, label in sorted(opponent_identities.items()))
        identities = "The other powers are played by:\n" + lines
    return f"""\
You are playing {power} in a game of Diplomacy (standard map, seven powers) against six other players.
Your goal is to maximize your expected score under the scoring rule below. Nothing else counts.

{SCORING_RULES[scoring]}

{ZERO_SUM[scoring]}

GAME END: the game ends with a solo (18+ centers), a unanimous draw vote, or after Winter {cfg.told_max_year}. If it
reaches the end of {cfg.told_max_year}, it ends as a draw among all surviving powers and is scored like any other draw.

TIMING RULES: supply centers change owner only at the end of each Fall turn (after Fall moves and retreats). A
center you move into in Spring is not yours until you are still there at the end of Fall, and a center you leave
stays yours until another power ends a Fall turn in it. If a draw vote passes, the game ends at once, before that
phase's orders are carried out, and the draw is scored on the centers owned at that moment. A power is eliminated
when it has no supply centers and no units left.

{_press_rules(cfg)}

{_vote_rules(cfg)}

OPPONENTS: {identities}

LIMITS: each of your responses, including any thinking, is limited to {output_limit} output tokens. A response that
runs past the limit is lost.

REPLY FORMAT: every reply is one JSON object with the fields messages, orders, draw_vote and diary. In a negotiation
round, fill "messages" and leave the rest empty (orders [], draw_vote false, diary ""). When submitting orders, fill
"orders", "draw_vote" and "diary" and leave "messages" empty; messages sent with orders are not delivered.

{MOVEMENT_RULES}

{ORDER_SYNTAX}

{map_reference()}

{memory_rules(memory_window, diary_max_chars)}"""


# ----------------------------------------------------------------- per-call prompts

def _draw_shares(ctx: PhaseContext, scoring: str) -> str:
    """What each survivor would score under `scoring` if a draw passed right now."""
    counts = {p: len(cs) for p, cs in ctx.centers.items()}
    alive = [p for p in ctx.units if ctx.units[p] or ctx.centers[p]]
    now = {"outcome": "draw_vote", "winners": alive, "final_centers": counts, "survivors": alive}
    shares = scorers.SCORERS[scoring](now)
    rows = ", ".join(f"{p} {shares[p]:.3f}" for p in sorted(alive))
    return (f"IF A DRAW PASSED NOW ({prompts_name(scoring)}, on current center ownership): {rows}. "
            f"You ({ctx.power}) would score {shares[ctx.power]:.3f}.")


def prompts_name(scoring: str) -> str:
    return SCORING_NAMES[scoring]


def _board(ctx: PhaseContext) -> str:
    owned = {c for cs in ctx.centers.values() for c in cs}
    rows = []
    for p in sorted(ctx.units):
        units, centers = ctx.units[p], ctx.centers[p]
        if not units and not centers:
            rows.append(f"{p}: eliminated")
            continue
        rows.append(f"{p} ({len(centers)} centers: {', '.join(sorted(centers)) or '-'}) units: {', '.join(units) or '-'}")
    unowned = [c for c in ctx.supply_centers if c not in owned]
    rows.append(f"Unowned centers: {', '.join(unowned) or '-'}")
    return "\n".join(rows)


def _history(entries: list[dict]) -> str:
    out = []
    for h in entries:
        lines = []
        for p in sorted(h["orders"]):
            parts = []
            for o in h["orders"][p] or []:
                unit = " ".join(o.split()[:2])
                res = h["results"].get(unit) or []
                parts.append(o + (f" [{', '.join(res)}]" if res else ""))
            if parts:
                lines.append(f"  {p}: " + "; ".join(parts))
        out.append(f"{h['phase']}:\n" + ("\n".join(lines) if lines else "  (no orders)"))
    return "\n".join(out)


def _messages(messages: list) -> str:
    lines = []
    for m in messages:
        to = "everyone" if m.recipients == ["GLOBAL"] else ", ".join(m.recipients)
        lines.append(f"[{m.phase} round {m.round + 1}] {m.sender} -> {to}: {m.text}")
    return "\n".join(lines)


def _votes(ctx: PhaseContext) -> str:
    if ctx.config is not None and not ctx.config.draw_votes:
        return ""
    if not ctx.votes_held:
        return "No draw vote has been held yet."
    if ctx.draw_votes is None:
        return f"Your most recent draw vote: {str(ctx.my_draw_vote).lower()} (ballots are secret)."
    votes = ", ".join(f"{p}={'yes' if v else 'no'}" for p, v in sorted(ctx.draw_votes.items()))
    return f"Most recent draw votes: {votes}."


def _legal(ctx: PhaseContext) -> str:
    if not ctx.legal_orders:
        return "(you have nothing to order this phase)"
    return "\n".join(f"{loc}: " + " | ".join(opts) for loc, opts in ctx.legal_orders.items())


def _phase_name(ctx: PhaseContext) -> str:
    season = {"S": "Spring", "F": "Fall", "W": "Winter"}[ctx.phase[0]]
    kind = {"M": "movement", "R": "retreats", "A": "adjustments"}[ctx.phase_type]
    return f"{ctx.phase} ({season} {ctx.year}, {kind})"


def _update(ctx: PhaseContext, new_history: list[dict], new_messages: list, first_turn: bool,
            show_board: bool, show_votes: bool, scoring: str | None = None, show_shares: bool = True,
            undelivered: list[dict] | None = None) -> str:
    """Only what is new since this power's last turn. The board is shown when it may have
    changed (a power's first turn in a phase), and votes only when they changed."""
    parts = [("The game begins. " if first_turn else "") + f"Current phase: {_phase_name(ctx)}."]
    if new_history:
        parts.append("PHASES COMPLETED SINCE YOUR LAST TURN ([..] marks orders that failed or units dislodged):\n"
                     + _history(new_history))
    if ctx.press_mode != "gunboat":
        # The power's own messages aren't repeated: they're already in its own earlier replies.
        parts.append("NEW MESSAGES TO YOU SINCE YOUR LAST TURN:\n" + (_messages(new_messages) or "(none)"))
        if undelivered:
            parts.append("SOME OF YOUR MESSAGES WERE NOT DELIVERED:\n" + "\n".join(
                f"- to {', '.join(r['recipients'])}: {r['reason']} (\"{r['text'][:80]}\")" for r in undelivered))
    if show_board:
        parts.append(f"BOARD NOW:\n{_board(ctx)}")
    # Draw shares change only when center ownership does, so they're shown only then.
    if show_shares and scoring and ctx.config is not None and ctx.config.draw_votes:
        parts.append(_draw_shares(ctx, scoring))
    votes = _votes(ctx)
    if votes and show_votes:
        parts.append(votes)
    return "\n\n".join(parts)


LEGAL_ABOVE = "(your legal orders for this phase are listed earlier in this phase's conversation)"


def negotiate_prompt(ctx: PhaseContext, new_history: list[dict], new_messages: list, first_turn: bool,
                     show_board: bool = True, show_votes: bool = True, show_legal: bool = True,
                     scoring: str | None = None, show_shares: bool = True,
                     undelivered: list[dict] | None = None) -> str:
    rounds = ctx.config.negotiation_rounds if ctx.config else 1
    budget = ctx.press_budget
    budget_line = (f"You have {budget['remaining']} of {budget['limit']} press tokens left this {budget['scope']} "
                   f"({CHARS_PER_TOKEN} characters per token)." if budget else "There is no press budget.")
    recipients = ('"GLOBAL" only (public press)' if ctx.press_mode == "public"
                  else 'one or more power names, or ["GLOBAL"] for everyone')
    legal = (f"\n\nYOUR ORDERABLE UNITS AND LEGAL ORDERS THIS PHASE:\n{_legal(ctx)}"
             if show_legal else "")
    return (_update(ctx, new_history, new_messages, first_turn, show_board, show_votes, scoring, show_shares,
                    undelivered) + legal
            + f"\n\nTASK: negotiation round {ctx.negotiation_round + 1} of {rounds} for {ctx.phase}. {budget_line} "
              f"Write any messages you want to send now (an empty list is fine). Recipients: {recipients}.")


def orders_prompt(ctx: PhaseContext, new_history: list[dict], new_messages: list, first_turn: bool,
                  diary_max_chars: int, show_board: bool = True, show_votes: bool = True,
                  show_legal: bool = True, scoring: str | None = None, show_shares: bool = True,
                  undelivered: list[dict] | None = None) -> str:
    if ctx.phase_type == "A":
        delta = ctx.adjustment_delta
        need = (f"You may build up to {delta} unit(s)." if delta > 0 else
                f"You must disband {-delta} unit(s)." if delta < 0 else "You have no builds or disbands.")
    elif ctx.phase_type == "R":
        need = "Order each of your dislodged units to retreat or disband."
    else:
        need = "Give one order per unit."
    vote = ("Set draw_vote to true to accept a draw shared by all surviving powers now, false otherwise."
            if ctx.vote_phase else "This is a retreat phase: there is no draw vote; set draw_vote to false.")
    head = _update(ctx, new_history, new_messages, first_turn, show_board, show_votes, scoring, show_shares,
                   undelivered)
    if ctx.feedback:
        head = ("YOUR PREVIOUS SUBMISSION WAS REJECTED:\n- " + "\n- ".join(ctx.feedback)
                + "\nFix these problems and submit again.\n\n" + head)
    legal = f"YOUR ORDERABLE UNITS AND LEGAL ORDERS THIS PHASE:\n{_legal(ctx)}" if show_legal else LEGAL_ABOVE
    return (head + "\n\n" + legal
            + f"\n\nTASK: submit your orders for {ctx.phase}. {need} Copy orders exactly from the legal list. "
              f"{vote} Write your diary entry for this phase (at most {diary_max_chars} characters).")


def memory_digest(entries: list[tuple[str, str]], board_history: list[dict], first_phase: str, last_phase: str) -> str:
    lines = "\n".join(f"- {phase}: {text}" for phase, text in entries) or "- (no diary entries)"
    board = _history(board_history) if board_history else "(no phases completed in that period)"
    return (f"[MEMORY] The conversation in your turns from {first_phase} through {last_phase} (messages and your "
            "replies) was removed to fit your memory window. The game record is kept in full.\n\n"
            "ALL ORDERS AND RESULTS UP TO THAT POINT ([..] marks orders that failed or units dislodged):\n"
            f"{board}\n\nYOUR DIARY ENTRIES FROM THOSE TURNS:\n{lines}")


# One schema for every reply. The schema is part of what providers cache, so negotiation
# and order turns sharing it keeps each agent's conversation on a single cache chain.
TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {"type": "array", "items": {
            "type": "object",
            "properties": {"recipients": {"type": "array", "items": {"type": "string"}},
                           "text": {"type": "string"}},
            "required": ["recipients", "text"], "additionalProperties": False}},
        "orders": {"type": "array", "items": {"type": "string"}},
        "draw_vote": {"type": "boolean"},
        "diary": {"type": "string"},
    },
    "required": ["messages", "orders", "draw_vote", "diary"],
    "additionalProperties": False,
}
NEGOTIATE_SCHEMA = ORDERS_SCHEMA = TURN_SCHEMA
