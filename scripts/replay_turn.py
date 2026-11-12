"""Re-ask one saved order turn, to debug why an agent chose what it did.

Rebuilds the exact system prompt and conversation the agent saw for its order turn in a phase,
then sends it again with optional overrides (model, effort, provider, extra system text) and
prints what comes back. Nothing is written to the game.

    python scripts/replay_turn.py runs/<run>/game_000 RUSSIA S1906M --n 3
    python scripts/replay_turn.py runs/<run>/game_000 RUSSIA S1906M --effort high
    python scripts/replay_turn.py runs/<run>/game_000 RUSSIA S1906M --model opus --n 2
    python scripts/replay_turn.py runs/<run>/game_000 RUSSIA S1906M --append "Your opponents are AI models."
    python scripts/replay_turn.py runs/<run>/game_000 ENGLAND W1905A --set announced_max_year=1908

--set changes only the system prompt; the turn prompts are replayed as they were sent.

--model names a model key from the run's config (e.g. opus, haiku). The original seat's provider
settings are used otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.agents import prompts  # noqa: E402
from dipbench.agents.llm_agent import LLMAgent  # noqa: E402
from dipbench.experiment import game_config, plan_games  # noqa: E402
from dipbench.llm import CallLogger, make_provider  # noqa: E402
from dipbench.llm.base import LLMRequest  # noqa: E402

PROVIDER_KEYS = {"provider", "model", "base_url", "api_key_env", "json_mode", "enable_thinking", "cache_ttl",
                 "billing", "cli", "extra_args", "thinking_budget"}


def order_turn_index(turns: list[dict], phase: str) -> int:
    """The last turn in `phase` whose reply has orders or a diary (the order submission)."""
    for i in range(len(turns) - 1, -1, -1):
        t = turns[i]
        if t["phase"] != phase:
            continue
        try:
            d = json.loads(t["assistant"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and (d.get("orders") or d.get("diary")):
            return i
    raise SystemExit(f"no order turn found for {phase}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_dir")
    ap.add_argument("power")
    ap.add_argument("phase")
    ap.add_argument("--n", type=int, default=1, help="how many times to re-ask")
    ap.add_argument("--model", default=None, help="model key from the run config to use instead")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--append", default=None, help="text appended to the system prompt")
    ap.add_argument("--set", nargs="+", default=[], metavar="FIELD=VALUE",
                    help="override GameConfig fields in the system prompt, e.g. announced_max_year=1908")
    args = ap.parse_args()

    gdir = Path(args.game_dir)
    power = args.power.upper()
    cfg = json.loads((gdir.parent / "config.json").read_text())
    planned = next(g for g in plan_games(cfg) if g.game_id == gdir.name)
    seats = json.loads((gdir / "seats.json").read_text())
    gcfg = game_config(cfg, planned)
    for item in args.set:
        field, value = item.split("=", 1)
        setattr(gcfg, field, json.loads(value) if value[:1] in "0123456789tfn[{\"" else value)
    seat = planned.seats[power]
    agent_cfg = cfg.get("agent", {})

    state_file = gdir / "agents.json"
    states = json.loads(state_file.read_text()) if state_file.exists() else \
        json.loads((gdir / "checkpoint.json").read_text())["agents"]
    state = states[power]
    idx = order_turn_index(state["turns"], args.phase)
    original = json.loads(state["turns"][idx]["assistant"])

    # Rebuild the conversation as it was sent: restore the agent with only the turns before this one.
    output_limit = int(agent_cfg.get("output_limit", 4096))
    agent = LLMAgent(power, seat.scoring, None, gcfg, output_limit=output_limit, identity="replay",
                     memory_window=seats["memory_window"],
                     diary_max_chars=int(agent_cfg.get("diary_max_chars", 800)))
    agent.set_state(state)
    agent.turns = state["turns"][:idx]
    # Only trims made before this turn apply. A trim is logged with the phase it happened in, so
    # trims in earlier phases count; one in this same phase may or may not have preceded the turn.
    earlier = {t["phase"] for t in agent.turns} - {args.phase}
    agent.trimmed_through = max([tr["total_trimmed"] for tr in state.get("trims", []) if tr["phase"] in earlier],
                                default=0)
    if any(tr["phase"] == args.phase for tr in state.get("trims", [])):
        print("note: memory was trimmed during this phase; the replayed memory may differ slightly")
    diary_phases = {t["phase"] for t in agent.turns}
    agent.diary = [e for e in state.get("diary", []) if e[0] in diary_phases]
    agent.board_log = state["board_log"][:state["turns"][idx]["board_upto"]]
    history = agent._messages()

    identities = None
    if gcfg.identity_disclosure != "hidden":
        identities = {p: s["identity"] for p, s in seats["seats"].items() if p != power}
    system = prompts.system_prompt(power, seat.scoring, gcfg, output_limit, identities,
                                   seats["memory_window"], int(agent_cfg.get("diary_max_chars", 800)))
    spec = cfg["models"][args.model or seat.model]
    if args.append:
        system += "\n\n" + args.append
    elif spec.get("thinking_hint"):
        system += "\n\nTHINKING: " + spec["thinking_hint"]

    logger = CallLogger()
    provider = make_provider({k: v for k, v in spec.items() if k in PROVIDER_KEYS}, logger)
    effort = args.effort or spec.get("effort")
    print(f"{power} {args.phase}: replaying turn {idx} with {spec['model']} (effort {effort}), "
          f"{len(history) // 2} earlier turns")
    print(f"ORIGINAL: vote={original.get('draw_vote')} orders={original.get('orders')}")
    print(f"  diary: {original.get('diary', '')[:400]}\n")

    for i in range(args.n):
        req = LLMRequest(system=system, prompt=state["turns"][idx]["user"], schema=prompts.TURN_SCHEMA,
                         history=history, max_output_tokens=output_limit, effort=effort,
                         game_id="replay", power=power, phase=args.phase, purpose="replay")
        res = provider.complete(req)
        r = res.record
        d = res.data or {}
        holds = sum(1 for o in d.get("orders") or [] if o.endswith(" H"))
        print(f"REPLAY {i + 1}: vote={d.get('draw_vote')} holds={holds}/{len(d.get('orders') or [])} "
              f"orders={d.get('orders')}")
        print(f"  out={r.output_tokens} thinking={r.thinking_tokens} error={r.error}")
        print(f"  diary: {d.get('diary') or ''}")
        if res.reasoning:
            print(f"  reasoning: {res.reasoning[:1500]}")
        print()


if __name__ == "__main__":
    main()
