"""Pilot: short LLM games in each press mode, then a token and cost report with per-game
extrapolations. Stops (and reports what it has) when the spending cap is reached.

    ANTHROPIC_API_KEY=... python scripts/pilot.py --max-usd 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.agents import LLMAgent  # noqa: E402
from dipbench.config import POWERS, GameConfig  # noqa: E402
from dipbench.llm import BudgetExceeded, CallLogger, LLMUnavailable  # noqa: E402
from dipbench.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from dipbench.llm.pricing import call_cost  # noqa: E402
from dipbench.loop import GameRunner  # noqa: E402

PRICE_SCENARIOS = {
    "7x claude-haiku-5-5": ["claude-haiku-5-5"] * 7,
    "7x claude-sonnet-5-5": ["claude-sonnet-5-5"] * 7,
    "7x claude-opus-5-5": ["claude-opus-5-5"] * 7,
    "1x claude-opus-5-5 + 6x claude-haiku-5-5": ["claude-opus-5-5"] + ["claude-haiku-5-5"] * 6,
    "1x claude-sonnet-5-5 + 6x claude-haiku-5-5": ["claude-sonnet-5-5"] + ["claude-haiku-5-5"] * 6,
}


def run_pilot_game(name, press_mode, models, years, out_dir, budget_left, effort, output_limit, rounds, resume):
    """Play (or, with resume, continue or skip) one pilot game. A game stopped by the spending
    cap or an API failure keeps its checkpoint and can be continued with --resume."""
    game_dir = out_dir / name
    game_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, result_path = game_dir / "checkpoint.json", game_dir / "result.json"
    logger = CallLogger(max_usd=None, path=game_dir / "tokens.jsonl", resume=resume)
    already_spent = logger.total_usd
    logger.max_usd = already_spent + budget_left
    cfg = GameConfig(press_mode=press_mode, max_year=1900 + years, negotiation_rounds=rounds, game_id=name)
    agents = {p: LLMAgent(p, "dss", AnthropicProvider(models[p], logger=logger), cfg,
                          output_limit=output_limit, effort=effort) for p in POWERS}

    if resume and result_path.exists():
        result = json.loads(result_path.read_text())
        phases = [ph["name"] for ph in json.loads((game_dir / "game.json").read_text())["phases"]]
        phases = [ph for ph in phases if ph != result["final_phase"]]
        status, outcome, resumes = "complete (from earlier run)", result["outcome"], result["resumes"]
        agent_errors = {}
    else:
        if resume and checkpoint.exists():
            runner = GameRunner.resume(checkpoint, agents)
        else:
            runner = GameRunner(cfg, agents, checkpoint)
        status, outcome = "complete", None
        try:
            result = runner.run()
            runner.save(game_dir, result)
            checkpoint.unlink(missing_ok=True)
            outcome = result.outcome
        except (BudgetExceeded, LLMUnavailable) as e:
            status = f"stopped, resumable with --resume: {e}"
        resumes = runner.resumes
        phases = [str(k) for k in runner.game.state_history.keys()]
        if runner.game.outcome:  # the phase a game ends in is recorded but never adjudicated
            phases = [ph for ph in phases if ph != runner.game.outcome[0]]
        agent_errors = {p: a.errors for p, a in agents.items() if a.errors}
    years_played = sum(1 for ph in phases if ph.endswith("M")) / 2  # two movement phases per year
    return {"name": name, "press_mode": press_mode, "models": models, "status": status, "outcome": outcome,
            "resumes": resumes, "phases_played": phases, "years_played": years_played,
            "summary": logger.summary(), "agent_errors": agent_errors,
            "records": [asdict(r) for r in logger.records]}


def reprice(records, models_by_power, out_ratio=None):
    """Cost if each power's calls had been made by another model with the same token counts
    (optionally scaling output tokens by a measured per-model ratio)."""
    total = 0.0
    for r in records:
        model = models_by_power[r["power"]]
        out = r["output_tokens"] * ((out_ratio or {}).get(model, 1.0))
        total += call_cost(model, r["input_tokens"], out, r["cache_read_tokens"], r["cache_write_tokens"])
    return total


def have_credentials() -> bool:
    """True if the Anthropic SDK can find an API key, auth token or login profile."""
    import anthropic
    client = anthropic.Anthropic()
    try:
        client._validate_headers(client.auth_headers, {})  # what the SDK checks before each request
        return True
    except TypeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["full", "public", "gunboat"])
    ap.add_argument("--sonnet-check", action="store_true", help="extra full-press game with FRANCE on Sonnet 5.5")
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--output-limit", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--target-years", type=int, default=20, help="game length to extrapolate to (1901-1920 = 20)")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--resume", action="store_true",
                    help="skip finished games and continue interrupted ones from their checkpoints")
    args = ap.parse_args()

    if not have_credentials():
        sys.exit("No Anthropic credentials found. Set ANTHROPIC_API_KEY in this shell "
                 "(or log in with `ant auth login`) and run again. Nothing was spent.")

    out_dir = Path(args.out)
    games, spent = [], 0.0
    plan = [(f"pilot_{m}", m, {p: args.model for p in POWERS}) for m in args.modes]
    if args.sonnet_check:
        plan.append(("pilot_full_sonnet_france", "full",
                     {p: ("claude-sonnet-5-5" if p == "FRANCE" else args.model) for p in POWERS}))
    for name, mode, models in plan:
        if spent >= args.max_usd:
            print(f"skipping {name}: budget used")
            continue
        print(f"running {name} ...", flush=True)
        g = run_pilot_game(name, mode, models, args.years, out_dir, args.max_usd - spent,
                           args.effort, args.output_limit, args.rounds, args.resume)
        spent += g["summary"]["cost_usd"]
        print(f"  {g['status']}, {g['summary']['calls']} calls, ${g['summary']['cost_usd']:.4f} (total ${spent:.4f})")
        games.append(g)

    # Measured output-token ratio of Sonnet vs the pilot model, from the Sonnet check game.
    out_ratio = {}
    check = next((g for g in games if g["name"] == "pilot_full_sonnet_france"), None)
    if check:
        def mean_out(recs):
            return sum(r["output_tokens"] for r in recs) / max(1, len(recs))
        son = [r for r in check["records"] if r["model"] == "claude-sonnet-5-5"]
        base = [r for r in check["records"] if r["model"] == args.model]
        if son and base and mean_out(base):
            out_ratio["claude-sonnet-5-5"] = mean_out(son) / mean_out(base)

    report = {"pilot_model": args.model, "years": args.years, "target_years": args.target_years,
              "total_spent_usd": spent, "sonnet_output_ratio": out_ratio.get("claude-sonnet-5-5"), "games": []}
    lines = [f"# Pilot report ({args.model}, {args.years} game-years per mode, {args.rounds} rounds)", "",
             f"Total spent: ${spent:.4f}", ""]
    for g in games:
        s, years = g["summary"], max(0.5, g["years_played"])
        scale = args.target_years / years
        est = {k: reprice(g["records"], dict(zip(POWERS, v)), out_ratio) * scale for k, v in PRICE_SCENARIOS.items()}
        report["games"].append({k: v for k, v in g.items() if k != "records"} | {"per_game_estimates_usd": est})
        lines += [f"## {g['name']} ({g['press_mode']}): {g['status']}",
                  f"- phases: {len(g['phases_played'])} over {g['years_played']:g} year(s); calls {s['calls']} "
                  f"(errors {s['errors']})",
                  f"- tokens: input {s['input_tokens']:,} + cache read {s['cache_read_tokens']:,} + cache write "
                  f"{s['cache_write_tokens']:,}; output {s['output_tokens']:,} (thinking {s['thinking_tokens']:,})",
                  f"- cost: ${s['cost_usd']:.4f} (${s['cost_usd'] / years:.4f} per game-year)",
                  "- by purpose: " + ", ".join(f"{k} {v['calls']} calls ${v['cost_usd']:.4f}"
                                               for k, v in s["by_purpose"].items()),
                  f"- estimated cost of a {args.target_years}-year game, same token counts repriced:"]
        lines += [f"  - {k}: ${v:.2f}" for k, v in est.items()]
        if g["agent_errors"]:
            lines.append(f"- agent errors: {sum(len(v) for v in g['agent_errors'].values())}")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "pilot_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
