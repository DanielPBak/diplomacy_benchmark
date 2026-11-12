"""One mixed-model game: one Claude seat against six local-model seats, identities hidden.

The Claude seat is chosen from the seed (or --claude-power). Every seat gets the same memory
window and output limit, sized for the smallest-context model, so memory and compute don't
differ between seats. Resumable with --resume; Claude spending is capped with --max-usd.

    ANTHROPIC_API_KEY=... python scripts/mixed_game.py --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.agents import LLMAgent  # noqa: E402
from dipbench.config import POWERS, GameConfig  # noqa: E402
from dipbench.llm import BudgetExceeded, CallLogger, LLMUnavailable, make_provider  # noqa: E402
from dipbench.loop import GameRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-model", default="claude-haiku-5-5")
    ap.add_argument("--claude-effort", default="medium")
    ap.add_argument("--claude-power", default=None, help="seat for Claude (default: chosen from --seed)")
    ap.add_argument("--claude-powers", nargs="+", default=None,
                    help="several Claude seats, or 'all' for an all-Claude table (overrides --claude-power)")
    ap.add_argument("--local-model", default="/Users/danbak/models/qwen3-8b-8bit")
    ap.add_argument("--local-identity", default="Qwen3-8B (8-bit MLX, thinking)")
    ap.add_argument("--local-base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--local-no-think", action="store_true")
    ap.add_argument("--press-mode", default="full", choices=["full", "public", "gunboat"])
    ap.add_argument("--scoring", default="dss", choices=["dss", "sos", "linear"])
    ap.add_argument("--identity-disclosure", default="hidden", choices=["hidden", "revealed", "false_labels"])
    ap.add_argument("--max-year", type=int, default=1920)
    ap.add_argument("--draw-vote-from-year", type=int, default=1901)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--output-limit", type=int, default=8192, help="per call, thinking included, same for every seat")
    ap.add_argument("--memory-window", type=int, default=None,
                    help="same for every seat; default 34000 (fits Qwen3-8B) with any local seat, else 200000")
    ap.add_argument("--max-workers", type=int, default=7)
    ap.add_argument("--max-usd", type=float, default=2.0, help="cap on Claude API spending for this game")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.claude_powers:
        claude_seats = list(POWERS) if args.claude_powers == ["all"] else [p.upper() for p in args.claude_powers]
    else:
        claude_seats = [(args.claude_power or random.Random(f"seat-{args.seed}").choice(POWERS)).upper()]
    all_claude = len(claude_seats) == len(POWERS)
    if args.memory_window is None:
        args.memory_window = 200_000 if all_claude else 34_000
    seat_tag = "all" if all_claude else "-".join(p.lower() for p in claude_seats)
    out = Path(args.out or f"runs/mixed_{args.claude_model}_{seat_tag}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.json"
    logger = CallLogger(path=out / "tokens.jsonl", resume=args.resume)
    logger.max_usd = logger.total_usd + args.max_usd if args.resume else args.max_usd

    cfg = GameConfig(press_mode=args.press_mode, max_year=args.max_year, negotiation_rounds=args.rounds,
                     draw_vote_from_year=args.draw_vote_from_year, identity_disclosure=args.identity_disclosure,
                     max_workers=args.max_workers, seed=args.seed, game_id=out.name)
    claude_spec = {"provider": "anthropic", "model": args.claude_model}
    local_spec = {"provider": "openai_compat", "model": args.local_model, "base_url": args.local_base_url,
                  "api_key_env": "LOCAL_LLM_API_KEY", "json_mode": "none",
                  "enable_thinking": False if args.local_no_think else None}
    agents = {}
    for p in POWERS:
        if p in claude_seats:
            agents[p] = LLMAgent(p, args.scoring, make_provider(claude_spec, logger), cfg, output_limit=args.output_limit,
                                 effort=args.claude_effort, identity=args.claude_model, memory_window=args.memory_window)
        else:
            agents[p] = LLMAgent(p, args.scoring, make_provider(local_spec, logger), cfg, output_limit=args.output_limit,
                                 effort=None, identity=args.local_identity, memory_window=args.memory_window)

    others = "no local seats" if all_claude else f"the other {len(POWERS) - len(claude_seats)} are {args.local_identity}"
    print(f"Claude ({args.claude_model}) plays {', '.join(claude_seats)}; {others}. Identities {args.identity_disclosure}; "
          f"memory window {args.memory_window:,}; cap {args.max_year}. Output -> {out}", flush=True)
    if args.resume and checkpoint.exists():
        runner = GameRunner.resume(checkpoint, agents, max_workers=args.max_workers)
        print(f"resuming from {runner.game.get_current_phase()}", flush=True)
    else:
        runner = GameRunner(cfg, agents, checkpoint)
    start = time.monotonic()
    try:
        result = runner.run()
    except (BudgetExceeded, LLMUnavailable) as e:
        print(f"STOPPED (resume with --resume): {e}")
        print(f"Claude spend so far ${logger.total_usd:.4f}")
        sys.exit(1)
    runner.save(out, result)
    checkpoint.unlink(missing_ok=True)
    (out / "seats.json").write_text(json.dumps({"claude_powers": claude_seats, "claude_model": args.claude_model,
                                                "local_model": None if all_claude else args.local_identity}, indent=2))

    s = logger.summary()
    claude_calls = [r for r in logger.records if r.power in claude_seats]
    print(f"\noutcome {result.outcome} at {result.final_phase} in {(time.monotonic() - start) / 60:.1f} min; "
          f"winners {result.winners}")
    print("final centers:", result.final_centers)
    claude_cost = sum(r.cost_usd for r in claude_calls)
    print(f"calls {s['calls']} (errors {s['errors']}); Claude calls {len(claude_calls)}, Claude cost ${claude_cost:.4f}, "
          f"Claude tokens in {sum(r.input_tokens + r.cache_read_tokens + r.cache_write_tokens for r in claude_calls):,} "
          f"out {sum(r.output_tokens for r in claude_calls):,}")
    for p in POWERS:
        rows = [r for r in runner.order_log if r["power"] == p]
        sent = [m for m in runner.press_log if m["sender"] == p]
        tag = "CLAUDE" if p in claude_seats else "qwen  "
        print(f"{tag} {p:8} centers {result.final_centers[p]:2}  submissions {len(rows)} (rejected "
              f"{sum(1 for r in rows if r['errors'])})  messages {len(sent)} (private/group "
              f"{sum(1 for m in sent if m['recipients'] != ['GLOBAL'])})  agent errors {len(agents[p].errors)}")
    print("draw votes:")
    for v in runner.vote_log:
        print(f"  {v['phase']}: yes {[p for p, b in v['ballots'].items() if b]}")


if __name__ == "__main__":
    main()
