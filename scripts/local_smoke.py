"""Smoke test for a local model behind an OpenAI-compatible server (MLX, Ollama, vLLM, ...).

By default one power (FRANCE) is the local model and the other six are random agents,
so the run is short: it checks that the model returns valid JSON, legal orders, messages
and draw votes. Zero API cost.

    python scripts/local_smoke.py --base-url http://localhost:8080/v1 --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.agents import LLMAgent, RandomAgent  # noqa: E402
from dipbench.config import POWERS, GameConfig  # noqa: E402
from dipbench.llm import CallLogger, make_provider  # noqa: E402
from dipbench.loop import GameRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="model name sent to the server; MLX loads whatever this names, so use the served path")
    ap.add_argument("--identity", default=None,
                    help="label recorded for this model in results, e.g. 'Qwen3-8B (8-bit MLX)'; defaults to --model")
    ap.add_argument("--json-mode", default="none", choices=["json_schema", "json_object", "none"])
    ap.add_argument("--press-mode", default="full", choices=["full", "public", "gunboat"])
    ap.add_argument("--powers", nargs="+", default=["FRANCE"], help="powers played by the local model ('all' for every power)")
    ap.add_argument("--max-year", type=int, default=1901)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--output-limit", type=int, default=8192)
    ap.add_argument("--memory-window", type=int, default=34000,
                    help="memory window in model-neutral tokens; Qwen3-8B's context is 40,960, so leave headroom")
    ap.add_argument("--out", default="runs/local_smoke")
    ap.add_argument("--no-think", action="store_true", help="turn off the model's reasoning (much faster)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted game from its checkpoint")
    ap.add_argument("--max-workers", type=int, default=7,
                    help="concurrent model calls; lower it if a local server runs out of memory")
    args = ap.parse_args()

    local_powers = POWERS if args.powers == ["all"] else [p.upper() for p in args.powers]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = CallLogger(path=out / "tokens.jsonl", resume=args.resume)
    cfg = GameConfig(press_mode=args.press_mode, max_year=args.max_year, negotiation_rounds=args.rounds,
                     game_id="local_smoke", max_workers=args.max_workers)
    spec = {"provider": "openai_compat", "model": args.model, "base_url": args.base_url,
            "api_key_env": "LOCAL_LLM_API_KEY", "json_mode": args.json_mode,
            "enable_thinking": False if args.no_think else None}
    agents = {p: (LLMAgent(p, "dss", make_provider(spec, logger), cfg, output_limit=args.output_limit, effort=None,
                           memory_window=args.memory_window, identity=args.identity)
                  if p in local_powers else RandomAgent(p, seed=1)) for p in POWERS}

    start = time.monotonic()
    checkpoint = out / "checkpoint.json"
    if args.resume and checkpoint.exists():
        runner = GameRunner.resume(checkpoint, agents, max_workers=args.max_workers)
        print(f"resuming from {runner.game.get_current_phase()}")
    else:
        runner = GameRunner(cfg, agents, checkpoint)
    result = runner.run()
    runner.save(out, result)
    checkpoint.unlink(missing_ok=True)
    elapsed = time.monotonic() - start

    print(f"outcome {result.outcome} at {result.final_phase} in {elapsed:.0f}s")
    s = logger.summary()
    print(f"local calls {s['calls']}, model errors {s['errors']}, prompt tokens {s['input_tokens'] + s['cache_read_tokens']:,}, "
          f"output tokens {s['output_tokens']:,}")
    for r in logger.records:
        print(f"  {r.phase} {r.power} {r.purpose}: {r.output_tokens} out tokens, {r.latency_s:.1f}s"
              + (f"  ERROR {r.error}" if r.error else ""))
    for p in local_powers:
        rows = [r for r in runner.order_log if r["power"] == p]
        bad = [r for r in rows if r["errors"]]
        print(f"{p}: {len(rows)} order submissions, {len(bad)} rejected; messages sent "
              f"{sum(1 for m in runner.press_log if m['sender'] == p)}; agent errors {len(agents[p].errors)}")
        for r in bad[:3]:
            print("   rejected:", json.dumps(r["errors"])[:300])
        diary = agents[p].diary
        print(f"   diary entries {len(diary)}, memory trims {len(agents[p].trims)}; last entry: "
              f"{(diary[-1][1][:300] if diary else '')!r}")


if __name__ == "__main__":
    main()
