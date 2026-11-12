"""Run every game in an experiment config (see dipbench/experiment.py for the format).

    ANTHROPIC_API_KEY=... python scripts/run.py configs/haiku_x7.json

Outputs go to runs/<name>/game_NNN/. Finished games are skipped and interrupted ones resume
from their checkpoints, so rerunning the same command continues the experiment. The
max_usd cap covers the whole experiment, including spending in earlier runs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.agents import HumanAgent, LLMAgent  # noqa: E402
from dipbench.config import POWERS  # noqa: E402
from dipbench.experiment import game_config, load_config, memory_window, plan_games  # noqa: E402
from dipbench.llm import BudgetExceeded, CallLogger, LLMUnavailable, make_provider  # noqa: E402
from dipbench.loop import GameRunner  # noqa: E402
from dipbench.scoring import score_all  # noqa: E402

PROVIDER_KEYS = {"provider", "model", "base_url", "api_key_env", "json_mode", "enable_thinking", "cache_ttl",
                 "billing", "cli", "extra_args", "thinking_budget"}


def spent_so_far(out: Path) -> float:
    total = 0.0
    for f in out.glob("game_*/tokens.jsonl"):
        total += sum(json.loads(l)["cost_usd"] for l in f.read_text().splitlines() if l)
    return total


def build_agents(cfg, planned, gcfg, logger):
    window = memory_window(cfg, planned)
    agent_cfg = cfg.get("agent", {})
    agents = {}
    human_ports = 0
    for power, seat in planned.seats.items():
        spec = cfg["models"][seat.model]
        if spec.get("provider") == "human":
            # Each human seat gets its own page; several humans use consecutive ports.
            agents[power] = HumanAgent(power, seat.scoring, gcfg, port=int(spec.get("port", 8765)) + human_ports,
                                       identity=spec.get("identity", "human"))
            human_ports += 1
            print(f"{power} is played by a human: open {agents[power].session.url}")
            continue
        provider = make_provider({k: v for k, v in spec.items() if k in PROVIDER_KEYS}, logger)
        agents[power] = LLMAgent(power, seat.scoring, provider, gcfg,
                                 output_limit=int(agent_cfg.get("output_limit", 4096)),
                                 effort=spec.get("effort"), identity=spec.get("identity", spec["model"]),
                                 memory_window=window, diary_max_chars=int(agent_cfg.get("diary_max_chars", 800)),
                                 thinking_hint=spec.get("thinking_hint"))
    return agents, window


def run_one(cfg, planned, out: Path, budget_left: float | None, max_workers: int | None):
    gdir = out / planned.game_id
    gdir.mkdir(parents=True, exist_ok=True)
    if (gdir / "result.json").exists():
        return "skipped (already finished)", json.loads((gdir / "result.json").read_text())
    checkpoint = gdir / "checkpoint.json"
    logger = CallLogger(path=gdir / "tokens.jsonl", resume=True)
    if budget_left is not None:
        logger.max_usd = logger.total_usd + budget_left
    gcfg = game_config(cfg, planned)
    agents, window = build_agents(cfg, planned, gcfg, logger)
    (gdir / "seats.json").write_text(json.dumps(
        {"seed": planned.seed, "memory_window": window,
         "seats": {p: asdict(s) | {"identity": agents[p].identity} for p, s in planned.seats.items()}}, indent=2))
    if checkpoint.exists():
        runner = GameRunner.resume(checkpoint, agents, max_workers=max_workers)
        status = f"resumed from {runner.game.get_current_phase()}, "
    else:
        if max_workers:
            gcfg.max_workers = max_workers
        runner = GameRunner(gcfg, agents, checkpoint)
        status = ""
    result = runner.run()
    runner.save(gdir, result)
    checkpoint.unlink(missing_ok=True)
    return status + "finished", result.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="default runs/<name>")
    ap.add_argument("--max-workers", type=int, default=None, help="concurrent model calls per game")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="run only these game indices")
    ap.add_argument("--dry-plan", action="store_true", help="print the seat plan and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = plan_games(cfg)
    out = Path(args.out or f"runs/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.json")

    print(f"{cfg['name']}: {len(plan)} game(s) -> {out}")
    for g in plan:
        seats = " ".join(f"{p[:3]}={s.model}" + (f"/{s.scoring}" if s.scoring != cfg.get("scoring", "dss") else "")
                         for p, s in g.seats.items())
        print(f"  {g.game_id} seed {g.seed}: {seats}")
    if args.dry_plan:
        return

    max_usd = cfg.get("max_usd")
    for g in plan:
        if args.only is not None and g.index not in args.only:
            continue
        budget_left = None if max_usd is None else max_usd - spent_so_far(out)
        if budget_left is not None and budget_left <= 0:
            print(f"budget of ${max_usd} used (${spent_so_far(out):.4f}); stopping. Raise max_usd and rerun to continue.")
            break
        start = time.monotonic()
        try:
            status, r = run_one(cfg, g, out, budget_left, args.max_workers)
        except (BudgetExceeded, LLMUnavailable) as e:
            print(f"{g.game_id}: STOPPED ({e}). Rerun the same command to resume.")
            break
        dss = score_all(r)["dss"]
        centers = " ".join(f"{p[:3]}:{r['final_centers'][p]}" for p in POWERS)
        dss_line = " ".join(f"{dss[p]:.2f}" for p in POWERS)
        print(f"{g.game_id}: {status} in {(time.monotonic() - start) / 60:.1f} min | {r['outcome']} at "
              f"{r['final_phase']} | centers {centers} | dss {dss_line}")
    print(f"total spent: ${spent_so_far(out):.4f}")


if __name__ == "__main__":
    main()
