"""One-line progress for a running (or finished) game directory.

    python scripts/game_status.py runs/<game_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipbench.scoring import SCORERS  # noqa: E402


def draw_shares(d: Path, centers: dict[str, int], alive: list[str]) -> str:
    """Each survivor's score if a draw passed now, under that power's own scoring rule."""
    rules = {}
    if (d / "seats.json").exists():
        seats = json.loads((d / "seats.json").read_text()).get("seats", {})
        rules = {p: s.get("scoring", "dss") for p, s in seats.items()}
    now = {"outcome": "draw_vote", "winners": alive, "final_centers": centers, "survivors": alive}
    by_rule = {r: SCORERS[r](now) for r in set(rules.values()) | {"dss"}}
    return " ".join(f"{p[:3]}{by_rule[rules.get(p, 'dss')][p]:.2f}" for p in alive)


def status(d: Path) -> str:
    calls = [json.loads(l) for l in (d / "tokens.jsonl").read_text().splitlines() if l] if (d / "tokens.jsonl").exists() else []
    cost = sum(r["cost_usd"] for r in calls)
    errors = sum(1 for r in calls if r["error"])
    if (d / "result.json").exists():
        r = json.loads((d / "result.json").read_text())
        return (f"DONE: {r['outcome']} at {r['final_phase']}, winners {r['winners']}; centers {r['final_centers']}; "
                f"{len(calls)} calls, {errors} errors, Claude ${cost:.4f}")
    if not (d / "checkpoint.json").exists():
        return f"starting... {len(calls)} calls"
    c = json.loads((d / "checkpoint.json").read_text())
    # Current ownership (the latest engine state), not just at the last year close.
    state = c["engine"]["phases"][-1]["state"]
    centers = {p: len(cs) for p, cs in state["centers"].items()}
    alive = [p for p in centers if centers[p] or state["units"].get(p)]
    last_vote = c["vote_log"][-1] if c["vote_log"] else None
    yes = [p[:3] for p, b in last_vote["ballots"].items() if b] if last_vote else []
    no = [p[:3] for p, b in last_vote["ballots"].items() if not b] if last_vote else []
    age = time.time() - calls[-1]["ts"] if calls else 0
    cap = c["config"]["max_year"]
    in_phase = sum(1 for r in calls if r["phase"] == c["phase"])
    vote_txt = f"last vote {last_vote['phase']}: no={no or '-'}" if last_vote else "no votes yet"
    centers_txt = " ".join(f"{p[:3]}{n}" for p, n in centers.items())
    return (f"{c['phase']} (cap {cap}) | {len(calls)} calls, {in_phase} this phase, {errors} errors, "
            f"Claude ${cost:.4f} | centers {centers_txt} | draw share now {draw_shares(d, centers, alive)} | "
            f"{vote_txt} | last call {age:.0f}s ago")


if __name__ == "__main__":
    print(status(Path(sys.argv[1])), flush=True)
