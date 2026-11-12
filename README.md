# dipbench

An LLM benchmark harness for Diplomacy. Each agent is told which scoring system it is
playing to (DSS, SoS or linear), and the harness measures how play changes. It supports
full-press games (negotiation between agents) and gunboat games (no communication).

Built on the **unmodified** [`diplomacy`](https://github.com/diplomacy/diplomacy) engine
v1.1.2 (AGPL-3.0), which it imports and wraps. This project is licensed AGPL-3.0-or-later
to match.

## Setup

Requires Python 3.10–3.12 (developed on 3.11; the engine is untested on 3.13).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Layout

| Path | Purpose |
|---|---|
| `dipbench/config.py` | `GameConfig`: `press_mode` (`full` / `public` / `gunboat`), negotiation rounds, year cap, draw votes, ballots, identity disclosure, order retries |
| `dipbench/loop.py` | Game loop: negotiation (full only), orders + required draw vote with bounded retry, adjudication, year cap |
| `dipbench/state.py` | Read-only per-power views given to agents. Gunboat views never contain messages |
| `dipbench/agents/` | Agent interface plus a zero-cost `RandomAgent` |

## LLM agents

- **Credentials:** API keys are read from the environment only. Anthropic uses
  `ANTHROPIC_API_KEY` (or an `ant auth login` profile). OpenAI-compatible endpoints read
  the variable named by `api_key_env`; local servers usually accept any value.
- **Providers** (`dipbench/llm/`):
  - `anthropic`: Claude models, with structured JSON output, adaptive thinking,
    configurable `effort`, and a cached system prompt.
  - `openai_compat`: any chat-completions endpoint, for local models via Ollama, vLLM,
    llama.cpp or LM Studio, or hosted open-weight models. Set `base_url`, and set
    `json_mode` to `json_schema`, `json_object` or `none` depending on what the server
    supports.
  - Build either with `make_provider({...})`.
- **Agent** (`dipbench/agents/llm_agent.py`, prompts in `prompts.py`): the system prompt
  is fixed for the whole game, so it is cacheable. It states the power, the scoring rule
  with a worked example, how the game ends, the press-mode rules (gunboat says plainly
  that no communication is possible), the press budget, draw-vote rules and ballot
  visibility, disclosed identities, the output cap, the order syntax, and **how the
  agent's own memory works**. A reply that isn't valid JSON gets one corrective turn.
- **Memory:**
  - **One conversation per power.** Each power plays the whole game as a single
    append-only conversation. Every turn sends only what is new since that power's last
    turn: completed phases' orders and results, new messages it can see (with round
    numbers), the current board, the latest votes, its legal orders and the task. Its
    earlier turns and replies stay in the conversation.
  - **Replies only.** No model's earlier private reasoning is carried over, which keeps
    models comparable, and the system prompt says so.
  - **Diary.** Every order submission writes a diary entry of at most `diary_max_chars`
    characters. Diary entries are never trimmed.
  - **Memory window.** If the conversation would exceed `memory_window` (model-neutral
    tokens, default 200,000), the oldest turns are replaced by a `[MEMORY]` note holding
    the diary entries from those turns. Each trim is logged in the agent's `trims` and in
    the checkpoint.
  - **Fairness.** Give every seat in a game the same window, sized for the
    smallest-context model at the table (e.g. about 34,000 when Qwen3-8B, with its
    40,960-token context, is playing).
  - **Caching.** With Anthropic models the whole conversation is prompt-cached, so
    re-reading earlier turns is billed at the cache-read rate.
- **`claude_cli` provider** (`dipbench/llm/claude_cli.py`): runs agents through Claude Code's
  headless mode (`claude -p`), so models can be used through a Claude subscription instead of
  API credits.
  - **Model spec:** `{"provider": "claude_cli", "model": "claude-sonnet-5-5", "effort": "high",
    "billing": "subscription"}`. Optional: `cli` (path to the CLI) and `extra_args`.
  - **What each call runs:** `--system-prompt` replaces Claude Code's own system prompt,
    `--tools ""` disables all tools, `--safe-mode` / `--strict-mcp-config` ignore your
    CLAUDE.md, hooks, plugins and MCP servers, and `--json-schema` enforces the reply format.
    The output cap goes through `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.
  - **Memory:** a headless call has no multi-turn history, so earlier turns are sent as an
    append-only transcript in the prompt. That still caches, but it's a different message
    layout from the API providers, so treat it as its own condition when comparing models.
  - **Billing:** `"subscription"` counts calls as $0 toward `max_usd` and logs the CLI's
    reported cost as `reported_cost_usd`; `"api"` counts it. Subscription usage limits apply:
    if you hit them, the game stops and resumes like any API failure.
- **Call log:** every call is written to `tokens.jsonl` with model, phase, power,
  purpose, input / cache-read / cache-write / output / thinking tokens, latency, cost,
  stop reason and error. `CallLogger(max_usd=...)` stops the game with `BudgetExceeded`
  once a spending cap is reached.
- **Prices** (`dipbench/llm/pricing.py`) are Anthropic list prices. Local models cost $0
  unless you call `register_price`.

### Experiments (config files)

```bash
.venv/bin/python scripts/run.py configs/haiku_vs_qwen.json --dry-plan   # show the seat plan
ANTHROPIC_API_KEY=... .venv/bin/python scripts/run.py configs/haiku_vs_qwen.json
```

A config (see `dipbench/experiment.py` and `configs/`) names the models, the table, and the
game settings:

- **`models`:** named model specs: provider, model, `effort`, `identity` label, and for
  local models `base_url`, `json_mode`, `enable_thinking` and `context_window`.
- **The table**, in one of two forms:
  - **`lineup` + `n_games` + `assignment`:** the seven models at the table. Entries can
    carry their own `scoring`, which follows the model to whichever seat it gets.
    `assignment` is one of:
    - `balanced` (the default): shuffle, then rotate one seat per game, so every slot plays
      every power once per block of 7 games;
    - `random`: an independent shuffle per game;
    - `fixed`: the lineup in power order.
  - **`games`:** explicit tables, one list of 7 seats per game.
- **`scoring`** (the default rule), **`game`** (any `GameConfig` field) and **`agent`**
  (`output_limit`, and `memory_window`, where `"auto"` uses the smallest context at the
  table), plus `seed` and `max_usd`.

Each game writes to `runs/<name>/game_NNN/`: the files listed under "Outputs per game"
below, plus `seats.json` (who sat where) and `agents.json` (each agent's conversation and
diary). Rerunning the same command skips finished games and resumes interrupted ones, and
`max_usd` caps spending across all runs of the config. Check progress with
`scripts/game_status.py runs/<name>/game_NNN`.

### Pilot

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/python scripts/pilot.py --max-usd 20
```

This plays 2 game-years in each press mode (full, public, gunboat) with all seven
powers on Claude Haiku 5.5. It writes `runs/pilot/pilot_report.md` with token counts,
cost per game-year, and estimated per-game cost for several model mixes; the estimates
reuse Haiku's token counts at other models' prices. `--sonnet-check` adds a game with
France on Sonnet 5.5, to measure how much more Sonnet writes than Haiku (off by default).

### Resuming interrupted games

Every game saves `checkpoint.json` in its output directory at the start of each phase.
A game interrupted by the spending cap, an API failure (credits exhausted, rate limits,
outages, network) or a crash can be continued from the start of the phase it was in, so
only that phase's calls are repeated. The call log is reloaded on resume, so earlier
spending still counts toward the cap.

- **Pilot:** `scripts/pilot.py --resume` skips games that finished and continues the
  rest.
- **In code:** `run_game(config, agents, out_dir=..., resume=True)`, or
  `GameRunner.resume(checkpoint_path, agents)`. The agents must be the same models as
  before (checked against the saved identities). Their state, such as LLM notes and
  RNG state, is restored from the checkpoint.
- **API failures stop the game.** They raise `LLMUnavailable` instead of silently
  playing on with missing orders. Model-side failures (invalid JSON, refusals,
  hitting the output cap, illegal orders) are part of the game and handled by the
  retries described below.
- Checkpoints are deleted once a game finishes and its outputs are written.
- A resumed game with seeded random agents finishes exactly like an uninterrupted
  one (`tests/test_resume.py`).

## Playing as a human

A seat whose model spec has `"provider": "human"` is played by a person through a local web page
(`dipbench/agents/human_agent.py`). Example: `configs/human_vs_6opus.json` (you are France).

    .venv/bin/python scripts/run.py configs/human_vs_6opus.json
    # then open http://127.0.0.1:8765/

- There is no timeout. On each of your turns (every negotiation round and every order
  submission) the whole game waits until you submit. Closing the page doesn't lose anything: reopen it.
- The page shows the map (drawn by the engine), the board, last phase's results, your messages
  (filterable by correspondent), draw shares and ballots, the full order history and the same rules
  the AI players get (minus the parts about model output and memory).
- Messages: pick recipients (one power, several, or everyone), queue as many messages as fit your
  press budget, then send them all to end the round, or send nothing. The same budget, recipient
  rules and simultaneous delivery apply as for AI players.
- Orders are chosen from the legal orders for each unit, with the draw vote in movement and Winter phases.
- Several human seats get consecutive ports (8765, 8766, ...). The page listens on 127.0.0.1 only.
- Ctrl-C stops the game; rerunning the command resumes from the start of the current phase.
- The AI players are told what `identity_disclosure` allows: with "hidden" they don't know a human
  is playing; with "revealed" the human seat is labelled with its `identity` ("human").

## Game rules as implemented

- **Full press:** every phase (movement, retreats and Winter adjustments) runs
  `negotiation_rounds` rounds before orders, and every survivor takes part. Each round,
  every survivor is called at the same time and returns any number of messages (zero
  is fine). Each message goes to one power, to any group of powers, or to `GLOBAL`
  (everyone), and every recipient sees the full recipient list. Messages in a round are
  simultaneous: they are delivered only after every power has spoken, so a reply takes
  one round. The last round's messages are read when agents write their orders. The
  engine skips retreat and Winter phases in which no power has anything to order, so
  those phases have no press.
- **No opting out:** every survivor is called in every round of every phase. There's
  no way to declare yourself done talking.
- **Press token budget:** each power may send `press_token_budget` tokens of message
  text (default 600, about 400 words) in each negotiation round. Unused budget doesn't
  carry over. `press_budget_scope` can be changed to `phase` (shared across that phase's
  rounds) or `game`. Tokens are counted model-neutrally (`dipbench/tokens.py`: 1 token per 4
  characters), so every model gets the same amount of text whatever its tokenizer. A
  group or `GLOBAL` message costs its length once. Agents are told the limit and what
  they have left every round (`ctx.press_budget`). A message that doesn't fit is
  rejected and logged; later messages that do fit still go out. Tokens sent per power
  are in `result.json` (`press_tokens_sent`). Set `press_token_budget: null` for no
  limit.
- **LLM output limit per call** (`dipbench/llm/limits.py`, thinking included): with the
  `tokens` basis (the default), every model gets the same cap (`max_output_tokens`,
  default 8192). With the `cost` basis, every model gets the same dollar budget per call
  (`usd_per_call`), so a cheaper model gets proportionally more tokens, between
  `min_tokens` and `max_tokens`. Per-power `overrides` take precedence over both. Agents
  are told their cap. This is separate from the press budget, which limits only the
  message text agents send each other.
- **Other caps:** `max_messages_per_round` and `max_message_chars` are off by default.
  When set, anything over a cap is rejected (or truncated) and logged with a reason in
  `rejected_press.jsonl`, never dropped silently. Recipients that aren't surviving other
  powers are also logged there; the rest of that message is still delivered.
- **Public press:** like full press, but only `GLOBAL` messages are delivered, so every
  power reads every message. Private messages are dropped and counted
  (and logged in `rejected_press.jsonl`). The engine's `PUBLIC_PRESS` rule is set for the record, and
  the harness enforces the restriction itself.
- **Gunboat:** the engine `NO_PRESS` rule is set, the negotiation step is skipped
  entirely, and the harness raises `PressViolation` if any message is created. (The
  engine's local `Game` doesn't enforce `NO_PRESS` itself.)
- **Draw votes, in movement and Winter phases, both modes:** every survivor submits an
  `OrderSubmission(orders, draw_vote)` in every movement and Winter phase. That includes
  survivors with nothing to order, such as a power with centers but no units, or one with
  no builds or disbands; they get an order call with no legal orders. `draw_vote`
  (true/false) is required. A missing vote is treated like an invalid order: the agent
  retries, and if it runs out of retries the vote counts as no. Retreat phases take no
  draw votes (only powers with retreats are called, and any vote they send is ignored).
  The survivors' ballots go on the engine's own `power.vote` fields, and
  `Game.has_draw_vote()` checks for unanimity. If every vote is yes, the game ends
  immediately via `Game.draw(winners=survivors)`, without adjudicating that phase.
  The engine skips a Winter phase when no power has builds or disbands, so those years
  get no Winter vote.
  Ballots are **public by default**: every agent sees all survivors' ballots from the most recent vote.
  With `secret_ballots: true`, an agent sees only its own vote. Either way, all ballots
  are logged for analysis. Votes count from `draw_vote_from_year` on, and
  `draw_votes: false` disables voting.
- **Identity disclosure** (`identity_disclosure`): `hidden` (the default) means agents
  see only power names. `revealed` means each agent sees the true model label of every
  other power (`Agent.identity`). `false_labels` shuffles the true labels so that as
  many as possible are wrong; the shuffle is reproducible from `seed`. Agents never see
  a label for their own power. True and shown labels are both saved in `result.json`.
- **Stalemates:** not detected yet, so a single holdout can stall a locked game until the
  year cap. See [TODO.md](TODO.md).
- **Solo:** 18+ centers, detected by the engine.
- **Year cap:** after Winter of `max_year`, the game ends as a draw among the survivors,
  recorded as `outcome: year_cap`, separately from `draw_vote`.
- **Invalid orders:** the agent gets the errors and up to `order_retries` more attempts.
  After that, only the legal subset is submitted, and the engine holds, disbands or
  civil-disorders whatever is left unordered.

## Scoring (`dipbench/scoring.py`)

Every game is scored under every system from its final state (`score_all(result)`;
`result.json` works too). `draw_vote`, `year_cap` and `engine_draw` are all scored as
draws among the survivors. `year_cap` is only flagged separately in analysis.

| System | Solo | Draw |
|---|---|---|
| `dss` | 1 to the winner, 0 to the others | 1/n to each of the n participants, 0 to eliminated powers |
| `sos` | 1 to the winner, 0 to the others | SC_i² / Σ SC_j² over the participants |
| `linear` | 1 to the winner, 0 to the others | SC_i / 34 |
| `linear_norm` | 1 to the winner, 0 to the others | SC_i / Σ survivors' SCs |

A solo is full credit in every system. A survivor with units but no centers is a draw
participant: it gets 1/n under DSS and 0 under SoS.

## Outputs per game

`game.json` (engine saved-game format), `result.json`, `votes.jsonl`, `orders.jsonl` (every
attempt, including rejected ones), and, in full and public press, `press.jsonl` (one row
per message, with its recipient list and round) and `rejected_press.jsonl`. The engine's
`Message` holds a single recipient, so in `game.json` a group message appears as one copy
per recipient.
