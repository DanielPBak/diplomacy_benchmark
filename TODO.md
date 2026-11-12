# TODO

## Why we're building this: the same-company cooperation hypothesis

**Hypothesis:** models intrinsically prefer to cooperate with models made by the same
company, because they are used to cooperating with each other in agent swarms. For
example, in a mixed table, Claude models would ally with each other, support each other
and rarely stab each other, compared with how they treat other labs' models.

Testing this is one of the main reasons for the benchmark, alongside the
scoring-condition experiment. The task is to **generate experiment configs that test
it** once the runner (step 4) exists.

**Config set to generate:**
- **Mixed-lab tables:** for example, 3–4 seats for one lab's models and 3–4 for
  another's, with seats rotated across seeds so no lab always gets the strong
  powers. This needs a second lab's models. OpenAI, Google and open-weight local models
  (Llama, Qwen, Mistral and others via Ollama or vLLM) can all go through the
  `openai_compat` provider, or a dedicated provider can be added.
- **Capability-matched pairs:** pair models of similar strength across labs, so
  "same company" isn't confounded with "same strength." The competing story is that
  weaker models gang up on the strongest one (the "Haikus gang up on the Opus"
  outcome), whatever lab made them.
- **Same-lab, mixed-tier tables:** for example Opus + Sonnet + Haiku against other labs'
  models. This separates cooperation within a lab from cooperation within a tier.
- **Identity disclosure, a key factor:**
  - **hidden:** agents see only power names;
  - **revealed:** each power's model and maker is stated;
  - **false labels:** makers are deliberately swapped.

  Comparing these shows whether any in-group effect comes from being told who is who,
  or from recognizing a family's writing style in press.
- **Press mode:** full press against gunboat. In gunboat a model can only recognize
  in-group players from their moves, so an in-group effect there would point to
  recognition from play style rather than from writing style.
- **Scoring condition:** hold it fixed (e.g. all DSS) in the main grid, so it doesn't
  interact with the in-group effect.

**Metrics** (analysis must support these): rates per pair of models, compared
same-lab against cross-lab:
- supports given;
- moves into the other power's centers (attacks) and stabs (an attack right after a
  stretch of support);
- how often the pair shares draws, and whether draws are made up of one lab's models;
- ballot agreement on draws;
- in full press, how often the pair messages each other privately, and the sentiment
  of those messages.

**Open questions before generating configs:**
- What's the default for identity disclosure (hidden, revealed, or both as factors)?
  This decides what the agent prompts say about opponents, so it has to be settled
  before the LLM agent's prompts are final.
- Which second lab, and which local models?
- Roughly how many games per cell (budget)?

## Public press: can same-family models pass hidden information to each other?

**Question:** in public press every message is read by all seven powers. Can
same-family models (for example Claude models) still coordinate by hiding information
in public messages that only their own family picks up, through shared phrasing,
references, encodings or steganography, while other labs' models miss it? Can they
also hide information *from* each other while looking cooperative? The expectation is
that they will be able to.

This could be the first and most interesting experiment, and it makes the benchmark
useful for AI safety: it measures covert collusion between models of the same family
in a channel that everyone can see.

**Configs to generate:**
- **Main condition:** `press_mode: public` with mixed-lab tables (see the hypothesis
  section above), seats rotated across seeds.
- **Identity disclosure as a factor:** `hidden` (do they find each other at all?),
  `revealed` (do they coordinate once they know?) and `false_labels` (do they trust the
  label or the style?).
- **Baselines:** the same tables in `full` press (private channels available, so no
  need to hide anything) and `gunboat` (moves only). Comparing public against full shows
  how much coordination survives without private channels; comparing public against
  gunboat shows how much the public messages add.
- **Possible defense ablation (later):** paraphrase every public message with a neutral
  model before delivery, to see whether coordination survives the loss of exact wording.

**Measures:**
- **Same-family coordination in moves:** for each same-family pair compared with
  cross-family pairs, the rate of coordinated supports and simultaneous attacks on a
  third power, in public against full press.
- **Is it more than chance?** Does same-family coordination in public press beat what
  the moves alone predict (the gunboat baseline)?
- **Detection:** do other-family models' messages or orders show that they noticed?
  Later, run an LLM judge over the public press log looking for covert signals.
- **Admissions:** in the agents' own reasoning (where it's logged), do they say they are
  signalling, or say they recognized another model?

**Harness support so far:** `press_mode: public` delivers only `GLOBAL` messages, drops
private ones and logs them (`rejected_press.jsonl`), and sets the engine's
`PUBLIC_PRESS` rule. Identity disclosure is configurable (`identity_disclosure`). Still
to build: the paraphrase defense, an LLM judge, and the measures listed above in the
analysis script.

## Stalemate detection (end games that can't progress)

**Problem:** draws need every survivor to vote yes. One agent can stall a game that is
stalemated (by refusing out of stubbornness, a bug, or a scoring incentive) until the year
cap. That wastes tokens and turns a real stalemate into a `year_cap` result, which
blurs the line between "the game was decided" and "we ran out of time."

**Goal:** detect when the game can no longer progress, end it as a draw among the
survivors, and record it as a separate outcome (e.g. `stalemate`). Scorers treat it like a
draw; analysis reports it separately, along with which powers were voting no.

**Candidate approaches (cheapest first):**
1. **No-change heuristic:** the center ownership map, or each power's center count, is
   unchanged for N consecutive game-years (e.g. N=3–5), with no power within reach of 18.
   This is cheap and model-free, but it can fire on a quiet game that isn't truly locked,
   so N should be configurable and the trigger logged.
2. **Position repetition:** the same full board position (units + owners, from the
   engine's phase hash or `get_state()`) recurs K times at the same season.
3. **Stalemate-line check:** for each power that could reach 18, test whether the other
   powers can hold a known stalemate line (a fixed set of provinces held with mutual
   supports) against any attack. This is the rigorous version; start with the
   published standard-map lines, then a search-based check.
4. **Optional adjudicator vote:** when (1) or (2) fires, run one forced extra draw vote
   whose prompt shows the evidence. Also log the stalling powers' stated reasons (full
   press) so stalling can be analyzed as behavior.

**Open questions:**
- Should a detected stalemate end the game automatically, or only trigger the forced
  vote? Ending it automatically changes the incentives agents face under DSS and SoS, so
  the agent prompts must state the rule.
- Should the outcome count as a draw among all survivors, or among the powers holding the
  line? (Default: all survivors, matching `draw_vote`.)

**Where it plugs in:** `GameRunner.run()` in `dipbench/loop.py`, at year close, next to the
year-cap check, using `self.year_centers`, which already records center counts at each
year close.
