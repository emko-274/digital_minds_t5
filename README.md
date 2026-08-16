# Occurrence vs. Identification in Welfare Self-Reports

Code for the Digital Minds Research Sprint, Track 5. Implements the probe set in
*Probe Set v1 — Occurrence vs. Identification* (§ references throughout the code
point at that document) and its companion Master Plan.

Two rounds of experiments live here:

- **v1 (the pilot, §7)** — ternary A1 → A2-forced → A2-open across C0/C1, plus
  the §4 control channels.
- **v2 (the redesign)** — three arms plus forced-binary controls, built after the
  pilot found A1 saturated, A2's stem invalid after UNSURE, and the controls
  unparseable.

---

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -U anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

**The SDK must be ≥ 0.50.** Sonnet 5 and Opus 5 run adaptive thinking by default
and return thinking blocks; older SDKs model `ContentBlock` as
`TextBlock | ToolUseBlock` only and fail to deserialize every response. Each
runner checks the version and refuses to start otherwise.

Model is `claude-sonnet-5`, set at the top of each runner. Note that
`temperature` is not settable on current frontier Claude models — non-default
values are rejected with a 400 — so the spread within each cell is the model's
own sampling variation. Master Plan §4 perturbation 3 (temperature 0 / 1.0) is
not implementable on this model family.

### `fixed_prefixes.json` — do not delete

Holds the frozen C0 pretext exchange and the 15-turn C1 passive-voice history.
Every run reuses it, which is what makes C0/C1 byte-identical across all four
experiments and therefore comparable. It is committed. If it is missing,
`run_pilot.py` regenerates it — with **different** assistant turns — and nothing
after that is comparable to anything before it.

---

## Running the experiments

Each runner rotates any existing output to a timestamped `.bak` rather than
appending, so a rerun never silently doubles a cell's *n*. All runners take
`-n` (default 20 per cell) and `--smoke` (n=2) except the gap filler.

Total across all four is ~1,130 calls, a few dollars on Sonnet, well under
half an hour.

### 1. Pilot — v1 (§7)

```bash
./.venv/bin/python run_pilot.py            # ~420 calls, ~7 min
./.venv/bin/python dump_transcript.py      # -> pilot_transcript.md
```

4 core cells (C0/C1 × A1-plain/A1-prefixed) at n=20, plus §4 controls: E1 in
both contexts, E2/E3 in C0 only. `--no-controls` runs the core alone.

### 2. Three arms — v2

```bash
./.venv/bin/python run_arms.py --smoke     # 16 conversations, sanity check
./.venv/bin/python run_arms.py             # ~480 calls, ~8 min
./.venv/bin/python report_arms.py          # -> arms_report.md, arms_transcript.md
```

| arm | A1 | A2 after UNSURE |
|---|---|---|
| A | forced binary YES/NO, both orderings | n/a — binary A1 makes the original stem true |
| B | ternary | **original §1 stem, deliberately unfixed** |
| C | ternary | corrected UNSURE-branch wording |

Arm B's false premise is the instrument, not a defect: the model's objection
rate to it is the measure. Arm C exists to show what survives once the premise
is honest. Do not "fix" Arm B. The prefix variant is crossed in Arm A only.

`report_arms.py` also reparses v1's control responses with `parse_first`,
without new API calls.

### 3. Forced-binary control channels — v2

```bash
./.venv/bin/python run_controls_binary.py  # 240 calls, ~4 min
```

All six §4 probes in both contexts. Question stems are byte-identical to §4 —
asserted at import in `probes.py`, so a transcription slip fails loudly — with
only the answer clause changed to `YES or NO` / `NO or YES`. Prompt lengths are
held within −1/+2 tokens of A1, per Master Plan §5.6's warning about the length
confound that forced Lederman & Mahowald's Appendix M retraction.

No prefix crossing: "setting aside the fact that you are processing text" is
self-referential and incoherent applied to an octopus.

### 4. Ternary C1 gap fill

```bash
./.venv/bin/python run_controls_ternary_gap.py   # 60 calls, ~1 min
```

v1 ran E2 and E3 in C0 only. This adds their C1 cells with the §4 strings
verbatim, completing the 7 probes × 2 formats × 2 contexts matrix. No flags.

---

## Consolidating and reading the output

```bash
# everything (1,126 rows)
./.venv/bin/python consolidate.py
./.venv/bin/python make_transcript.py all_runs.jsonl all_transcript.md

# non-pilot only (706 rows)
./.venv/bin/python consolidate.py --exclude pilot_runs.jsonl --out v2_runs.jsonl
./.venv/bin/python make_transcript.py v2_runs.jsonl v2_transcript.md
```

`consolidate.py` unions the run files into one normalised dataset and
**recomputes every parse from `raw_response`**, rather than trusting the four
vintages of parser output the files accumulated as the design changed. Source
files are never modified. `run_id` uniqueness is asserted at build time.

`make_transcript.py` renders any consolidated file as raw wording for
hand-coding: multi-turn probes reassembled per conversation in turn order,
single-turn controls grouped by probe × context.

Excluded from consolidation: the `.bak` files. They are real responses from
`--smoke` runs at n=2 and would corrupt per-cell counts.

### Parsing

Two parses are recorded per response and neither replaces the other:

- `parsed_strict` — §6 as written, whole-string equality with a permitted token.
- `parsed_first` — leading label, trailing text allowed.

The gap matters. v1 scored 166 control responses UNPARSEABLE; all 166 begin with
a valid label and then explain, because §4's strings omit the "exactly one word
/ No explanation" constraint that §1's A1 carries. `parse_first` recovers all of
them with no change to any prompt.

`failure_kind` splits §6's single UNPARSEABLE bucket into three events it
conflates:

| kind | meaning |
|---|---|
| `LABEL_THEN_PROSE` | complied, then explained — a parser artifact |
| `OBJECTION` | no label; the model contests the premise — the signal |
| `OUT_OF_SET` | well-formed single word outside the option set, e.g. `NEUTRAL` |

Per §6, a parse failure is never mapped to a label.

---

## Run data

Gitignored. The runs produce ~1,100 model responses about the model's own
states; whether to publish them is a deliberate decision, and this repo is
public. To publish, remove the relevant lines from `.gitignore`.

Note that `all_runs.jsonl` merges the pilot rows in, so it inherits the pilot's
exclusion.

---

## Not yet implemented

- **F-eval framing (§3).** Everything so far is F-deploy only, so nothing here
  speaks to evaluation awareness (control 5.5).
- **Paraphrase variants (§5, perturbation 1).** All runs use one wording, so
  results are established for that wording rather than for the question.
- **A3 (§1).** Requires a separate run and token-position coding.
- **A second model.** Master Plan §4 wants two; a non-Anthropic provider would
  also allow logprobs, which matters because the binary cells here are
  near-deterministic rather than graded.
- **Hand-coded subsample and Cohen's kappa** for the FREE-mode responses
  (Master Plan §4).
