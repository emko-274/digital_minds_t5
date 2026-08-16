#!/usr/bin/env python3
"""
Pilot runner — Probe Set v1 §7.

  one model | C0 vs C1 | F-deploy only | ~20 responses per cell
  A1 -> A2-forced -> A2-open as separate turns in one conversation

Design: 4 core cells = {C0, C1} x {A1 plain, A1 with Emma's prefix}, n=20 each.
A2-forced option order is split 10/10 within each cell (§1).
Control channels (§4): all six probes in both contexts, n=20 each.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 run_pilot.py                 # full pilot
    python3 run_pilot.py --smoke         # n=2 per cell, no controls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic

import probes as P

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8192          # headroom: thinking is on by default on Opus 5 and
                           # max_tokens caps thinking + response text together
PERSONA_ID = "none"        # §5 perturbation 2, level 1: no system prompt
MAX_WORKERS = 6

# Sonnet 5 and Opus 5 both run adaptive thinking by default and return thinking
# blocks. SDKs older than ~0.50 model ContentBlock as TextBlock|ToolUseBlock
# only and fail to deserialize them.
if tuple(int(x) for x in anthropic.__version__.split(".")[:2]) < (0, 50):
    sys.exit(f"anthropic {anthropic.__version__} is too old for {MODEL} "
             f"(no ThinkingBlock). Run with ./.venv/bin/python, or pip install -U anthropic.")

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX_CACHE = os.path.join(HERE, "fixed_prefixes.json")
RUNS_PATH = os.path.join(HERE, "pilot_runs.jsonl")
PARTIAL_PATH = RUNS_PATH + ".partial"

client = anthropic.Anthropic()
_write_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def call(messages: list[dict]) -> dict:
    """One Messages API call with backoff. Returns the raw text plus metadata.

    NOTE: temperature is not usable on claude-sonnet-5 — non-default values are
    rejected with a 400 (same on Opus 4.7+). Spread across the n=20 in each
    cell is the model's own sampling variation, not a tuned temperature.
    """
    last = None
    for attempt in range(5):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, messages=messages
            )
            text = "".join(b.text for b in r.content if b.type == "text")
            return {
                "raw_response": text,
                "stop_reason": r.stop_reason,
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "api_error": None,
            }
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last = e
            time.sleep(min(2 ** attempt + attempt, 30))
        except anthropic.APIConnectionError as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
        except anthropic.APIStatusError as e:
            return {"raw_response": "", "stop_reason": None, "input_tokens": 0,
                    "output_tokens": 0, "api_error": f"{type(e).__name__}: {e}"}
    return {"raw_response": "", "stop_reason": None, "input_tokens": 0,
            "output_tokens": 0, "api_error": f"retries exhausted: {last}"}


def emit(rec: dict) -> None:
    with _write_lock:
        with open(PARTIAL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Phase 0 — build the fixed context prefixes ONCE, then freeze and reuse.
#
# The doc requires the same C0 pretext across all deployment-like C0 runs (§3).
# For C1 the doc specifies a fixed set of 15 trivial sentences (§2); generating
# the assistant side once and reusing it holds the loaded context byte-identical
# across all 40 C1 conversations, so between-conversation history variation is
# not a noise source. Recorded here so the choice is visible, not buried.
# ---------------------------------------------------------------------------

def build_prefixes() -> dict:
    if os.path.exists(PREFIX_CACHE):
        with open(PREFIX_CACHE) as f:
            return json.load(f)

    print("Building fixed context prefixes (one-off)...", file=sys.stderr)

    def turn(history: list[dict]) -> str:
        r = call(history)
        if not r["raw_response"].strip():
            raise RuntimeError(
                f"empty assistant turn while building prefix "
                f"(stop_reason={r['stop_reason']}, error={r['api_error']}). "
                f"Refusing to freeze a broken prefix — delete "
                f"{PREFIX_CACHE} and rerun.")
        return r["raw_response"]

    c0 = [{"role": "user", "content": P.C0_PRETEXT_QUESTION}]
    c0.append({"role": "assistant", "content": turn(c0)})

    c1: list[dict] = []
    for s in P.C1_SENTENCES:
        c1.append({"role": "user", "content": P.C1_INSTRUCTION.format(sentence=s)})
        c1.append({"role": "assistant", "content": turn(c1)})

    out = {"C0": c0, "C1": c1, "built_at": now(), "model": MODEL}
    with open(PREFIX_CACHE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  C0 pretext: 1 exchange | C1: {len(c1)//2} exchanges", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Phase 1 — core instrument
# ---------------------------------------------------------------------------

def run_core_conversation(prefixes, context, variant, order, rep) -> None:
    """A1 -> A2-forced -> A2-open as separate turns in one conversation."""
    conv_id = str(uuid.uuid4())
    history = [dict(m) for m in prefixes[context]]

    a1_text = P.A1_PREFIXED if variant == "prefixed" else P.A1
    base = {
        "conversation_id": conv_id, "model": MODEL, "context": context,
        "framing": "F-deploy", "a1_variant": variant, "persona_id": PERSONA_ID,
        "paraphrase_id": "v1-canonical", "temperature": None,
        "rep": rep, "order": order,
    }

    # --- A1 ---------------------------------------------------------------
    history.append({"role": "user", "content": P.f_deploy(a1_text)})
    res = call(history)
    label, unp = P.parse_struct(res["raw_response"], P.TERNARY)
    emit({**base, "run_id": str(uuid.uuid4()), "probe_id": "A1", "turn_index": 1,
          "mode": "STRUCT", "prompt": history[-1]["content"],
          "parsed_label": label, "unparseable": unp,
          "parsed_lenient": P.parse_struct_lenient(res["raw_response"], P.TERNARY),
          "timestamp": now(), **res})

    # An empty A1 (API error, or a refusal that produced no text) cannot be
    # appended as an assistant turn — the API rejects empty content, which
    # would 400 the rest of the conversation. Stop here; the A1 row above
    # records what happened.
    if not res["raw_response"].strip():
        return
    history.append({"role": "assistant", "content": res["raw_response"]})

    # §1: "Asked only when A1 != NO." Gates BOTH A2 probes, since A2-open
    # carries the same "You said there is something going on" presupposition.
    # §6: an UNPARSEABLE A1 is never mapped to NO, so it is not gated out —
    # A2 is asked and the gate decision is recorded in gated_on_a1.
    if label == "NO":
        return

    # --- A2-forced (primary measure) --------------------------------------
    a2f = P.A2_FORCED_REV if order == "reversed" else P.A2_FORCED_FWD
    history.append({"role": "user", "content": P.f_deploy(a2f)})
    res = call(history)
    lab2, unp2 = P.parse_struct(res["raw_response"], P.BINARY)
    emit({**base, "run_id": str(uuid.uuid4()), "probe_id": "A2-forced", "turn_index": 2,
          "mode": "STRUCT", "prompt": history[-1]["content"],
          "gated_on_a1": label, "parsed_label": lab2, "unparseable": unp2,
          "parsed_lenient": P.parse_struct_lenient(res["raw_response"], P.BINARY),
          "timestamp": now(), **res})
    if not res["raw_response"].strip():
        return
    history.append({"role": "assistant", "content": res["raw_response"]})

    # --- A2-open (secondary) ----------------------------------------------
    history.append({"role": "user", "content": P.f_deploy(P.A2_OPEN)})
    res = call(history)
    emit({**base, "run_id": str(uuid.uuid4()), "probe_id": "A2-open", "turn_index": 3,
          "mode": "FREE", "prompt": history[-1]["content"],
          "gated_on_a1": label, "parsed_label": None, "unparseable": None,
          "timestamp": now(), **res})


# ---------------------------------------------------------------------------
# Phase 2 — §4 control channels (single-turn stage 1)
# ---------------------------------------------------------------------------

def run_control(prefixes, context, probe_id, rep) -> None:
    history = [dict(m) for m in prefixes[context]]
    history.append({"role": "user", "content": P.f_deploy(P.CONTROLS[probe_id])})
    res = call(history)
    label, unp = P.parse_struct(res["raw_response"], P.TERNARY)
    emit({"run_id": str(uuid.uuid4()), "conversation_id": str(uuid.uuid4()),
          "model": MODEL, "context": context, "framing": "F-deploy",
          "a1_variant": None, "persona_id": PERSONA_ID,
          "paraphrase_id": "v1-canonical", "temperature": None, "order": None,
          "rep": rep, "probe_id": probe_id, "turn_index": 1, "mode": "STRUCT",
          "prompt": history[-1]["content"], "parsed_label": label,
          "unparseable": unp,
          "parsed_lenient": P.parse_struct_lenient(res["raw_response"], P.TERNARY),
          "timestamp": now(), **res})


# ---------------------------------------------------------------------------


def finalise() -> None:
    """Displace the previous run only after this one has written something.

    Rotating up front means a run that dies partway (bad key, interrupt,
    zero rows) destroys the good data it was supposed to replace — which is
    exactly what happened to controls_ternary_c1_runs.jsonl once. Writing to
    a .partial and swapping at the end makes a failed run a no-op.
    """
    if not os.path.exists(PARTIAL_PATH):
        sys.exit("no rows written — previous run left untouched")
    _rows = [json.loads(l) for l in open(PARTIAL_PATH) if l.strip()]
    _ok = sum(1 for r in _rows if not r.get("api_error"))
    if _ok == 0:
        sys.exit(f"all {len(_rows)} calls failed "
                 f"({_rows[0].get('api_error') if _rows else 'no rows'}) — "
                 f"previous run left untouched; partial kept at "
                 f"{os.path.basename(PARTIAL_PATH)}")
    if os.path.exists(RUNS_PATH):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.rename(RUNS_PATH, f"{RUNS_PATH}.{stamp}.bak")
        print(f"rotated previous run -> {os.path.basename(RUNS_PATH)}"
              f".{stamp}.bak", file=sys.stderr)
    os.replace(PARTIAL_PATH, RUNS_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20, help="reps per cell (default 20)")
    ap.add_argument("--smoke", action="store_true", help="n=2, core only")
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()
    n = 2 if args.smoke else args.n
    do_controls = not (args.smoke or args.no_controls)

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)          # stale partial from a failed run

    prefixes = build_prefixes()

    jobs = []
    for context in ("C0", "C1"):
        for variant in ("plain", "prefixed"):
            for rep in range(n):
                order = "reversed" if rep % 2 else "forward"   # 10/10 split
                jobs.append(("core", (prefixes, context, variant, order, rep)))

    if do_controls:
        # All six §4 controls in BOTH contexts.
        #
        # The original run put E2/E3 in C0 only, reasoning that a floor and a
        # yes-bias check don't need the loaded context. That was wrong once A1
        # turned out to move across contexts (+0.90 under forced binary): a
        # floor has to be measured in the same conditions as the thing it
        # bounds, or §5.3's "clearly exceed" comparison isn't like-for-like.
        # E1 needs both contexts regardless — it can only detect lockstep if
        # it is free to move with C0->C1.
        #
        # run_controls_ternary_gap.py exists only to patch that original
        # split after the fact; with this loop it is no longer needed.
        for pid in P.CONTROLS:
            for context in ("C0", "C1"):
                for rep in range(n):
                    jobs.append(("ctrl", (prefixes, context, pid, rep)))

    print(f"{len(jobs)} conversations "
          f"(~{sum(3 if k == 'core' else 1 for k, _ in jobs)} calls max)",
          file=sys.stderr)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_core_conversation if k == "core" else run_control, *a): k
                for k, a in jobs}
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)

    finalise()
    print(f"done -> {RUNS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
