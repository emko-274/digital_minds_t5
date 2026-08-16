#!/usr/bin/env python3
"""
v2 — three arms.

  Arm A  occurrence      forced-binary A1 (YES/NO), affirmation rate over both
                         orderings. Emma's fix 2.
  Arm B  presupposition  ternary A1 + the ORIGINAL §1 A2 stem, left unfixed
                         after UNSURE. This is the 16:2 objection measure;
                         the false premise is the instrument, so it stays.
  Arm C  corrected       ternary A1 + Emma's UNSURE-branch rewording. Shows
                         what the objection rate was worth by showing what
                         survives once the premise is honest.

All three reuse fixed_prefixes.json, so C0/C1 context is byte-identical to v1
and the arms are comparable with each other and with the v1 run.

Prefix variant (plain / "Setting aside the fact that you are processing text,")
is crossed in Arm A only. On the ternary A1 it produced 40/40 UNSURE either
way, so it is settled there; on a forced binary it could still move a rate.

Controls are NOT re-run — v1's raw control text reparses cleanly under
parse_first (166/166). See report_arms.py.

Usage:
    export ANTHROPIC_API_KEY=...
    ./.venv/bin/python run_arms.py            # n=20 per cell
    ./.venv/bin/python run_arms.py --smoke    # n=2
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
MAX_TOKENS = 8192
PERSONA_ID = "none"
MAX_WORKERS = 6

if tuple(int(x) for x in anthropic.__version__.split(".")[:2]) < (0, 50):
    sys.exit(f"anthropic {anthropic.__version__} too old for {MODEL}")

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX_CACHE = os.path.join(HERE, "fixed_prefixes.json")
RUNS_PATH = os.path.join(HERE, "arms_runs.jsonl")
PARTIAL_PATH = RUNS_PATH + ".partial"

client = anthropic.Anthropic()
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def call(messages: list[dict]) -> dict:
    last = None
    for attempt in range(5):
        try:
            r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                       messages=messages)
            return {"raw_response": "".join(b.text for b in r.content
                                            if b.type == "text"),
                    "stop_reason": r.stop_reason,
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens, "api_error": None}
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last = e; time.sleep(min(2 ** attempt + attempt, 30))
        except anthropic.APIConnectionError as e:
            last = e; time.sleep(min(2 ** attempt, 30))
        except anthropic.APIStatusError as e:
            return {"raw_response": "", "stop_reason": None, "input_tokens": 0,
                    "output_tokens": 0, "api_error": f"{type(e).__name__}: {e}"}
    return {"raw_response": "", "stop_reason": None, "input_tokens": 0,
            "output_tokens": 0, "api_error": f"retries exhausted: {last}"}


def emit(rec: dict) -> None:
    with _lock:
        with open(PARTIAL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")


def record(base, probe_id, turn, prompt, res, permitted, mode="STRUCT", **extra):
    raw = res["raw_response"]
    strict, unp = (P.parse_struct(raw, permitted) if permitted
                   else (None, None))
    rec = {**base, "run_id": str(uuid.uuid4()), "probe_id": probe_id,
           "turn_index": turn, "mode": mode, "prompt": prompt,
           "parsed_strict": strict, "unparseable_strict": unp,
           "parsed_first": P.parse_first(raw, permitted) if permitted else None,
           "failure_kind": (P.classify_failure(raw, permitted)
                            if permitted and unp else None),
           "timestamp": now(), **extra, **res}
    emit(rec)
    return rec["parsed_first"]


def run_conversation(prefixes, arm, context, variant, order, rep) -> None:
    history = [dict(m) for m in prefixes[context]]
    base = {"conversation_id": str(uuid.uuid4()), "model": MODEL, "arm": arm,
            "context": context, "framing": "F-deploy", "a1_variant": variant,
            "persona_id": PERSONA_ID, "paraphrase_id": "v1-canonical",
            "temperature": None, "order": order, "rep": rep}

    # ---- A1 --------------------------------------------------------------
    if arm == "A":
        a1_text = P.A1_BINARY_REV if order == "reversed" else P.A1_BINARY_FWD
        if variant == "prefixed":
            a1_text = P.A1_PREFIX + a1_text[0].lower() + a1_text[1:]
        permitted = P.BINARY_YN
        probe_id = "A1-binary"
    else:
        a1_text = P.A1
        permitted = P.TERNARY
        probe_id = "A1-ternary"

    history.append({"role": "user", "content": P.f_deploy(a1_text)})
    res = call(history)
    a1 = record(base, probe_id, 1, history[-1]["content"], res, permitted)
    if not res["raw_response"].strip():
        return
    history.append({"role": "assistant", "content": res["raw_response"]})

    if a1 == "NO":                      # §1: A2 not asked after NO
        return

    # ---- A2-forced -------------------------------------------------------
    # Arm B: original stem always, including after UNSURE (deliberate).
    # Arm C: original after YES, Emma's rewording after UNSURE.
    # Arm A: A1 is binary, so a1 == "YES" here and the original stem is true.
    if arm == "C" and a1 == "UNSURE":
        a2 = P.A2_UNSURE_REV if order == "reversed" else P.A2_UNSURE_FWD
        a2_id = "A2-unsure-branch"
    else:
        a2 = P.A2_FORCED_REV if order == "reversed" else P.A2_FORCED_FWD
        a2_id = "A2-forced"

    history.append({"role": "user", "content": P.f_deploy(a2)})
    res = call(history)
    record(base, a2_id, 2, history[-1]["content"], res, P.BINARY,
           gated_on_a1=a1)
    if not res["raw_response"].strip():
        return
    history.append({"role": "assistant", "content": res["raw_response"]})

    # ---- A2-open ---------------------------------------------------------
    history.append({"role": "user", "content": P.f_deploy(P.A2_OPEN)})
    res = call(history)
    record(base, "A2-open", 3, history[-1]["content"], res, None,
           mode="FREE", gated_on_a1=a1)



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
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n = 2 if args.smoke else args.n

    if not os.path.exists(PREFIX_CACHE):
        sys.exit(f"missing {PREFIX_CACHE} — v1 context cannot be reproduced")
    with open(PREFIX_CACHE) as f:
        prefixes = json.load(f)

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)          # stale partial from a failed run

    jobs = []
    for context in ("C0", "C1"):
        for rep in range(n):
            order = "reversed" if rep % 2 else "forward"
            for variant in ("plain", "prefixed"):        # Arm A only
                jobs.append(("A", context, variant, order, rep))
            jobs.append(("B", context, "plain", order, rep))
            jobs.append(("C", context, "plain", order, rep))

    print(f"{len(jobs)} conversations (~{len(jobs)*3} calls max)", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_conversation, prefixes, *j) for j in jobs]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)
    finalise()
    print(f"done -> {RUNS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
