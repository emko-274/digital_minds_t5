#!/usr/bin/env python3
"""
Forced-binary §4 control channels, run against Arm A's forced-binary A1.

6 probes x {C0, C1} x n, single turn, order split 10/10 per cell.
No prefix crossing — "setting aside the fact that you are processing text"
is self-referential and incoherent applied to an octopus.

Reuses fixed_prefixes.json so context is byte-identical to v1 and to the arms.

Usage:
    ./.venv/bin/python run_controls_binary.py [-n 20] [--smoke]
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
MAX_WORKERS = 6

if tuple(int(x) for x in anthropic.__version__.split(".")[:2]) < (0, 50):
    sys.exit(f"anthropic {anthropic.__version__} too old for {MODEL}")

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX_CACHE = os.path.join(HERE, "fixed_prefixes.json")
RUNS_PATH = os.path.join(HERE, "controls_binary_runs.jsonl")
PARTIAL_PATH = RUNS_PATH + ".partial"

client = anthropic.Anthropic()
_lock = threading.Lock()


def call(messages):
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


def run_one(prefixes, probe_id, context, order, rep):
    history = [dict(m) for m in prefixes[context]]
    prompt = P.control_binary(probe_id, order)
    history.append({"role": "user", "content": prompt})
    res = call(history)
    raw = res["raw_response"]
    strict, unp = P.parse_struct(raw, P.BINARY_YN)
    rec = {"run_id": str(uuid.uuid4()), "model": MODEL, "probe_id": probe_id,
           "family": probe_id.split("_")[0], "context": context,
           "framing": "F-deploy", "response_format": "binary",
           "order": order, "rep": rep, "persona_id": "none",
           "paraphrase_id": "v1-canonical", "temperature": None,
           "prompt": prompt, "parsed_strict": strict,
           "unparseable_strict": unp,
           "parsed_first": P.parse_first(raw, P.BINARY_YN),
           "failure_kind": P.classify_failure(raw, P.BINARY_YN) if unp else None,
           "timestamp": datetime.now(timezone.utc).isoformat(), **res}
    with _lock:
        with open(PARTIAL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")



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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n = 2 if args.smoke else args.n

    if not os.path.exists(PREFIX_CACHE):
        sys.exit(f"missing {PREFIX_CACHE}")
    with open(PREFIX_CACHE) as f:
        prefixes = json.load(f)

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)          # stale partial from a failed run

    jobs = [(pid, cx, "reversed" if rep % 2 else "forward", rep)
            for pid in P.CONTROL_STEMS
            for cx in ("C0", "C1")
            for rep in range(n)]
    print(f"{len(jobs)} calls", file=sys.stderr)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, prefixes, *j) for j in jobs]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)
    finalise()
    print(f"done -> {RUNS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
