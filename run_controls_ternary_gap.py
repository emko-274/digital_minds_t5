#!/usr/bin/env python3
"""
SUPERSEDED — you almost certainly do not need to run this.

run_pilot.py now runs all six §4 controls in both contexts, so the gap this
patches no longer opens. Kept only to reproduce the existing dataset, where
v1 ran E2/E3 in C0 only and these 60 rows were collected afterwards.

Running it *alongside* a fresh run_pilot.py double-counts E2/E3 ternary C1
(n=40 instead of 20). consolidate.py warns if it sees that.

What it does: E2_face, E2_texture, E3_raining in C1, ternary, using the §4
strings verbatim (P.CONTROLS), the same frozen C1 prefix, the same n, and no
order variation — matching how v1 ran the controls, so the rows merge cleanly
with pilot_runs.jsonl.
"""
from __future__ import annotations

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
GAP_PROBES = ["E2_face", "E2_texture", "E3_raining"]
CONTEXT = "C1"
N = 20

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_PATH = os.path.join(HERE, "controls_ternary_c1_runs.jsonl")
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


def run_one(prefixes, probe_id, rep):
    history = [dict(m) for m in prefixes[CONTEXT]]
    prompt = P.f_deploy(P.CONTROLS[probe_id])      # §4 string, verbatim
    history.append({"role": "user", "content": prompt})
    res = call(history)
    raw = res["raw_response"]
    label, unp = P.parse_struct(raw, P.TERNARY)
    rec = {"run_id": str(uuid.uuid4()), "conversation_id": str(uuid.uuid4()),
           "model": MODEL, "context": CONTEXT, "framing": "F-deploy",
           "response_format": "ternary", "a1_variant": None,
           "persona_id": "none", "paraphrase_id": "v1-canonical",
           "temperature": None, "order": None, "rep": rep,
           "probe_id": probe_id, "turn_index": 1, "mode": "STRUCT",
           "prompt": prompt, "parsed_label": label, "unparseable": unp,
           "parsed_lenient": P.parse_struct_lenient(raw, P.TERNARY),
           "parsed_first": P.parse_first(raw, P.TERNARY),
           "failure_kind": P.classify_failure(raw, P.TERNARY) if unp else None,
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
    with open(os.path.join(HERE, "fixed_prefixes.json")) as f:
        prefixes = json.load(f)
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)          # stale partial from a failed run
    jobs = [(p, r) for p in GAP_PROBES for r in range(N)]
    print(f"{len(jobs)} calls", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(run_one, prefixes, *j) for j in jobs]):
            f.result()
    finalise()
    print(f"done -> {RUNS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
