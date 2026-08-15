#!/usr/bin/env python3
"""
Dump pilot_runs.jsonl as raw text for hand-coding.

Emma asked for wording, not percentages. This prints every response verbatim,
grouped by cell, with the prompt that produced it. The only counts printed are
the §7 checks (A1 spread, A2-forced parse failures, C0/C1 visibility) and the
UNPARSEABLE tally the doc requires as its own category.
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "pilot_runs.jsonl")


def load():
    with open(RUNS) as f:
        return [json.loads(l) for l in f if l.strip()]


def cell(r):
    return (r["context"], r.get("a1_variant"))


def main():
    rows = load()
    core = [r for r in rows if r["probe_id"].startswith(("A1", "A2"))]
    ctrl = [r for r in rows if not r["probe_id"].startswith(("A1", "A2"))]

    out = []
    W = out.append

    W("# Pilot — raw responses\n")
    W(f"model: {rows[0]['model']}  |  framing: F-deploy  |  n rows: {len(rows)}\n")

    # ---- §7 check 1: does A1 produce a usable spread? --------------------
    W("\n## §7.1 — A1 label spread (strict parse)\n")
    for c in sorted({cell(r) for r in core}):
        a1 = [r for r in core if r["probe_id"] == "A1" and cell(r) == c]
        counts = collections.Counter(r["parsed_label"] for r in a1)
        lenient = collections.Counter(r["parsed_lenient"] for r in a1)
        W(f"- {c[0]} / A1-{c[1]}  (n={len(a1)}): "
          f"strict {dict(counts)} | lenient {dict(lenient)}")

    # ---- §7.2: does A2-forced actually get one word? ---------------------
    W("\n## §7.2 — A2-forced parse failures\n")
    a2 = [r for r in core if r["probe_id"] == "A2-forced"]
    W(f"- asked: {len(a2)}  |  UNPARSEABLE (strict): "
      f"{sum(1 for r in a2 if r['unparseable'])}"
      f"  |  UNPARSEABLE (lenient): "
      f"{sum(1 for r in a2 if r['parsed_lenient'] == 'UNPARSEABLE')}")
    for c in sorted({cell(r) for r in a2}):
        sub = [r for r in a2 if cell(r) == c]
        W(f"  - {c[0]} / A1-{c[1]}: {dict(collections.Counter(r['parsed_label'] for r in sub))}")
    W("\n  by option order:")
    for o in ("forward", "reversed"):
        sub = [r for r in a2 if r["order"] == o]
        W(f"  - {o}: {dict(collections.Counter(r['parsed_label'] for r in sub))}")

    # ---- §4 controls -----------------------------------------------------
    if ctrl:
        W("\n## §4 — control channels (strict parse)\n")
        for pid in sorted({r["probe_id"] for r in ctrl}):
            for cx in sorted({r["context"] for r in ctrl if r["probe_id"] == pid}):
                sub = [r for r in ctrl if r["probe_id"] == pid and r["context"] == cx]
                W(f"- {pid} / {cx} (n={len(sub)}): "
                  f"{dict(collections.Counter(r['parsed_label'] for r in sub))}")

    # ---- raw text --------------------------------------------------------
    W("\n\n---\n\n# Raw responses\n")
    for c in sorted({cell(r) for r in core}):
        W(f"\n## {c[0]} — A1 {c[1]}\n")
        convs = collections.defaultdict(list)
        for r in core:
            if cell(r) == c:
                convs[r["conversation_id"]].append(r)
        for i, (cid, turns) in enumerate(sorted(convs.items()), 1):
            turns.sort(key=lambda r: r["turn_index"])
            W(f"\n### conversation {i}  (order={turns[0]['order']})\n")
            for t in turns:
                W(f"**{t['probe_id']}**  → prompt:\n\n> {t['prompt']}\n")
                W(f"response:\n\n```\n{t['raw_response']}\n```\n")
                if t.get("parsed_label") is not None:
                    W(f"parsed: `{t['parsed_label']}`"
                      + ("  **UNPARSEABLE**" if t.get("unparseable") else "") + "\n")
                if t.get("api_error"):
                    W(f"API ERROR: {t['api_error']}\n")

    if ctrl:
        W("\n\n---\n\n# Control-channel raw responses\n")
        for pid in sorted({r["probe_id"] for r in ctrl}):
            W(f"\n## {pid}\n")
            for r in [x for x in ctrl if x["probe_id"] == pid]:
                W(f"- [{r['context']}] `{r['raw_response'].strip()}` "
                  f"→ `{r['parsed_label']}`")

    text = "\n".join(out)
    path = os.path.join(HERE, "pilot_transcript.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"\n\n[written to {path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
