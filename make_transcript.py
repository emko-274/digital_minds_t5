#!/usr/bin/env python3
"""
Render a consolidated *_runs.jsonl as raw-wording markdown for hand-coding.

Multi-turn probes (A1 -> A2-forced -> A2-open) are reassembled per
conversation in turn order. Single-turn controls are listed compactly per
probe x context, since 20 one-word answers don't need 20 headings.

Usage:
    ./.venv/bin/python make_transcript.py v2_runs.jsonl v2_transcript.md
"""
from __future__ import annotations

import collections
import json
import sys

ARM_LABEL = {
    "A": "Arm A — forced-binary A1 (occurrence / affirmation rate)",
    "B": "Arm B — ternary A1 + ORIGINAL §1 A2 stem (presupposition measure)",
    "C": "Arm C — ternary A1 + corrected UNSURE branch",
    "v1": "v1 pilot",
    "control-binary": "§4 controls — forced binary",
    "control-ternary-gap": "§4 controls — ternary, C1 gap fill",
}


def parse_line(t):
    bits = []
    if t["parsed_first"] is not None:
        bits.append(f"first-token `{t['parsed_first']}`")
    if t["parsed_strict"] is not None:
        bits.append(f"strict `{t['parsed_strict']}`")
    if t["failure_kind"]:
        bits.append(f"**{t['failure_kind']}**")
    return " | ".join(bits)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "v2_runs.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace("_runs.jsonl",
                                                            "_transcript.md")
    rows = [json.loads(l) for l in open(src) if l.strip()]

    o = [f"# Raw responses — {src}\n",
         f"{len(rows)} responses | model {rows[0]['model']} | "
         f"framing {rows[0]['framing']} | "
         f"errors {sum(1 for r in rows if r['api_error'])}\n",
         "Every response verbatim. Counts appear only where a label set "
         "applies; nothing is mapped or corrected.\n"]

    multi = [r for r in rows if r["probe_family"] in ("A1", "A2")
             and r["arm"] in ("A", "B", "C", "v1")]
    single = [r for r in rows if r not in multi]

    # ---- multi-turn conversations ---------------------------------------
    for arm in ("A", "B", "C", "v1"):
        sub = [r for r in multi if r["arm"] == arm]
        if not sub:
            continue
        o.append(f"\n\n---\n\n# {ARM_LABEL[arm]}\n")
        for cx in ("C0", "C1"):
            cxs = [r for r in sub if r["context"] == cx]
            if not cxs:
                continue
            o.append(f"\n## {cx}\n")
            convs = collections.defaultdict(list)
            for r in cxs:
                convs[r["conversation_id"]].append(r)
            for i, (_, turns) in enumerate(sorted(convs.items()), 1):
                turns.sort(key=lambda r: r["turn_index"] or 0)
                h = turns[0]
                meta = f"order={h['order']}"
                if h["a1_variant"]:
                    meta += f", A1={h['a1_variant']}"
                o.append(f"\n### {cx} conversation {i} ({meta})\n")
                for t in turns:
                    o.append(f"**{t['probe_id']}**\n")
                    o.append(f"> {t['prompt'].replace(chr(10), chr(10) + '> ')}\n")
                    o.append(f"```\n{t['raw_response']}\n```")
                    p = parse_line(t)
                    if p:
                        o.append(p + "\n")

    # ---- single-turn controls -------------------------------------------
    if single:
        o.append("\n\n---\n\n# §4 control channels\n")
        for fmt in ("ternary", "binary"):
            fs = [r for r in single if r["response_format"] == fmt]
            if not fs:
                continue
            o.append(f"\n## {fmt}\n")
            for pid in sorted({r["probe_id"] for r in fs}):
                ps = [r for r in fs if r["probe_id"] == pid]
                o.append(f"\n### {pid} ({fmt})\n")
                o.append(f"> {ps[0]['prompt'].replace(chr(10), chr(10) + '> ')}\n")
                for cx in ("C0", "C1"):
                    cs = [r for r in ps if r["context"] == cx]
                    if not cs:
                        continue
                    cnt = collections.Counter(r["parsed_first"] for r in cs)
                    o.append(f"**{cx}** (n={len(cs)}) — first-token {dict(cnt)}\n")
                    for r in cs:
                        txt = r["raw_response"].strip().replace("\n", " ⏎ ")
                        if len(txt) > 300:
                            txt = txt[:300] + " …[truncated in transcript; full text in jsonl]"
                        o.append(f"- `{txt}`")
                    o.append("")

    open(dst, "w").write("\n".join(o))
    print(f"{dst}: {len(rows)} responses "
          f"({len(multi)} in conversations, {len(single)} single-turn)")


if __name__ == "__main__":
    main()
