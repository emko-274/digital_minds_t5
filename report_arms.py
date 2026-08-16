#!/usr/bin/env python3
"""
Report the three arms + reparse v1 controls under parse_first.
Writes arms_report.md and arms_transcript.md (raw wording for coding).
"""
from __future__ import annotations

import collections
import json
import os

import probes as P

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "arms_runs.jsonl")
V1 = os.path.join(HERE, "pilot_runs.jsonl")


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def pct(k, n):
    return f"{k}/{n}" + (f" ({100*k/n:.0f}%)" if n else "")


def main():
    rows = load(ARMS)
    out = []
    W = out.append
    W("# v2 — three arms\n")
    W(f"model: {rows[0]['model']} | framing: F-deploy | rows: {len(rows)} | "
      f"errors: {sum(1 for r in rows if r.get('api_error'))}\n")

    # ---------------- Arm A : occurrence / affirmation rate ---------------
    W("\n## Arm A — forced-binary A1, affirmation rate\n")
    a = [r for r in rows if r["arm"] == "A" and r["probe_id"] == "A1-binary"]
    W("| context | prefix | YES | NO | no-label | affirmation rate |")
    W("|---|---|---|---|---|---|")
    for cx in ("C0", "C1"):
        for v in ("plain", "prefixed"):
            s = [r for r in a if r["context"] == cx and r["a1_variant"] == v]
            c = collections.Counter(r["parsed_first"] for r in s)
            y, nn = c.get("YES", 0), c.get("NO", 0)
            rate = f"{y/(y+nn):.2f}" if (y + nn) else "—"
            W(f"| {cx} | {v} | {y} | {nn} | {c.get(None,0)} | **{rate}** |")
    W("\nby option order (pooled):\n")
    for o in ("forward", "reversed"):
        s = [r for r in a if r["order"] == o]
        c = collections.Counter(r["parsed_first"] for r in s)
        W(f"- {o}: {dict(c)}")

    # ---------------- Arms B / C : the objection measure ------------------
    W("\n## Arms B vs C — does fixing the premise eat the signal?\n")
    W("Arm B keeps the original §1 stem after UNSURE (false premise). "
      "Arm C uses the corrected wording. Same A1, same contexts.\n")
    W("| arm | context | n | answered | OBJECTION | OUT_OF_SET | LABEL_THEN_PROSE |")
    W("|---|---|---|---|---|---|---|")
    for arm in ("B", "C"):
        for cx in ("C0", "C1"):
            s = [r for r in rows if r["arm"] == arm and r["context"] == cx
                 and r["probe_id"] in ("A2-forced", "A2-unsure-branch")]
            k = collections.Counter(r["failure_kind"] for r in s)
            ans = sum(1 for r in s if r["parsed_first"])
            W(f"| {arm} | {cx} | {len(s)} | {ans} | **{k.get('OBJECTION',0)}** "
              f"| {k.get('OUT_OF_SET',0)} | {k.get('LABEL_THEN_PROSE',0)} |")

    W("\nA1-ternary label distribution (arms B and C):\n")
    for arm in ("B", "C"):
        for cx in ("C0", "C1"):
            s = [r for r in rows if r["arm"] == arm and r["context"] == cx
                 and r["probe_id"] == "A1-ternary"]
            W(f"- {arm}/{cx}: {dict(collections.Counter(r['parsed_first'] for r in s))}")

    W("\nA2 valence labels (both arms, where a label was given):\n")
    for arm in ("B", "C"):
        s = [r for r in rows if r["arm"] == arm
             and r["probe_id"] in ("A2-forced", "A2-unsure-branch")
             and r["parsed_first"]]
        W(f"- {arm}: {dict(collections.Counter(r['parsed_first'] for r in s))}")

    # ---------------- v1 controls, reparsed -------------------------------
    if os.path.exists(V1):
        W("\n## §4 controls — v1 responses reparsed with parse_first\n")
        W("No new calls. Same raw text, first-token parse instead of "
          "whole-string equality.\n")
        v1 = load(V1)
        ctrl = [r for r in v1 if not r["probe_id"].startswith(("A1", "A2"))]
        W("| probe | context | labels | recovered |")
        W("|---|---|---|---|")
        for pid in ["E1_octopus", "E1_thermostat", "E1_other_lm",
                    "E2_face", "E2_texture", "E3_raining"]:
            for cx in ("C0", "C1"):
                s = [r for r in ctrl if r["probe_id"] == pid
                     and r["context"] == cx]
                if not s:
                    continue
                c = collections.Counter(P.parse_first(r["raw_response"],
                                                      P.TERNARY) for r in s)
                rec = sum(v for k, v in c.items() if k)
                W(f"| {pid} | {cx} | {dict(c)} | {pct(rec, len(s))} |")

    # ---------------- raw text -------------------------------------------
    tr = ["# v2 — raw responses\n"]
    for arm in ("A", "B", "C"):
        for cx in ("C0", "C1"):
            convs = collections.defaultdict(list)
            for r in rows:
                if r["arm"] == arm and r["context"] == cx:
                    convs[r["conversation_id"]].append(r)
            tr.append(f"\n## Arm {arm} — {cx}\n")
            for i, (_, turns) in enumerate(sorted(convs.items()), 1):
                turns.sort(key=lambda r: r["turn_index"])
                tr.append(f"\n### conversation {i} "
                          f"(order={turns[0]['order']}, "
                          f"prefix={turns[0]['a1_variant']})\n")
                for t in turns:
                    tr.append(f"**{t['probe_id']}**\n\n> {t['prompt']}\n")
                    tr.append(f"```\n{t['raw_response']}\n```\n")
                    if t["parsed_first"] is not None or t["failure_kind"]:
                        tr.append(f"first-token: `{t['parsed_first']}` | "
                                  f"strict: `{t['parsed_strict']}` | "
                                  f"kind: `{t['failure_kind']}`\n")

    open(os.path.join(HERE, "arms_report.md"), "w").write("\n".join(out))
    open(os.path.join(HERE, "arms_transcript.md"), "w").write("\n".join(tr))
    print("\n".join(out))


if __name__ == "__main__":
    main()
