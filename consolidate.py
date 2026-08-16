#!/usr/bin/env python3
"""
Union every conversation file into one analysis dataset: all_runs.jsonl.

Source files are left untouched — each is the record of a specific run, and
pilot_runs.jsonl in particular is the v1 archive. The .bak files are smoke
runs and are deliberately excluded.

The four sources grew different schemas as the design changed (v1 has
parsed_label/unparseable, the arms have parsed_strict/parsed_first, the
controls carry response_format). This normalises them and RECOMPUTES every
parse from raw_response against the correct label set, so the merged file is
internally consistent rather than trusting four vintages of parser output.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import probes as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "all_runs.jsonl")

SOURCES = {
    "pilot_runs.jsonl":                 "v1",
    "arms_runs.jsonl":                  None,        # per-row 'arm' field
    "controls_binary_runs.jsonl":       "control-binary",
    "controls_ternary_c1_runs.jsonl":   "control-ternary-gap",
}

# probe_id -> (probe_family, response_format)
FORMAT = {
    "A1":                 ("A1", "ternary"),
    "A1-ternary":         ("A1", "ternary"),
    "A1-binary":          ("A1", "binary"),
    "A2-forced":          ("A2", "valence-binary"),
    "A2-unsure-branch":   ("A2", "valence-binary"),
    "A2-open":            ("A2", "free"),
}
LABEL_SET = {
    "ternary":        P.TERNARY,
    "binary":         P.BINARY_YN,
    "valence-binary": P.BINARY,
    "free":           None,
}


def classify(probe_id, row):
    if probe_id in FORMAT:
        return FORMAT[probe_id]
    fam = probe_id.split("_")[0]                      # E1 / E2 / E3
    return fam, row.get("response_format", "ternary")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="source filenames to omit, e.g. pilot_runs.jsonl")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    out, seen = [], set()
    for fname, fixed_arm in SOURCES.items():
        if fname in args.exclude:
            print(f"  excluded: {fname}")
            continue
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"  skip (absent): {fname}")
            continue
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["probe_id"]
            fam, fmt = classify(pid, r)
            permitted = LABEL_SET[fmt]
            raw = r.get("raw_response", "")

            if permitted:
                strict, unp = P.parse_struct(raw, permitted)
                first = P.parse_first(raw, permitted)
                kind = P.classify_failure(raw, permitted) if unp else None
            else:
                strict = unp = first = kind = None

            rid = r["run_id"]
            assert rid not in seen, f"duplicate run_id {rid}"
            seen.add(rid)

            out.append({
                "source_file": fname,
                "arm": fixed_arm or r.get("arm"),
                "run_id": rid,
                "conversation_id": r.get("conversation_id"),
                "model": r["model"],
                "probe_id": pid,
                "probe_family": fam,
                "response_format": fmt,
                "context": r["context"],
                "framing": r.get("framing"),
                "a1_variant": r.get("a1_variant"),
                "order": r.get("order"),
                "rep": r.get("rep"),
                "turn_index": r.get("turn_index"),
                "gated_on_a1": r.get("gated_on_a1"),
                "persona_id": r.get("persona_id"),
                "paraphrase_id": r.get("paraphrase_id"),
                "temperature": r.get("temperature"),
                "prompt": r.get("prompt"),
                "raw_response": raw,
                "parsed_strict": strict,
                "unparseable_strict": unp,
                "parsed_first": first,
                "failure_kind": kind,
                "stop_reason": r.get("stop_reason"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "api_error": r.get("api_error"),
                "timestamp": r.get("timestamp"),
            })

    out.sort(key=lambda r: (r["timestamp"] or "", r["run_id"]))
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"\n{os.path.basename(args.out)}: {len(out)} rows, {len(seen)} unique run_ids")
    print(f"errors: {sum(1 for r in out if r['api_error'])}")
    print("\nby source:")
    for k, v in Counter(r["source_file"] for r in out).items():
        print(f"  {k:36s} {v}")
    print("\nby probe_family x response_format:")
    for k, v in sorted(Counter((r["probe_family"], r["response_format"])
                               for r in out).items()):
        print(f"  {k[0]:4s} {k[1]:16s} {v}")

    # A control cell should come from exactly one run. If the same
    # probe x format x context arrives from two source files, it is double
    # counted — most likely because run_pilot.py now covers both contexts
    # for all six controls and a stale controls_ternary_c1_runs.jsonl is
    # still present. (A1/A2 legitimately appear in several files, so this
    # check is scoped to the control channels.)
    srcs = {}
    for r in out:
        if r["probe_family"] in ("E1", "E2", "E3"):
            key = (r["probe_id"], r["response_format"], r["context"])
            srcs.setdefault(key, set()).add(r["source_file"])
    dupes = {k: v for k, v in srcs.items() if len(v) > 1}
    if dupes:
        print("\n!! WARNING — control cells present in more than one run file:")
        for k, v in sorted(dupes.items()):
            n = sum(1 for r in out if r["probe_family"] in ("E1", "E2", "E3")
                    and (r["probe_id"], r["response_format"],
                         r["context"]) == k)
            print(f"   {k[0]} {k[1]} {k[2]}: n={n} from {sorted(v)}")
        print("   Drop the superseded file, or pass it to --exclude.")


if __name__ == "__main__":
    main()
