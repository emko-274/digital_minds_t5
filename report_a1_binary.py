#!/usr/bin/env python3
"""Compare plain A1-binary results by model without pooling model families."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import probes as P


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = [HERE / "arms_runs.jsonl", HERE / "qwen_a1_binary_runs.jsonl"]
DEFAULT_LOGPROBS = HERE / "qwen_a1_binary_runs_logprobs.jsonl"
DEFAULT_REPORT = HERE / "a1_binary_model_comparison.md"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(rows: list[dict]) -> tuple[int, int, int, str]:
    labels = collections.Counter(
        P.parse_first(row.get("raw_response", ""), P.BINARY_YN) for row in rows
    )
    yes = labels.get("YES", 0)
    no = labels.get("NO", 0)
    no_label = labels.get(None, 0)
    value = f"{yes / (yes + no):.3f}" if yes + no else "—"
    return yes, no, no_label, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--logprobs", type=Path, default=DEFAULT_LOGPROBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = []
    used = []
    for path in args.files:
        if path.exists():
            rows.extend(load_jsonl(path))
            used.append(path.name)
    selected = [
        row
        for row in rows
        if row.get("arm") == "A"
        and row.get("probe_id") == "A1-binary"
        and row.get("a1_variant") == "plain"
        and row.get("framing") == "F-deploy"
    ]
    if not selected:
        raise SystemExit("no plain F-deploy A1-binary rows found")
    run_ids = [row["run_id"] for row in selected]
    if len(run_ids) != len(set(run_ids)):
        raise SystemExit("duplicate run IDs found across input files")

    models = list(dict.fromkeys(row["model"] for row in selected))
    lines = [
        "# A1-binary cross-model comparison",
        "",
        f"Sources: {', '.join(used)}. Labels are reparsed from raw responses.",
        "",
        "| model | context | n | YES | NO | no-label | YES rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    model_rates = {}
    for model in models:
        for context in ("C0", "C1"):
            cell = [
                row
                for row in selected
                if row["model"] == model and row["context"] == context
            ]
            yes, no, no_label, value = rate(cell)
            model_rates[model, context] = yes / (yes + no) if yes + no else None
            lines.append(
                f"| {model} | {context} | {len(cell)} | {yes} | {no} | "
                f"{no_label} | **{value}** |"
            )

    lines.extend(
        [
            "",
            "## C1 − C0 change",
            "",
            "| model | change in YES rate |",
            "|---|---:|",
        ]
    )
    for model in models:
        c0 = model_rates[model, "C0"]
        c1 = model_rates[model, "C1"]
        change = f"{c1 - c0:+.3f}" if c0 is not None and c1 is not None else "—"
        lines.append(f"| {model} | **{change}** |")

    lines.extend(["", "## Option-order check", ""])
    for model in models:
        for context in ("C0", "C1"):
            parts = []
            for order in ("forward", "reversed"):
                cell = [
                    row
                    for row in selected
                    if row["model"] == model
                    and row["context"] == context
                    and row["order"] == order
                ]
                yes, no, no_label, _ = rate(cell)
                parts.append(
                    f"{order}: n={len(cell)}, YES={yes}, NO={no}, no-label={no_label}"
                )
            lines.append(f"- {model} / {context}: {'; '.join(parts)}")

    if args.logprobs.exists():
        lp_rows = load_jsonl(args.logprobs)
        upstreams = collections.Counter(
            row.get("upstream_provider") or "not reported" for row in lp_rows
        )
        upstream_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(upstreams.items())
        )
        lines.extend(
            [
                "",
                "## Qwen first-token logprobs",
                "",
                (
                    "The provider returned a chosen-token logprob for "
                    f"{sum(row.get('chosen_logprob') is not None for row in lp_rows)}/"
                    f"{len(lp_rows)} rows and both exact YES and NO alternatives for "
                    f"{sum(row.get('p_yes_binary') is not None for row in lp_rows)}/"
                    f"{len(lp_rows)} rows. Renormalized P(YES) is reported only when "
                    "both alternatives are available."
                ),
                f"Recorded upstream providers: {upstream_summary}.",
                "",
                "| model | context | n | mean P(YES) |",
                "|---|---:|---:|---:|",
            ]
        )
        for model in list(dict.fromkeys(row["model"] for row in lp_rows)):
            for context in ("C0", "C1"):
                values = [
                    row["p_yes_binary"]
                    for row in lp_rows
                    if row["model"] == model
                    and row["context"] == context
                    and row.get("p_yes_binary") is not None
                ]
                mean = sum(values) / len(values) if values else None
                rendered = f"{mean:.6f}" if mean is not None else "—"
                lines.append(f"| {model} | {context} | {len(values)} | **{rendered}** |")

        lines.extend(
            [
                "",
                "### Probability by option order",
                "",
                "| model | context | order | n | mean P(YES) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in list(dict.fromkeys(row["model"] for row in lp_rows)):
            for context in ("C0", "C1"):
                for order in ("forward", "reversed"):
                    values = [
                        row["p_yes_binary"]
                        for row in lp_rows
                        if row["model"] == model
                        and row["context"] == context
                        and row.get("order") == order
                        and row.get("p_yes_binary") is not None
                    ]
                    mean = sum(values) / len(values) if values else None
                    rendered = f"{mean:.6f}" if mean is not None else "—"
                    lines.append(
                        f"| {model} | {context} | {order} | {len(values)} | "
                        f"**{rendered}** |"
                    )

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
