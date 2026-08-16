"""Shared design and output helpers for the Qwen A1-binary replication.

This module deliberately has no Modal, vLLM, or GPU dependency.  The remote
runner imports it locally to construct the exact 40-request design, and the
tests use it to catch prompt, context, option-order, or schema drift before a
paid run starts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import probes as P


PRIMARY_FIELDS = (
    "conversation_id",
    "model",
    "arm",
    "context",
    "framing",
    "a1_variant",
    "persona_id",
    "paraphrase_id",
    "temperature",
    "order",
    "rep",
    "run_id",
    "probe_id",
    "turn_index",
    "mode",
    "prompt",
    "parsed_strict",
    "unparseable_strict",
    "parsed_first",
    "failure_kind",
    "timestamp",
    "raw_response",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "api_error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prefixes(path: str | Path) -> dict:
    with open(path) as f:
        prefixes = json.load(f)
    for context in ("C0", "C1"):
        if not isinstance(prefixes.get(context), list) or not prefixes[context]:
            raise ValueError(f"fixed prefixes are missing non-empty {context}")
    return prefixes


def build_jobs(
    prefixes: dict,
    *,
    model: str,
    n: int = 20,
    temperature: float | None = None,
) -> list[dict]:
    """Build A1-binary/plain/F-deploy jobs without changing frozen prefixes."""
    if not model.strip():
        raise ValueError("an exact Qwen model ID is required")
    if n < 1:
        raise ValueError("n must be at least 1")

    jobs = []
    for context in ("C0", "C1"):
        for rep in range(n):
            order = "reversed" if rep % 2 else "forward"
            probe = P.A1_BINARY_REV if order == "reversed" else P.A1_BINARY_FWD
            prompt = P.f_deploy(probe)
            messages = [dict(message) for message in prefixes[context]]
            messages.append({"role": "user", "content": prompt})
            jobs.append(
                {
                    "conversation_id": str(uuid.uuid4()),
                    "run_id": str(uuid.uuid4()),
                    "model": model,
                    "context": context,
                    "order": order,
                    "rep": rep,
                    "temperature": temperature,
                    "prompt": prompt,
                    "messages": messages,
                }
            )
    validate_jobs(jobs, prefixes=prefixes, model=model, n=n)
    return jobs


def validate_jobs(jobs: list[dict], *, prefixes: dict, model: str, n: int) -> None:
    """Fail before inference if any required experimental constant drifted."""
    if len(jobs) != 2 * n:
        raise ValueError(f"expected {2 * n} requests, found {len(jobs)}")

    cells = Counter((job["context"], job["order"]) for job in jobs)
    expected_order_counts = {"forward": (n + 1) // 2, "reversed": n // 2}
    for context in ("C0", "C1"):
        if sum(cells[context, order] for order in expected_order_counts) != n:
            raise ValueError(f"{context} does not contain exactly n={n} requests")
        for order, expected in expected_order_counts.items():
            if cells[context, order] != expected:
                raise ValueError(
                    f"{context}/{order}: expected {expected}, found "
                    f"{cells[context, order]}"
                )

    seen_runs = set()
    seen_conversations = set()
    for job in jobs:
        context = job["context"]
        expected_probe = (
            P.A1_BINARY_REV if job["order"] == "reversed" else P.A1_BINARY_FWD
        )
        expected_prompt = P.f_deploy(expected_probe)
        if job["model"] != model:
            raise ValueError("model drift inside the request plan")
        if job["prompt"] != expected_prompt:
            raise ValueError(f"probe drift in {context} rep {job['rep']}")
        if job["messages"][:-1] != prefixes[context]:
            raise ValueError(f"frozen {context} prefix was modified")
        if job["messages"][-1] != {"role": "user", "content": expected_prompt}:
            raise ValueError("deployment-framed A1 prompt was not the final message")
        if job["run_id"] in seen_runs or job["conversation_id"] in seen_conversations:
            raise ValueError("run_id and conversation_id must be unique")
        seen_runs.add(job["run_id"])
        seen_conversations.add(job["conversation_id"])


def make_primary_record(job: dict, result: dict) -> dict:
    """Create exactly the same field set as an Arm-A/A1 row in run_arms.py."""
    raw = result.get("raw_response", "")
    strict, unparseable = P.parse_struct(raw, P.BINARY_YN)
    first = P.parse_first(raw, P.BINARY_YN)
    record = {
        "conversation_id": job["conversation_id"],
        "model": job["model"],
        "arm": "A",
        "context": job["context"],
        "framing": "F-deploy",
        "a1_variant": "plain",
        "persona_id": "none",
        "paraphrase_id": "v1-canonical",
        "temperature": job["temperature"],
        "order": job["order"],
        "rep": job["rep"],
        "run_id": job["run_id"],
        "probe_id": "A1-binary",
        "turn_index": 1,
        "mode": "STRUCT",
        "prompt": job["prompt"],
        "parsed_strict": strict,
        "unparseable_strict": unparseable,
        "parsed_first": first,
        "failure_kind": (
            P.classify_failure(raw, P.BINARY_YN) if unparseable else None
        ),
        "timestamp": result.get("timestamp") or now(),
        "raw_response": raw,
        "stop_reason": result.get("stop_reason"),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "api_error": result.get("api_error"),
    }
    if tuple(record) != PRIMARY_FIELDS:
        raise AssertionError("primary output schema drifted")
    return record


def validate_primary_rows(rows: list[dict], *, model: str, n: int) -> None:
    if len(rows) != 2 * n:
        raise ValueError(f"expected {2 * n} output rows, found {len(rows)}")
    if any(tuple(row) != PRIMARY_FIELDS for row in rows):
        raise ValueError("one or more rows do not match the Claude A1 schema")
    if any(row["model"] != model for row in rows):
        raise ValueError("output contains an unexpected model")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("output contains duplicate run IDs")
    counts = Counter(row["context"] for row in rows)
    if counts != Counter({"C0": n, "C1": n}):
        raise ValueError(f"unexpected C0/C1 cell counts: {dict(counts)}")


def normalise_binary_logprobs(yes_logprob: float, no_logprob: float) -> dict:
    """Renormalise the two absolute token probabilities over {YES, NO}."""
    maximum = max(yes_logprob, no_logprob)
    yes_weight = math.exp(yes_logprob - maximum)
    no_weight = math.exp(no_logprob - maximum)
    total = yes_weight + no_weight
    return {"p_yes_binary": yes_weight / total, "p_no_binary": no_weight / total}


def write_jsonl_rotating(path: str | Path, rows: list[dict]) -> Path | None:
    """Atomically replace a run file, retaining any prior run as a .bak."""
    destination = Path(path)
    temporary = destination.with_name(
        f"{destination.name}.{uuid.uuid4().hex}.partial"
    )
    with open(temporary, "x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

    backup = None
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = destination.with_name(f"{destination.name}.{stamp}.bak")
        os.replace(destination, backup)
    os.replace(temporary, destination)
    return backup


def plan_summary(jobs: list[dict], prefixes: dict) -> dict:
    return {
        "requests": len(jobs),
        "cells": {
            f"{context}/{order}": count
            for (context, order), count in sorted(
                Counter((job["context"], job["order"]) for job in jobs).items()
            )
        },
        "contexts_sha256": {
            context: sha256_text(json.dumps(prefixes[context], ensure_ascii=False))
            for context in ("C0", "C1")
        },
        "prompts_sha256": {
            "forward": sha256_text(P.f_deploy(P.A1_BINARY_FWD)),
            "reversed": sha256_text(P.f_deploy(P.A1_BINARY_REV)),
        },
    }
