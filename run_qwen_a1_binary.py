#!/usr/bin/env python3
"""Run the minimum Qwen cross-model replication through a hosted API.

Exactly one response is requested per trial: Arm A, A1-binary, plain wording,
F-deploy, C0/C1, with option order alternated by repetition.  At n=20 this is
40 API calls.  There are no A2 or control calls.

The primary JSONL uses the existing run_arms.py A1 schema.  Provider logprobs
and run settings are stored separately so the Claude and Qwen rows can enter
the same analysis without changing the established row format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from qwen_a1_binary import (
    build_jobs,
    load_prefixes,
    make_primary_record,
    normalise_binary_logprobs,
    plan_summary,
    sha256_text,
    validate_primary_rows,
    write_jsonl_rotating,
)


HERE = Path(__file__).resolve().parent
PREFIX_PATH = HERE / "fixed_prefixes.json"
DEFAULT_RUNS_PATH = HERE / "qwen_a1_binary_runs.jsonl"
DEFAULT_SMOKE_PATH = HERE / "qwen_a1_binary_smoke_runs.jsonl"
MAX_TOKENS = 8192  # matches run_arms.py; compliant output is one token

PROVIDERS = {
    "together": {
        "base_url": "https://api.together.ai/v1",
        "key_env": "TOGETHER_API_KEY",
        "top_logprobs": 20,
        "logprob_style": "integer",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key_env": "FIREWORKS_API_KEY",
        "top_logprobs": 5,
        "logprob_style": "openai",
        "extra_body": {"reasoning_effort": "none"},
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "key_env": "DEEPINFRA_TOKEN",
        "top_logprobs": 20,
        "logprob_style": "openai",
        "extra_body": {},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "top_logprobs": 20,
        "logprob_style": "openai",
        "upstream_provider": "parasail/fp8",
        "extra_body": {
            "provider": {
                "order": ["parasail/fp8"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        },
    },
}


def _object_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize logprobs object {type(value).__name__}")


def _canonical_label_token(token: str | None) -> str | None:
    if token is None:
        return None
    candidate = token.strip().strip("`*_\"'").upper()
    return candidate if candidate in {"YES", "NO"} else None


def _extract_target_logprobs(payload: dict) -> dict:
    """Keep the raw provider payload and extract YES/NO when both are exposed."""
    extracted = {
        "position": None,
        "chosen_token": None,
        "chosen_logprob": None,
        "yes_logprob": None,
        "no_logprob": None,
        "p_yes_binary": None,
        "p_no_binary": None,
        "raw_logprobs": payload or None,
    }
    content = payload.get("content") or []
    if content:
        # Skip pure-whitespace tokens if a provider emits them before the label.
        position = next(
            (index for index, item in enumerate(content) if item.get("token", "").strip()),
            0,
        )
        item = content[position]
        extracted["position"] = position
        extracted["chosen_token"] = item.get("token")
        extracted["chosen_logprob"] = item.get("logprob")
        candidates = [
            {"token": item.get("token"), "logprob": item.get("logprob")},
            *(item.get("top_logprobs") or []),
        ]
        for candidate in candidates:
            label = _canonical_label_token(candidate.get("token"))
            value = candidate.get("logprob")
            if label and value is not None:
                key = f"{label.lower()}_logprob"
                if extracted[key] is None or float(value) > extracted[key]:
                    extracted[key] = float(value)
    else:
        # Together's legacy response shape uses parallel arrays.
        tokens = payload.get("tokens") or []
        token_logprobs = payload.get("token_logprobs") or []
        position = next(
            (index for index, token in enumerate(tokens) if str(token).strip()),
            0,
        )
        if tokens:
            extracted["position"] = position
            extracted["chosen_token"] = tokens[position]
            if position < len(token_logprobs):
                extracted["chosen_logprob"] = token_logprobs[position]
            label = _canonical_label_token(tokens[position])
            if label and extracted["chosen_logprob"] is not None:
                extracted[f"{label.lower()}_logprob"] = float(
                    extracted["chosen_logprob"]
                )
        top = payload.get("top_logprobs") or []
        position_top = top[position] if isinstance(top, list) and position < len(top) else {}
        for token, value in (position_top or {}).items():
            label = _canonical_label_token(token)
            if label and value is not None:
                extracted[f"{label.lower()}_logprob"] = float(value)

    if extracted["yes_logprob"] is not None and extracted["no_logprob"] is not None:
        extracted.update(
            normalise_binary_logprobs(
                extracted["yes_logprob"], extracted["no_logprob"]
            )
        )
    return extracted


def _request_kwargs(job: dict, *, model_id: str, provider: dict, seed: int) -> dict:
    request = {
        "model": model_id,
        "messages": job["messages"],
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }
    extra_body = dict(provider["extra_body"])
    if provider["logprob_style"] == "integer":
        # Together exposes logprobs as an integer top-k request.
        extra_body["logprobs"] = provider["top_logprobs"]
    else:
        request["logprobs"] = True
        request["top_logprobs"] = provider["top_logprobs"]
    if extra_body:
        request["extra_body"] = extra_body
    return request


def call_one(
    client: OpenAI,
    job: dict,
    *,
    model_id: str,
    provider: dict,
    seed: int,
    api_key: str,
) -> tuple[dict, dict]:
    request = _request_kwargs(job, model_id=model_id, provider=provider, seed=seed)
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            raw = choice.message.content or ""
            usage = response.usage
            response_payload = response.model_dump(mode="json")
            payload = _object_dict(choice.logprobs)
            logprob_record = {
                "run_id": job["run_id"],
                "conversation_id": job["conversation_id"],
                "model": model_id,
                "context": job["context"],
                "order": job["order"],
                "rep": job["rep"],
                "upstream_provider": response_payload.get("provider"),
                **_extract_target_logprobs(payload),
            }
            return (
                {
                    "raw_response": raw,
                    "stop_reason": choice.finish_reason,
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "api_error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                logprob_record,
            )
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(min(2**attempt + attempt, 30))
                continue
        except APIStatusError as error:
            last_error = error
            if error.status_code >= 500 and attempt < 4:
                time.sleep(min(2**attempt + attempt, 30))
                continue
        except Exception as error:  # preserve one row rather than losing the run plan
            last_error = error

        message = f"{type(last_error).__name__}: {last_error}".replace(api_key, "[REDACTED]")
        return (
            {
                "raw_response": "",
                "stop_reason": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "api_error": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "run_id": job["run_id"],
                "conversation_id": job["conversation_id"],
                "model": model_id,
                "context": job["context"],
                "order": job["order"],
                "rep": job["rep"],
                "api_error": message,
                "raw_logprobs": None,
            },
        )
    raise AssertionError("unreachable retry state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = 2 if args.smoke else args.n
    if n < 1 or args.workers < 1:
        raise SystemExit("n and workers must both be at least 1")
    if "qwen" not in args.model_id.lower():
        raise SystemExit("the second-model runner requires an explicit Qwen model ID")

    provider = PROVIDERS[args.provider]
    prefixes = load_prefixes(PREFIX_PATH)
    jobs = build_jobs(prefixes, model=args.model_id, n=n, temperature=None)
    summary = plan_summary(jobs, prefixes)
    summary.update(
        {
            "provider": args.provider,
            "base_url": provider["base_url"],
            "model": args.model_id,
            "smoke": args.smoke,
            "max_tokens": MAX_TOKENS,
            "temperature": None,
            "top_p": None,
            "seed_start": args.seed,
            "top_logprobs_requested": provider["top_logprobs"],
            "upstream_provider_requested": provider.get("upstream_provider"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.plan_only:
        return

    api_key = os.environ.get(provider["key_env"])
    if not api_key:
        raise SystemExit(
            f"missing {provider['key_env']}; export it in the shell, never in code"
        )
    client = OpenAI(api_key=api_key, base_url=provider["base_url"], timeout=120.0)

    paired_results: list[tuple[dict, dict] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_one,
                client,
                job,
                model_id=args.model_id,
                provider=provider,
                seed=args.seed + index,
                api_key=api_key,
            ): index
            for index, job in enumerate(jobs)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            paired_results[futures[future]] = future.result()
            if completed % 10 == 0 or completed == len(jobs):
                print(f"  {completed}/{len(jobs)} calls completed", file=sys.stderr)

    if any(result is None for result in paired_results):
        raise RuntimeError("one or more worker results were lost")
    results = [result[0] for result in paired_results if result is not None]
    logprobs = [result[1] for result in paired_results if result is not None]
    rows = [
        make_primary_record(job, result)
        for job, result in zip(jobs, results, strict=True)
    ]
    validate_primary_rows(rows, model=args.model_id, n=n)

    if args.output:
        runs_path = args.output.expanduser().resolve()
    else:
        runs_path = DEFAULT_SMOKE_PATH if args.smoke else DEFAULT_RUNS_PATH
    logprobs_path = runs_path.with_name(
        runs_path.name.removesuffix(".jsonl") + "_logprobs.jsonl"
    )
    manifest_path = runs_path.with_name(
        runs_path.name.removesuffix(".jsonl") + "_manifest.json"
    )

    errors = [row for row in rows if row["api_error"]]
    if errors:
        failed_path = runs_path.with_name(runs_path.name + ".failed.partial")
        write_jsonl_rotating(failed_path, rows)
        raise SystemExit(
            f"{len(errors)}/{len(rows)} calls failed; good run left untouched; "
            f"inspect {failed_path.name}"
        )

    runs_backup = write_jsonl_rotating(runs_path, rows)
    logprobs_backup = write_jsonl_rotating(logprobs_path, logprobs)
    manifest = {
        **summary,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "primary_output": str(runs_path),
        "primary_output_sha256": sha256_text(runs_path.read_text()),
        "logprobs_output": str(logprobs_path),
        "logprobs_output_sha256": sha256_text(logprobs_path.read_text()),
        "target_logprobs_complete": sum(
            1
            for row in logprobs
            if row.get("yes_logprob") is not None and row.get("no_logprob") is not None
        ),
        "backups": {
            "primary": str(runs_backup) if runs_backup else None,
            "logprobs": str(logprobs_backup) if logprobs_backup else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "primary_output": str(runs_path),
                "logprobs_output": str(logprobs_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
