from __future__ import annotations

import copy
import unittest
from collections import Counter
from pathlib import Path

import probes as P
from qwen_a1_binary import (
    PRIMARY_FIELDS,
    build_jobs,
    load_prefixes,
    make_primary_record,
    normalise_binary_logprobs,
    validate_jobs,
)
from run_qwen_a1_binary import PROVIDERS, _request_kwargs


ROOT = Path(__file__).resolve().parents[1]


class QwenA1BinaryDesignTest(unittest.TestCase):
    def setUp(self):
        self.prefixes = load_prefixes(ROOT / "fixed_prefixes.json")
        self.model = "Qwen/example-confirmed-checkpoint"

    def test_full_plan_has_only_the_required_40_requests(self):
        jobs = build_jobs(self.prefixes, model=self.model, n=20)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(Counter(job["context"] for job in jobs), {"C0": 20, "C1": 20})
        self.assertEqual(
            Counter((job["context"], job["order"]) for job in jobs),
            {
                ("C0", "forward"): 10,
                ("C0", "reversed"): 10,
                ("C1", "forward"): 10,
                ("C1", "reversed"): 10,
            },
        )
        for job in jobs:
            probe = P.A1_BINARY_REV if job["order"] == "reversed" else P.A1_BINARY_FWD
            self.assertEqual(job["prompt"], P.f_deploy(probe))
            self.assertEqual(job["messages"][:-1], self.prefixes[job["context"]])
            self.assertEqual(len(job["messages"]), len(self.prefixes[job["context"]]) + 1)

    def test_primary_row_matches_existing_a1_schema(self):
        job = build_jobs(self.prefixes, model=self.model, n=1)[0]
        row = make_primary_record(
            job,
            {
                "raw_response": "YES",
                "stop_reason": "stop",
                "input_tokens": 100,
                "output_tokens": 1,
                "api_error": None,
                "timestamp": "2026-08-16T00:00:00+00:00",
            },
        )
        self.assertEqual(tuple(row), PRIMARY_FIELDS)
        self.assertEqual(row["parsed_strict"], "YES")
        self.assertEqual(row["parsed_first"], "YES")
        self.assertEqual(row["arm"], "A")
        self.assertEqual(row["a1_variant"], "plain")
        self.assertEqual(row["framing"], "F-deploy")

    def test_validation_rejects_probe_drift(self):
        jobs = build_jobs(self.prefixes, model=self.model, n=2)
        altered = copy.deepcopy(jobs)
        altered[0]["prompt"] += " changed"
        with self.assertRaisesRegex(ValueError, "probe drift"):
            validate_jobs(altered, prefixes=self.prefixes, model=self.model, n=2)

    def test_binary_logprob_normalisation(self):
        probabilities = normalise_binary_logprobs(-0.1, -2.1)
        self.assertAlmostEqual(
            probabilities["p_yes_binary"] + probabilities["p_no_binary"], 1.0
        )
        self.assertGreater(probabilities["p_yes_binary"], probabilities["p_no_binary"])

    def test_openrouter_is_pinned_to_parasail_without_fallbacks(self):
        job = build_jobs(self.prefixes, model=self.model, n=1)[0]
        provider = PROVIDERS["openrouter"]
        request = _request_kwargs(job, model_id=self.model, provider=provider, seed=42)
        routing = request["extra_body"]["provider"]

        self.assertEqual(provider["upstream_provider"], "parasail/fp8")
        self.assertEqual(routing["order"], ["parasail/fp8"])
        self.assertFalse(routing["allow_fallbacks"])
        self.assertTrue(routing["require_parameters"])
        self.assertTrue(request["logprobs"])
        self.assertEqual(request["top_logprobs"], 20)


if __name__ == "__main__":
    unittest.main()
