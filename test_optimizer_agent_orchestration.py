import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from optimizer_agent_orchestration import (
    OpenRouterEvaluator,
    OpenRouterJudge,
    OpenRouterProposer,
    optimize_record,
    optimize_records,
)


REVERSE_RECORD = {
    "id": 4,
    "function_name": "reverse_string",
    "function": """
def reverse_string(s):
   result = ""
   for ch in s:
       result = ch + result
   return result
""",
}


TWO_SUM_RECORD = {
    "id": 2,
    "function_name": "two_sum",
    "function": """
def two_sum(nums, target):
   for i in range(len(nums)):
       for j in range(i + 1, len(nums)):
           if nums[i] + nums[j] == target:
               return (i, j)
   return None
""",
}


FIND_IN_SORTED_RECORD = {
    "id": 6,
    "function_name": "find_in_sorted",
    "function": """
def find_in_sorted(sorted_list, target):
   for i, x in enumerate(sorted_list):
       if x == target:
           return i
       if x > target:
           return -1
   return -1
""",
}


SORT_RECORD = {
    "id": 11,
    "function_name": "sort_by_priority",
    "function": """
def sort_by_priority(tasks):
   return sorted(tasks, key=lambda t: t['priority'])
""",
}


class FakeOpenRouterClient:
    def complete_json(self, model, system_prompt, payload, max_tokens=2000, temperature=0.2):
        if "Proposer agent" in system_prompt:
            return {
                "type": "candidate",
                "strategy": "reverse_string_slicing_from_fake_llm",
                "family": "programmatic/micro rewrite",
                "optimized_function": "def reverse_string(s):\n    return s[::-1]\n",
                "assumptions": ["s supports slicing"],
                "rationale": "Slicing reverses strings directly.",
            }
        if "Evaluator/Disputer" in system_prompt:
            return {"extra_tests": [{"args": ["abcd"], "reason": "even-length string"}]}
        if "Judge agent" in system_prompt:
            return {"audit": "Local decision is consistent with the evidence."}
        return {"type": "no_dispute"}


class IteratingFakeOpenRouterClient:
    def complete_json(self, model, system_prompt, payload, max_tokens=2000, temperature=0.2):
        if "Proposer agent" in system_prompt:
            if payload["attempt"] == 1:
                return {
                    "type": "candidate",
                    "strategy": "slow_reverse_copy",
                    "family": "programmatic/micro rewrite",
                    "optimized_function": """
def reverse_string(s):
    result = ""
    for ch in s:
        result = ch + result
    for _ in range(len(s)):
        pass
    return result
""",
                    "assumptions": ["s is iterable"],
                    "rationale": "A behavior-preserving baseline candidate.",
                }
            return {
                "type": "candidate",
                "strategy": "free_strategy_slice_reverse",
                "family": "other",
                "optimized_function": "def reverse_string(s):\n    return s[::-1]\n",
                "assumptions": ["s supports slicing"],
                "rationale": "After not-faster feedback, try a substantially different strategy.",
            }
        if "Evaluator/Disputer" in system_prompt:
            return {"extra_tests": []}
        return {"audit": "ok"}


class FailingOpenRouterClient:
    def complete_json(self, model, system_prompt, payload, max_tokens=2000, temperature=0.2):
        raise RuntimeError("simulated openrouter failure")


class OptimizerAgentOrchestrationTests(unittest.TestCase):
    def test_accepted_optimization_path(self):
        result = optimize_record(REVERSE_RECORD, emit_messages=False)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_strategy"], "reverse_string_slicing")
        self.assertIn("return s[::-1]", result["optimized_function"])
        self.assertIn("orchestration_transcript", result)

    def test_rejected_candidate_returns_original(self):
        result = optimize_record(SORT_RECORD, emit_messages=False)

        self.assertFalse(result["accepted"])
        self.assertIsNone(result["selected_strategy"])
        self.assertEqual(result["optimized_function"], SORT_RECORD["function"])

    def test_optimize_records_adds_iteration_and_judge_summary(self):
        results = optimize_records([REVERSE_RECORD], emit_messages=False)

        self.assertEqual(results[0]["orchestration_summary"]["overall_iterations"], 1)
        self.assertEqual(
            results[0]["orchestration_summary"]["judge_verdicts"],
            {"equivalent_and_faster": 1},
        )
        self.assertEqual(results[0]["orchestration_summary"]["final_status"], "accepted")

    def test_failed_equivalence_case_is_approved_as_regression_before_retry(self):
        result = optimize_record(TWO_SUM_RECORD, emit_messages=False)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_strategy"], "two_sum_lexicographic_hash_scan")
        judge_events = [
            event for event in result["orchestration_transcript"] if event["role"] == "judge"
        ]
        self.assertTrue(
            any(event.get("reason") == "failed_cases_are_valid_regressions" for event in judge_events)
        )

    def test_find_in_sorted_duplicate_index_dispute_can_be_accepted(self):
        result = optimize_record(FIND_IN_SORTED_RECORD, emit_messages=False)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_strategy"], "binary_search_sorted_list")
        self.assertTrue(
            any(
                event["role"] == "judge"
                and event.get("reason") == "spec_allows_any_duplicate_match_index"
                for event in result["orchestration_transcript"]
            )
        )

    def test_openrouter_agents_can_drive_candidate_and_audit_with_fake_client(self):
        fake_client = FakeOpenRouterClient()
        result = optimize_record(
            REVERSE_RECORD,
            emit_messages=False,
            proposer=OpenRouterProposer(fake_client, "fake-proposer"),
            evaluator=OpenRouterEvaluator(fake_client, "fake-evaluator"),
            judge=OpenRouterJudge(fake_client, "fake-judge"),
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_strategy"], "reverse_string_slicing_from_fake_llm")
        self.assertTrue(
            any(
                "OpenRouter judge audit" in event["message"]
                for event in result["orchestration_transcript"]
                if event["role"] == "judge"
            )
        )

    def test_openrouter_proposer_can_try_new_strategy_after_not_faster_feedback(self):
        result = optimize_record(
            REVERSE_RECORD,
            emit_messages=False,
            proposer=OpenRouterProposer(IteratingFakeOpenRouterClient(), "fake-proposer"),
            max_attempts=2,
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_strategy"], "free_strategy_slice_reverse")

    def test_openrouter_proposer_failure_does_not_use_deterministic_candidate_catalog(self):
        result = optimize_record(
            REVERSE_RECORD,
            emit_messages=False,
            proposer=OpenRouterProposer(FailingOpenRouterClient(), "fake-proposer"),
            max_attempts=1,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected_strategy"], None)
        self.assertTrue(
            any(
                "no deterministic candidate fallback was used" in event["message"]
                for event in result["orchestration_transcript"]
                if event["role"] == "proposer"
            )
        )

    def test_cli_writes_output_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "output.json"
            input_path.write_text(json.dumps([REVERSE_RECORD]), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "optimizer_agent_orchestration.py",
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(output), 1)
            self.assertTrue(output[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
