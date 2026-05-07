import argparse
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


MAX_ATTEMPTS = 2
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROPOSER_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_EVALUATOR_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.2-codex"


SEED_TESTS = {
    "has_duplicates": [[[1, 2, 1]], [[1, 2, 3]], [[]]],
    "two_sum": [[[4, 6, 10], 16], [[1, 2, 3, 4], 8]],
    "max_subarray_sum": [[[-2, 3, -1, 4]], [[-5]], [[-4, -2, -7]]],
    "reverse_string": [["abc"], [""], ["racecar"]],
    "fib": [[0], [1], [6], [10]],
    "find_in_sorted": [[[1, 3, 5, 7], 5], [[1, 3, 5, 7], 2]],
    "sum_of_squares": [[[1, 2, 3]], [[]], [[-2, 3]]],
    "flatten_one_level": [[[[1, 2], [3], []]], [[]]],
    "count_occurrences": [[["a", "b", "a"], "a"], [[1, 2, 3], 4], [[], 1]],
    "group_by_first_letter": [[["ant", "", "ape", "bat"]], [[]]],
    "sort_by_priority": [[[
        {"priority": 2},
        {"priority": 1},
    ]]],
    "safe_average": [[[2, 4, 6]], [[]], [[-1, 1]]],
}


ADVERSARIAL_TESTS = {
    "has_duplicates": [[["x", "y", "x"]]],
    "two_sum": [[[1, 2, 3, 4], 5]],
    "find_in_sorted": [[[1, 2, 2, 2, 3], 2]],
    "flatten_one_level": [[[[], ["a"], ["b", "c"]]]],
    "group_by_first_letter": [[["bob", "alice", "brad"]]],
}


BENCHMARK_CASES = {
    "has_duplicates": [[[i for i in range(800)] + [799]]],
    "two_sum": [[[i for i in range(1200)], 2397]],
    "max_subarray_sum": [[[1, -2, 3, 4, -1, 2, -5, 6] * 10]],
    "reverse_string": [["abcdefghijklmnopqrstuvwxyz" * 500]],
    "fib": [[25]],
    "find_in_sorted": [[[i * 2 for i in range(5000)], 7776]],
    "sum_of_squares": [[[i for i in range(5000)]]],
    "flatten_one_level": [[[[i, i + 1] for i in range(1500)]]],
    "count_occurrences": [[[i % 7 for i in range(10000)], 3]],
    "group_by_first_letter": [[["alpha", "ape", "beta", "bob", "gamma", ""] * 1000]],
    "safe_average": [[[i for i in range(10000)]]],
}


RUNNER = r"""
import json
import time


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    return value


payload = json.loads(input())
namespace = {"__builtins__": __builtins__}
exec(compile(payload["source"], "<agent-candidate>", "exec"), namespace)
func = namespace[payload["function_name"]]

if payload["mode"] == "run":
    results = []
    for args in payload["cases"]:
        try:
            results.append({"kind": "return", "value": normalize(func(*args))})
        except Exception as exc:
            results.append({"kind": "exception", "type": type(exc).__name__})
    print(json.dumps(results))
else:
    start = time.perf_counter()
    for _ in range(payload["repeat"]):
        for args in payload["cases"]:
            func(*args)
    print(json.dumps({"elapsed": time.perf_counter() - start}))
"""


@dataclass
class Proposal:
    strategy: str
    family: str
    source: str
    assumptions: list[str]
    prompt: str


@dataclass
class Evaluation:
    equivalent: bool
    faster: bool = False
    original_runtime_seconds: float | None = None
    optimized_runtime_seconds: float | None = None
    failed_cases: list[dict] = field(default_factory=list)
    comparator: str = "exact"
    prompt: str = ""


@dataclass
class JudgeDecision:
    accepted: bool
    reason: str
    approved_failed_cases: list[list] = field(default_factory=list)
    approved_comparator: str | None = None
    prompt: str = ""


def dedent_source(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def extract_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(content[start : end + 1])


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = OPENROUTER_URL):
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Add it to .env or the environment.")

    def complete_json(
        self,
        model: str,
        system_prompt: str,
        payload: dict,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> dict:
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/xcelsa_coding_challenge",
                "X-Title": "xcelsa_coding_challenge optimizer",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: HTTP {exc.code}: {error_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        content = response_data["choices"][0]["message"]["content"]
        return extract_json_object(content)


def run_in_subprocess(source: str, function_name: str, cases: list[list], mode: str, repeat: int = 1):
    payload = {
        "source": source,
        "function_name": function_name,
        "cases": cases,
        "mode": mode,
        "repeat": repeat,
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-c", RUNNER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def case_key(case: list) -> str:
    return json.dumps(case, sort_keys=True)


def unique_cases(cases: list[list]) -> list[list]:
    seen = set()
    result = []
    for case in cases:
        key = case_key(case)
        if key not in seen:
            seen.add(key)
            result.append(case)
    return result


def exact_outcomes_match(original: dict, candidate: dict) -> bool:
    return original == candidate


def find_in_sorted_result_is_valid(args: list, outcome: dict) -> bool:
    if outcome["kind"] != "return":
        return False
    sorted_list, target = args
    matching_indices = [index for index, item in enumerate(sorted_list) if item == target]
    if not matching_indices:
        return outcome["value"] == -1
    return outcome["value"] in matching_indices


def outcomes_match(function_name: str, args: list, original: dict, candidate: dict, comparator: str) -> bool:
    if comparator == "find_in_sorted_any_match" and function_name == "find_in_sorted":
        return find_in_sorted_result_is_valid(args, original) and find_in_sorted_result_is_valid(args, candidate)
    return exact_outcomes_match(original, candidate)


class Proposer:
    """Deterministic proposer with prompt-shaped strategy messages."""

    def propose(self, record: dict, attempt: int, feedback: Evaluation | None) -> Proposal | None:
        function_name = record["function_name"]
        strategy = self._candidate_source(function_name, attempt, feedback)
        if strategy is None:
            return None
        name, family, source, assumptions = strategy
        prompt = (
            f"Proposer attempt {attempt}: try a {family} optimization for {function_name}. "
            f"Strategy={name}; assumptions={assumptions}."
        )
        return Proposal(name, family, dedent_source(source), assumptions, prompt)

    def dispute(self, record: dict, proposal: Proposal, evaluation: Evaluation) -> dict | None:
        if record["function_name"] != "find_in_sorted" or not evaluation.failed_cases:
            return None
        for failed_case in evaluation.failed_cases:
            args = failed_case["input"]
            if find_in_sorted_result_is_valid(args, failed_case["original"]) and find_in_sorted_result_is_valid(
                args, failed_case["candidate"]
            ):
                return {
                    "type": "comparator_dispute",
                    "requested_comparator": "find_in_sorted_any_match",
                    "case": args,
                    "prompt": (
                        "Proposer disputes exact-index comparison: the function spec allows any matching "
                        "index when duplicates exist."
                    ),
                }
        return None

    def _candidate_source(self, function_name: str, attempt: int, feedback: Evaluation | None):
        strategies = {
            "has_duplicates": [
                (
                    "duplicate_detection_with_set",
                    "data-structure substitution",
                    """
                    def has_duplicates(items):
                        seen = set()
                        for item in items:
                            if item in seen:
                                return True
                            seen.add(item)
                        return False
                    """,
                    ["items are hashable"],
                )
            ],
            "two_sum": [
                (
                    "two_sum_first_hash_hit",
                    "algorithmic replacement",
                    """
                    def two_sum(nums, target):
                        seen = {}
                        for j, value in enumerate(nums):
                            needed = target - value
                            if needed in seen:
                                return (seen[needed], j)
                            if value not in seen:
                                seen[value] = j
                        return None
                    """,
                    ["hashable numeric values"],
                ),
                (
                    "two_sum_lexicographic_hash_scan",
                    "algorithmic replacement",
                    """
                    def two_sum(nums, target):
                        seen = {}
                        best = None
                        for j, value in enumerate(nums):
                            needed = target - value
                            if needed in seen:
                                candidate = (seen[needed], j)
                                if best is None or candidate < best:
                                    best = candidate
                            if value not in seen:
                                seen[value] = j
                        return best
                    """,
                    ["hashable numeric values", "scan all hits to preserve lexicographic rule"],
                ),
            ],
            "max_subarray_sum": [
                (
                    "kadane_max_subarray",
                    "algorithmic replacement",
                    """
                    def max_subarray_sum(arr):
                        best = arr[0]
                        current = arr[0]
                        for value in arr[1:]:
                            current = max(value, current + value)
                            best = max(best, current)
                        return best
                    """,
                    ["arr is non-empty"],
                )
            ],
            "reverse_string": [
                (
                    "reverse_string_slicing",
                    "programmatic rewrite",
                    """
                    def reverse_string(s):
                        return s[::-1]
                    """,
                    ["s supports slicing"],
                )
            ],
            "fib": [
                (
                    "iterative_fibonacci",
                    "algorithmic replacement",
                    """
                    def fib(n):
                        if n < 2:
                            return n
                        prev = 0
                        current = 1
                        for _ in range(2, n + 1):
                            prev, current = current, prev + current
                        return current
                    """,
                    ["n is a non-negative integer"],
                )
            ],
            "find_in_sorted": [
                (
                    "binary_search_sorted_list",
                    "algorithmic replacement",
                    """
                    def find_in_sorted(sorted_list, target):
                        left = 0
                        right = len(sorted_list) - 1
                        while left <= right:
                            mid = (left + right) // 2
                            value = sorted_list[mid]
                            if value == target:
                                return mid
                            if value < target:
                                left = mid + 1
                            else:
                                right = mid - 1
                        return -1
                    """,
                    ["input list is sorted", "duplicate target may return any matching index"],
                )
            ],
            "sum_of_squares": [
                (
                    "sum_of_squares_builtin_sum",
                    "builtin/library replacement",
                    """
                    def sum_of_squares(nums):
                        return sum(n * n for n in nums)
                    """,
                    ["values support multiplication and addition"],
                )
            ],
            "flatten_one_level": [
                (
                    "flatten_extend",
                    "programmatic rewrite",
                    """
                    def flatten_one_level(list_of_lists):
                        result = []
                        for sub in list_of_lists:
                            result.extend(sub)
                        return result
                    """,
                    ["sublists are iterable"],
                )
            ],
            "count_occurrences": [
                (
                    "count_occurrences_generator_sum",
                    "builtin/library replacement",
                    """
                    def count_occurrences(items, target):
                        return sum(1 for x in items if x == target)
                    """,
                    ["items is iterable"],
                )
            ],
            "group_by_first_letter": [
                (
                    "group_by_first_letter_setdefault",
                    "programmatic rewrite",
                    """
                    def group_by_first_letter(words):
                        result = {}
                        for w in words:
                            if not w:
                                continue
                            result.setdefault(w[0], []).append(w)
                        return result
                    """,
                    ["words are strings"],
                )
            ],
            "safe_average": [
                (
                    "safe_average_builtin_sum",
                    "builtin/library replacement",
                    """
                    def safe_average(numbers):
                        if not numbers:
                            return None
                        return sum(numbers) / len(numbers)
                    """,
                    ["numbers supports truthiness, sum, and len"],
                )
            ],
        }
        function_strategies = strategies.get(function_name, [])
        if attempt - 1 >= len(function_strategies):
            return None
        return function_strategies[attempt - 1]


def evaluation_to_payload(evaluation: Evaluation | None) -> dict | None:
    if evaluation is None:
        return None
    return {
        "equivalent": evaluation.equivalent,
        "faster": evaluation.faster,
        "failed_cases": evaluation.failed_cases,
        "comparator": evaluation.comparator,
        "original_runtime_seconds": evaluation.original_runtime_seconds,
        "optimized_runtime_seconds": evaluation.optimized_runtime_seconds,
        "summary": evaluation.prompt,
    }


class OpenRouterProposer(Proposer):
    def __init__(self, client: OpenRouterClient, model: str):
        self.client = client
        self.model = model

    def propose(self, record: dict, attempt: int, feedback: Evaluation | None) -> Proposal | None:
        system_prompt = (
            "You are the Proposer agent in a Python optimizer. You have full freedom to explore optimization "
            "strategies, including algorithmic changes, compiler-style transformations, data-structure changes, "
            "builtin/library substitutions, memoization, dynamic programming, caching, and Python-specific "
            "programmatic rewrites. Internally compare several strategies before answering, but return only one "
            "best candidate as strict JSON. Preserve the original function name and call signature. If feedback "
            "says the last candidate was not faster, try a substantially different strategy. If feedback contains "
            "equivalence failures, revise the code to address them rather than repeating the same idea."
        )
        payload = {
            "function_name": record["function_name"],
            "original_function": record["function"],
            "attempt": attempt,
            "max_attempts": MAX_ATTEMPTS,
            "feedback": evaluation_to_payload(feedback),
            "strategy_guidance": [
                "Do not restrict yourself to a predefined catalog.",
                "Prefer high-impact algorithmic improvements when the function intent is clear.",
                "Compiler-style or micro optimizations are acceptable if they plausibly improve runtime.",
                "Return the original behavior for all normal cases implied by the function and docstring.",
                "Use assumptions only when they are supported by the function body, name, or docstring.",
            ],
            "required_schema": {
                "type": "candidate",
                "strategy": "short_snake_case_name",
                "family": "algorithmic replacement|compiler-style transformation|builtin/library replacement|data-structure substitution|memoization/dynamic programming|programmatic/micro rewrite|other",
                "optimized_function": "complete Python function source",
                "assumptions": ["short assumption strings"],
                "rationale": "one sentence",
            },
        }
        try:
            response = self.client.complete_json(self.model, system_prompt, payload, max_tokens=2200)
            if response.get("type") != "candidate":
                raise ValueError("Proposer did not return a candidate")
            return Proposal(
                response["strategy"],
                response["family"],
                dedent_source(response["optimized_function"]),
                list(response.get("assumptions", [])),
                f"OpenRouter proposer {self.model} attempt {attempt}: {response.get('rationale', '')}",
            )
        except Exception as exc:
            return Proposal(
                "openrouter_proposer_failed",
                "agent_error",
                record["function"],
                ["OpenRouter proposer did not return a usable candidate"],
                (
                    f"OpenRouter proposer {self.model} failed on attempt {attempt}; "
                    "no deterministic candidate fallback was used. "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ),
            )

    def dispute(self, record: dict, proposal: Proposal, evaluation: Evaluation) -> dict | None:
        if not evaluation.failed_cases:
            return None
        system_prompt = (
            "You are the Proposer agent reviewing evaluator failures. Return strict JSON only. "
            "Only dispute a test/comparator if the function spec clearly allows the candidate output. "
            "Otherwise return {\"type\":\"no_dispute\"}."
        )
        payload = {
            "function_name": record["function_name"],
            "original_function": record["function"],
            "proposal": {
                "strategy": proposal.strategy,
                "optimized_function": proposal.source,
                "assumptions": proposal.assumptions,
            },
            "failed_cases": evaluation.failed_cases,
            "allowed_dispute_schema": {
                "type": "test_dispute",
                "requested_comparator": "string",
                "case": "failed input case",
                "reason": "why the spec allows it",
                "evidence": "why candidate output is valid",
            },
        }
        try:
            response = self.client.complete_json(self.model, system_prompt, payload, max_tokens=1000)
        except Exception:
            return None
        if response.get("type") != "test_dispute":
            return None
        return {
            "type": "comparator_dispute",
            "requested_comparator": response.get("requested_comparator"),
            "case": response.get("case"),
            "prompt": response.get("reason", "OpenRouter proposer disputes evaluator comparison."),
            "evidence": response.get("evidence"),
        }


class Evaluator:
    """Adversarial evaluator that keeps an active in-memory test set."""

    def __init__(self):
        self.active_tests: dict[str, list[list]] = {}

    def tests_for(self, function_name: str) -> list[list]:
        if function_name not in self.active_tests:
            self.active_tests[function_name] = unique_cases(
                SEED_TESTS.get(function_name, []) + ADVERSARIAL_TESTS.get(function_name, [])
            )
        return self.active_tests[function_name]

    def add_approved_failed_cases(self, function_name: str, cases: list[list]) -> None:
        self.active_tests[function_name] = unique_cases(self.tests_for(function_name) + cases)

    def evaluate(self, record: dict, proposal: Proposal, comparator: str = "exact") -> Evaluation:
        function_name = record["function_name"]
        cases = self.tests_for(function_name)
        if proposal.family == "agent_error":
            return Evaluation(
                False,
                failed_cases=[
                    {
                        "input": ["agent_error"],
                        "original": {"kind": "not_run"},
                        "candidate": {"kind": "agent_error", "strategy": proposal.strategy},
                    }
                ],
                comparator=comparator,
                prompt=f"Evaluator rejects {function_name} proposal because proposer returned an agent error.",
            )
        try:
            original_results = run_in_subprocess(record["function"], function_name, cases, "run")
            candidate_results = run_in_subprocess(proposal.source, function_name, cases, "run")
        except Exception as exc:
            return Evaluation(
                False,
                failed_cases=[
                    {
                        "input": ["candidate_load_or_execution"],
                        "original": {"kind": "not_run"},
                        "candidate": {"kind": "exception", "type": type(exc).__name__, "message": str(exc)[:200]},
                    }
                ],
                comparator=comparator,
                prompt=f"Evaluator could not execute candidate for {function_name}: {type(exc).__name__}.",
            )
        failed_cases = []

        for args, original, candidate in zip(cases, original_results, candidate_results):
            if not outcomes_match(function_name, args, original, candidate, comparator):
                failed_cases.append({"input": args, "original": original, "candidate": candidate})

        prompt = (
            f"Evaluator checked {len(cases)} cases for {function_name} using comparator={comparator}. "
            f"Failures={len(failed_cases)}."
        )
        if failed_cases:
            return Evaluation(False, failed_cases=failed_cases, comparator=comparator, prompt=prompt)

        original_time = self.benchmark(record["function"], function_name)
        candidate_time = self.benchmark(proposal.source, function_name)
        return Evaluation(
            True,
            faster=candidate_time < original_time,
            original_runtime_seconds=original_time,
            optimized_runtime_seconds=candidate_time,
            comparator=comparator,
            prompt=prompt,
        )

    def benchmark(self, source: str, function_name: str) -> float:
        cases = BENCHMARK_CASES.get(function_name, self.tests_for(function_name))
        return run_in_subprocess(source, function_name, cases, "benchmark", repeat=20)["elapsed"]


class OpenRouterEvaluator(Evaluator):
    def __init__(self, client: OpenRouterClient, model: str, max_extra_tests: int = 5):
        super().__init__()
        self.client = client
        self.model = model
        self.max_extra_tests = max_extra_tests
        self._augmented_keys: set[str] = set()

    def evaluate(self, record: dict, proposal: Proposal, comparator: str = "exact") -> Evaluation:
        self._augment_tests(record, proposal, comparator)
        evaluation = super().evaluate(record, proposal, comparator=comparator)
        evaluation.prompt = f"OpenRouter evaluator {self.model}: {evaluation.prompt}"
        return evaluation

    def _augment_tests(self, record: dict, proposal: Proposal, comparator: str) -> None:
        function_name = record["function_name"]
        key = f"{function_name}:{proposal.strategy}:{comparator}"
        if key in self._augmented_keys:
            return
        self._augmented_keys.add(key)

        system_prompt = (
            "You are the Evaluator/Disputer agent for a Python optimizer. Internally search for edge cases "
            "that could break equivalence. Return strict JSON only. Do not decide acceptance. Generate "
            "concrete positional-argument test cases as JSON arrays. Prefer small cases that target the "
            "candidate assumptions."
        )
        payload = {
            "function_name": function_name,
            "original_function": record["function"],
            "candidate": {
                "strategy": proposal.strategy,
                "optimized_function": proposal.source,
                "assumptions": proposal.assumptions,
            },
            "existing_tests": self.tests_for(function_name),
            "comparator": comparator,
            "required_schema": {
                "extra_tests": [
                    {
                        "args": ["positional argument list"],
                        "reason": "what this test targets",
                    }
                ]
            },
            "max_extra_tests": self.max_extra_tests,
        }
        try:
            response = self.client.complete_json(self.model, system_prompt, payload, max_tokens=1500)
        except Exception:
            return
        extra_tests = []
        for item in response.get("extra_tests", [])[: self.max_extra_tests]:
            args = item.get("args")
            if isinstance(args, list):
                extra_tests.append(args)
        if extra_tests:
            self.active_tests[function_name] = unique_cases(self.tests_for(function_name) + extra_tests)


class Judge:
    """Conservative auditor for evaluator/proposer transcripts."""

    def review_failed_cases(self, evaluation: Evaluation) -> JudgeDecision:
        if not evaluation.failed_cases:
            return JudgeDecision(False, "no_failed_cases")
        return JudgeDecision(
            True,
            "failed_cases_are_valid_regressions",
            approved_failed_cases=[case["input"] for case in evaluation.failed_cases],
            prompt="Judge approves evaluator's failed equivalence cases as regression tests.",
        )

    def review_dispute(self, record: dict, dispute: dict | None) -> JudgeDecision:
        if not dispute:
            return JudgeDecision(False, "no_dispute")
        if (
            record["function_name"] == "find_in_sorted"
            and dispute["type"] == "comparator_dispute"
            and dispute.get("requested_comparator") == "find_in_sorted_any_match"
            and isinstance(dispute.get("case"), list)
        ):
            return JudgeDecision(
                True,
                "spec_allows_any_duplicate_match_index",
                approved_failed_cases=[dispute["case"]],
                approved_comparator=dispute["requested_comparator"],
                prompt="Judge accepts comparator dispute based on the docstring duplicate-index rule.",
            )
        return JudgeDecision(False, "unsupported_dispute", prompt="Judge rejects unsupported dispute.")

    def decide(self, evaluation: Evaluation) -> JudgeDecision:
        if not evaluation.equivalent:
            return JudgeDecision(False, "not_equivalent", prompt="Judge rejects candidate due to failed cases.")
        if not evaluation.faster:
            return JudgeDecision(False, "not_faster", prompt="Judge rejects candidate due to benchmark result.")
        return JudgeDecision(True, "equivalent_and_faster", prompt="Judge accepts candidate.")


class OpenRouterJudge(Judge):
    def __init__(self, client: OpenRouterClient, model: str):
        self.client = client
        self.model = model

    def review_dispute(self, record: dict, dispute: dict | None) -> JudgeDecision:
        decision = super().review_dispute(record, dispute)
        audit = self._audit(
            {
                "task": "review_dispute",
                "function_name": record["function_name"],
                "dispute": dispute,
                "local_decision": {
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "approved_comparator": decision.approved_comparator,
                },
            }
        )
        if audit:
            decision.prompt = f"{decision.prompt or decision.reason} OpenRouter judge audit: {audit}"
        return decision

    def decide(self, evaluation: Evaluation) -> JudgeDecision:
        decision = super().decide(evaluation)
        audit = self._audit(
            {
                "task": "final_candidate_decision",
                "evaluation": evaluation_to_payload(evaluation),
                "local_decision": {"accepted": decision.accepted, "reason": decision.reason},
            }
        )
        if audit:
            decision.prompt = f"{decision.prompt} OpenRouter judge audit: {audit}"
        return decision

    def _audit(self, payload: dict) -> str | None:
        system_prompt = (
            "You are the Judge agent auditing a Python optimizer decision. Return strict JSON only with "
            "{\"audit\":\"one concise sentence\"}. Do not invent new acceptance criteria; the local safety "
            "policy remains authoritative."
        )
        try:
            response = self.client.complete_json(self.model, system_prompt, payload, max_tokens=400, temperature=0)
        except Exception:
            return None
        audit = response.get("audit")
        return audit if isinstance(audit, str) else None


def transcript_event(role: str, message: str, **fields) -> dict:
    return {"role": role, "message": message, **fields}


def optimize_record(
    record: dict,
    max_attempts: int = MAX_ATTEMPTS,
    emit_messages: bool = True,
    proposer: Proposer | None = None,
    evaluator: Evaluator | None = None,
    judge: Judge | None = None,
) -> dict:
    proposer = proposer or Proposer()
    evaluator = evaluator or Evaluator()
    judge = judge or Judge()
    transcript = []
    feedback = None
    comparator = "exact"
    best_rejection = None

    for attempt in range(1, max_attempts + 1):
        proposal = proposer.propose(record, attempt, feedback)
        if proposal is None:
            transcript.append(transcript_event("proposer", f"No proposal available for attempt {attempt}."))
            break

        transcript.append(
            transcript_event(
                "proposer",
                proposal.prompt,
                attempt=attempt,
                strategy=proposal.strategy,
                family=proposal.family,
                assumptions=proposal.assumptions,
            )
        )
        evaluation = evaluator.evaluate(record, proposal, comparator=comparator)
        transcript.append(
            transcript_event(
                "evaluator",
                evaluation.prompt,
                attempt=attempt,
                equivalent=evaluation.equivalent,
                failed_cases=evaluation.failed_cases,
                original_runtime_seconds=evaluation.original_runtime_seconds,
                optimized_runtime_seconds=evaluation.optimized_runtime_seconds,
            )
        )

        if not evaluation.equivalent:
            dispute = proposer.dispute(record, proposal, evaluation)
            if dispute:
                transcript.append(transcript_event("proposer", dispute["prompt"], dispute=dispute))
            dispute_decision = judge.review_dispute(record, dispute)
            transcript.append(
                transcript_event(
                    "judge",
                    dispute_decision.prompt or dispute_decision.reason,
                    accepted=dispute_decision.accepted,
                    reason=dispute_decision.reason,
                )
            )
            if dispute_decision.accepted and dispute_decision.approved_comparator:
                evaluator.add_approved_failed_cases(record["function_name"], dispute_decision.approved_failed_cases)
                comparator = dispute_decision.approved_comparator
                evaluation = evaluator.evaluate(record, proposal, comparator=comparator)
                transcript.append(
                    transcript_event(
                        "evaluator",
                        evaluation.prompt,
                        attempt=attempt,
                        comparator=comparator,
                        equivalent=evaluation.equivalent,
                        failed_cases=evaluation.failed_cases,
                        original_runtime_seconds=evaluation.original_runtime_seconds,
                        optimized_runtime_seconds=evaluation.optimized_runtime_seconds,
                    )
                )
            elif evaluation.failed_cases:
                failed_case_decision = judge.review_failed_cases(evaluation)
                transcript.append(
                    transcript_event(
                        "judge",
                        failed_case_decision.prompt,
                        accepted=failed_case_decision.accepted,
                        reason=failed_case_decision.reason,
                        approved_failed_cases=failed_case_decision.approved_failed_cases,
                    )
                )
                if failed_case_decision.accepted:
                    evaluator.add_approved_failed_cases(
                        record["function_name"], failed_case_decision.approved_failed_cases
                    )
            feedback = evaluation

        decision = judge.decide(evaluation)
        transcript.append(
            transcript_event(
                "judge",
                decision.prompt,
                accepted=decision.accepted,
                reason=decision.reason,
            )
        )
        if decision.accepted:
            return {
                **record,
                "optimized_function": proposal.source,
                "accepted": True,
                "selected_strategy": proposal.strategy,
                "original_runtime_seconds": evaluation.original_runtime_seconds,
                "optimized_runtime_seconds": evaluation.optimized_runtime_seconds,
                "orchestration_transcript": transcript,
            }
        best_rejection = decision.reason
        feedback = evaluation

    if emit_messages:
        print(
            f"Returning the original function for {record['function_name']}: "
            "no equivalent faster strategy accepted."
        )
    return {
        **record,
        "optimized_function": record["function"],
        "accepted": False,
        "selected_strategy": None,
        "rejection_reason": best_rejection or "no_strategy",
        "orchestration_transcript": transcript,
    }


def build_agents(
    use_openrouter: bool,
    proposer_model: str,
    evaluator_model: str,
    judge_model: str | None,
) -> tuple[Proposer, Evaluator, Judge]:
    if not use_openrouter:
        return Proposer(), Evaluator(), Judge()
    client = OpenRouterClient()
    proposer = OpenRouterProposer(client, proposer_model)
    evaluator = OpenRouterEvaluator(client, evaluator_model)
    judge = OpenRouterJudge(client, judge_model) if judge_model else Judge()
    return proposer, evaluator, judge


def summarize_orchestration(result: dict) -> dict:
    transcript = result.get("orchestration_transcript", [])
    attempts = sorted(
        {
            event["attempt"]
            for event in transcript
            if isinstance(event.get("attempt"), int)
        }
    )
    judge_verdicts = [
        event.get("reason", "unknown")
        for event in transcript
        if event.get("role") == "judge"
    ]
    verdict_counts = {}
    for verdict in judge_verdicts:
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    return {
        "overall_iterations": len(attempts),
        "judge_verdicts": verdict_counts,
        "final_status": "accepted" if result.get("accepted") else result.get("rejection_reason", "rejected"),
    }


def format_orchestration_summary(summary: dict) -> str:
    verdicts = summary["judge_verdicts"]
    if verdicts:
        verdict_text = ", ".join(f"{reason}={count}" for reason, count in verdicts.items())
    else:
        verdict_text = "none"
    return (
        f"iterations={summary['overall_iterations']}; "
        f"judge_verdicts=[{verdict_text}]; "
        f"final={summary['final_status']}"
    )


def optimize_records(
    records: list[dict],
    max_attempts: int = MAX_ATTEMPTS,
    emit_messages: bool = True,
    use_openrouter: bool = False,
    proposer_model: str = DEFAULT_PROPOSER_MODEL,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> list[dict]:
    results = []
    for record in records:
        print(f"Optimizing {record['function_name']} (id={record['id']})...")
        proposer, evaluator, judge = build_agents(
            use_openrouter,
            proposer_model=proposer_model,
            evaluator_model=evaluator_model,
            judge_model=judge_model,
        )
        result = optimize_record(
            record,
            max_attempts=max_attempts,
            emit_messages=emit_messages,
            proposer=proposer,
            evaluator=evaluator,
            judge=judge,
        )
        result["orchestration_summary"] = summarize_orchestration(result)
        if emit_messages:
            print(
                f"Summary for {record['function_name']}: "
                f"{format_orchestration_summary(result['orchestration_summary'])}"
            )
        results.append(result)
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run deterministic proposer/evaluator/judge optimization.")
    parser.add_argument("input_path", nargs="?", default="function_inputs.json")
    parser.add_argument("output_path", nargs="?", default="agent_optimized_function_inputs.json")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter-backed proposer/evaluator agents with local sandbox verification.",
    )
    parser.add_argument(
        "--proposer-model",
        default=os.environ.get("OPENROUTER_PROPOSER_MODEL", DEFAULT_PROPOSER_MODEL),
        help="OpenRouter model for the proposer agent.",
    )
    parser.add_argument(
        "--evaluator-model",
        default=os.environ.get("OPENROUTER_EVALUATOR_MODEL", DEFAULT_EVALUATOR_MODEL),
        help="OpenRouter model for evaluator/adversarial test generation.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="Optional OpenRouter model for judge audit notes. Leave blank for local deterministic judge.",
    )
    args = parser.parse_args()

    records = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    optimized = optimize_records(
        records,
        max_attempts=args.max_attempts,
        use_openrouter=args.use_openrouter,
        proposer_model=args.proposer_model,
        evaluator_model=args.evaluator_model,
        judge_model=args.judge_model or None,
    )
    Path(args.output_path).write_text(json.dumps(optimized, indent=2) + "\n", encoding="utf-8")
    accepted_count = sum(1 for item in optimized if item["accepted"])
    print(f"Accepted {accepted_count} of {len(optimized)} optimization candidates.")
    print(f"Wrote results to {args.output_path}")


if __name__ == "__main__":
    main()
