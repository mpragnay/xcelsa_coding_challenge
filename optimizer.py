import argparse
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


TEST_CASES = {
    "has_duplicates": [
        [[1, 2, 1]],
        [[1, 2, 3]],
        [[]],
    ],
    "two_sum": [
        [[4, 6, 10], 16],
        [[1, 2, 3, 4], 8],
        [[3, 2, 4, 1, 5], 6],
    ],
    "max_subarray_sum": [
        [[-2, 3, -1, 4]],
        [[-5]],
        [[-4, -2, -7]],
    ],
    "reverse_string": [
        ["abc"],
        [""],
        ["racecar"],
    ],
    "fib": [
        [0],
        [1],
        [6],
        [10],
    ],
    "find_in_sorted": [
        [[1, 3, 5, 7], 5],
        [[1, 3, 5, 7], 2],
        [[1, 2, 2, 2, 3], 2],
    ],
    "sum_of_squares": [
        [[1, 2, 3]],
        [[]],
        [[-2, 3]],
    ],
    "flatten_one_level": [
        [[[1, 2], [3], []]],
        [[]],
        [[[], ["a"], ["b", "c"]]],
    ],
    "count_occurrences": [
        [["a", "b", "a"], "a"],
        [[1, 2, 3], 4],
        [[], 1],
    ],
    "group_by_first_letter": [
        [["ant", "", "ape", "bat"]],
        [[]],
        [["bob", "alice", "brad"]],
    ],
    "sort_by_priority": [
        [[{"priority": 2}, {"priority": 1}]],
        [[{"priority": 1}, {"priority": 1}, {"priority": 0}]],
    ],
    "safe_average": [
        [[2, 4, 6]],
        [[]],
        [[-1, 1]],
    ],
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
    "sort_by_priority": [[list({"priority": i % 10} for i in range(5000, 0, -1))]],
    "safe_average": [[[i for i in range(10000)]]],
}


@dataclass(frozen=True)
class OptimizationRule:
    name: str
    function_name: str
    replacement_template: str

    def apply(self, record: dict) -> str | None:
        if record["function_name"] != self.function_name:
            return None
        return textwrap.dedent(self.replacement_template).strip() + "\n"


RULES = [
    OptimizationRule(
        "duplicate_detection_with_set",
        "has_duplicates",
        """
        def has_duplicates(items):
            seen = set()
            for item in items:
                if item in seen:
                    return True
                seen.add(item)
            return False
        """,
    ),
    OptimizationRule(
        "two_sum_hash_scan",
        "two_sum",
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
    ),
    OptimizationRule(
        "kadane_max_subarray",
        "max_subarray_sum",
        """
        def max_subarray_sum(arr):
            best = arr[0]
            current = arr[0]
            for value in arr[1:]:
                current = max(value, current + value)
                best = max(best, current)
            return best
        """,
    ),
    OptimizationRule(
        "reverse_string_slicing",
        "reverse_string",
        """
        def reverse_string(s):
            return s[::-1]
        """,
    ),
    OptimizationRule(
        "iterative_fibonacci",
        "fib",
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
    ),
    OptimizationRule(
        "binary_search_sorted_list",
        "find_in_sorted",
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
    ),
    OptimizationRule(
        "sum_of_squares_builtin_sum",
        "sum_of_squares",
        """
        def sum_of_squares(nums):
            return sum(n * n for n in nums)
        """,
    ),
    OptimizationRule(
        "flatten_extend",
        "flatten_one_level",
        """
        def flatten_one_level(list_of_lists):
            result = []
            for sub in list_of_lists:
                result.extend(sub)
            return result
        """,
    ),
    OptimizationRule(
        "count_occurrences_generator_sum",
        "count_occurrences",
        """
        def count_occurrences(items, target):
            return sum(1 for x in items if x == target)
        """,
    ),
    OptimizationRule(
        "group_by_first_letter_setdefault",
        "group_by_first_letter",
        """
        def group_by_first_letter(words):
            result = {}
            for w in words:
                if not w:
                    continue
                result.setdefault(w[0], []).append(w)
            return result
        """,
    ),
    OptimizationRule(
        "safe_average_builtin_sum",
        "safe_average",
        """
        def safe_average(numbers):
            if not numbers:
                return None
            return sum(numbers) / len(numbers)
        """,
    ),
]


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
exec(compile(payload["source"], "<candidate>", "exec"), namespace)
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
    cases = payload["cases"]
    repeat = payload["repeat"]
    start = time.perf_counter()
    for _ in range(repeat):
        for args in cases:
            func(*args)
    elapsed = time.perf_counter() - start
    print(json.dumps({"elapsed": elapsed}))
"""


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


def equivalent_on_tests(original_source: str, candidate_source: str, function_name: str) -> bool:
    cases = TEST_CASES.get(function_name, [])
    if not cases:
        return False
    original_results = run_in_subprocess(original_source, function_name, cases, "run")
    candidate_results = run_in_subprocess(candidate_source, function_name, cases, "run")
    return all(
        outcomes_match(function_name, args, original, candidate)
        for args, original, candidate in zip(cases, original_results, candidate_results)
    )


def outcomes_match(function_name: str, args: list, original: dict, candidate: dict) -> bool:
    if original["kind"] != candidate["kind"]:
        return False
    if original["kind"] == "exception":
        return original == candidate
    if function_name == "find_in_sorted":
        return find_in_sorted_result_is_valid(args, original["value"]) and find_in_sorted_result_is_valid(
            args, candidate["value"]
        )
    return original == candidate


def find_in_sorted_result_is_valid(args: list, value) -> bool:
    sorted_list, target = args
    matching_indices = [index for index, item in enumerate(sorted_list) if item == target]
    if not matching_indices:
        return value == -1
    return value in matching_indices


def benchmark(source: str, function_name: str) -> float:
    cases = BENCHMARK_CASES.get(function_name, TEST_CASES.get(function_name, []))
    return run_in_subprocess(source, function_name, cases, "benchmark", repeat=20)["elapsed"]


def optimize_record(record: dict, emit_messages: bool = True) -> dict:
    original_source = record["function"]
    function_name = record["function_name"]

    for rule in RULES:
        candidate_source = rule.apply(record)
        if candidate_source is None:
            continue

        if not equivalent_on_tests(original_source, candidate_source, function_name):
            continue

        original_time = benchmark(original_source, function_name)
        candidate_time = benchmark(candidate_source, function_name)
        if candidate_time < original_time:
            return {
                **record,
                "optimized_function": candidate_source,
                "selected_rule": rule.name,
                "original_runtime_seconds": original_time,
                "optimized_runtime_seconds": candidate_time,
                "accepted": True,
            }

    if emit_messages:
        print(f"Returning the original function for {function_name}: no equivalent faster rule accepted.")
    return {
        **record,
        "optimized_function": original_source,
        "selected_rule": None,
        "accepted": False,
    }


def optimize_records(records: list[dict], emit_messages: bool = True) -> list[dict]:
    return [optimize_record(record, emit_messages=emit_messages) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply simple rule-based Python optimizations.")
    parser.add_argument("input_path", nargs="?", default="function_inputs.json")
    parser.add_argument("output_path", nargs="?", default="optimized_function_inputs.json")
    args = parser.parse_args()

    records = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    optimized = optimize_records(records)
    Path(args.output_path).write_text(json.dumps(optimized, indent=2) + "\n", encoding="utf-8")
    accepted_count = sum(1 for record in optimized if record["accepted"])
    print(f"Accepted {accepted_count} of {len(optimized)} optimization candidates.")
    print(f"Wrote results to {args.output_path}")


if __name__ == "__main__":
    main()
