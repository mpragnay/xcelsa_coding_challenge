import json
import unittest

from dataset_parser import parse_dataset_file


DATASET_PATH = "Python Optimization Dataset.txt"

SMOKE_CASES = {
    "has_duplicates": (([1, 2, 1],), True),
    "two_sum": (([4, 6, 10], 16), (1, 2)),
    "max_subarray_sum": (([-2, 3, -1, 4],), 6),
    "reverse_string": (("abc",), "cba"),
    "fib": ((6,), 8),
    "find_in_sorted": (([1, 3, 5, 7], 5), 2),
    "sum_of_squares": (([1, 2, 3],), 14),
    "flatten_one_level": (([[1, 2], [3], []],), [1, 2, 3]),
    "count_occurrences": ((["a", "b", "a"], "a"), 2),
    "group_by_first_letter": ((["ant", "", "ape", "bat"],), {"a": ["ant", "ape"], "b": ["bat"]}),
    "sort_by_priority": (([{"priority": 2}, {"priority": 1}],), [{"priority": 1}, {"priority": 2}]),
    "safe_average": (([2, 4, 6],), 4),
}


def load_function(source: str, function_name: str):
    namespace = {"__builtins__": __builtins__}
    exec(compile(source, f"<{function_name}>", "exec"), namespace)
    return namespace[function_name]


class DatasetParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = parse_dataset_file(DATASET_PATH)
        cls.functions = {
            record["function_name"]: load_function(record["function"], record["function_name"])
            for record in cls.records
        }

    def test_dataset_is_converted_to_json_ready_records(self):
        self.assertEqual(len(self.records), 12)
        self.assertEqual(self.records[0]["function_name"], "has_duplicates")
        self.assertEqual(self.records[-1]["function_name"], "safe_average")

        encoded = json.dumps(self.records)
        decoded = json.loads(encoded)
        self.assertEqual(decoded, self.records)


def make_smoke_test(function_name: str, args: tuple, expected):
    def test(self):
        self.assertEqual(self.functions[function_name](*args), expected)

    test.__name__ = f"test_{function_name}_smoke_case"
    return test


for smoke_function_name, (smoke_args, smoke_expected) in SMOKE_CASES.items():
    setattr(
        DatasetParserTests,
        f"test_{smoke_function_name}_smoke_case",
        make_smoke_test(smoke_function_name, smoke_args, smoke_expected),
    )


if __name__ == "__main__":
    unittest.main()
