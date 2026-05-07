import unittest

from optimizer import equivalent_on_tests, find_in_sorted_result_is_valid, optimize_record, optimize_records


class OptimizerTests(unittest.TestCase):
    def test_reverse_string_rule_is_accepted(self):
        record = {
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

        result = optimize_record(record, emit_messages=False)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_rule"], "reverse_string_slicing")
        self.assertIn("return s[::-1]", result["optimized_function"])

    def test_bad_candidate_fails_equivalence_bench(self):
        original = """
def reverse_string(s):
   result = ""
   for ch in s:
       result = ch + result
   return result
"""
        candidate = """
def reverse_string(s):
   return s
"""

        self.assertFalse(equivalent_on_tests(original, candidate, "reverse_string"))

    def test_unknown_function_returns_original(self):
        record = {
            "id": 99,
            "function_name": "already_good",
            "function": """
def already_good(nums):
    return sum(nums)
""",
        }

        result = optimize_record(record, emit_messages=False)

        self.assertFalse(result["accepted"])
        self.assertIsNone(result["selected_rule"])
        self.assertEqual(result["optimized_function"], record["function"])

    def test_find_in_sorted_accepts_any_duplicate_match_index(self):
        args = [[1, 2, 2, 2, 3], 2]

        self.assertTrue(find_in_sorted_result_is_valid(args, 1))
        self.assertTrue(find_in_sorted_result_is_valid(args, 2))
        self.assertTrue(find_in_sorted_result_is_valid(args, 3))
        self.assertFalse(find_in_sorted_result_is_valid(args, 0))
        self.assertFalse(find_in_sorted_result_is_valid(args, -1))

    def test_optimizer_handles_multiple_records(self):
        records = [
            {
                "id": 4,
                "function_name": "reverse_string",
                "function": """
def reverse_string(s):
   result = ""
   for ch in s:
       result = ch + result
   return result
""",
            },
            {
                "id": 99,
                "function_name": "already_good",
                "function": """
def already_good(nums):
    return sum(nums)
""",
            },
        ]

        results = optimize_records(records, emit_messages=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["accepted"])
        self.assertFalse(results[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
