import argparse
import ast
import json
from pathlib import Path


def parse_python_functions(source: str) -> list[dict]:
    """Extract top-level Python functions from source text into JSON-ready records."""
    source = source.lstrip("\ufeff")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Dataset is not valid Python: {exc.msg}") from exc

    records = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        function_source = ast.get_source_segment(source, node)
        if function_source is None:
            raise ValueError(f"Could not extract source for function {node.name}")

        records.append(
            {
                "id": len(records) + 1,
                "function_name": node.name,
                "function": function_source,
            }
        )

    if not records:
        raise ValueError("Dataset does not contain any top-level functions")
    return records


def parse_dataset_file(input_path: str | Path) -> list[dict]:
    source = Path(input_path).read_text(encoding="utf-8-sig")
    return parse_python_functions(source)


def write_dataset_json(input_path: str | Path, output_path: str | Path) -> list[dict]:
    records = parse_dataset_file(input_path)
    Path(output_path).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Python function dataset to JSON.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="Python Optimization Dataset.txt",
        help="Path to the text file containing Python functions.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="function_inputs.json",
        help="Path where the JSON output should be written.",
    )
    args = parser.parse_args()

    records = write_dataset_json(args.input_path, args.output_path)
    print(f"Wrote {len(records)} functions to {args.output_path}")


if __name__ == "__main__":
    main()
