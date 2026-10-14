"""Capture tokenizer-only fixtures from pinned source, without importing Torch/model."""

import argparse
import ast
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import re


def capture(package, tokenizer_dir):
    from tokenizers import Tokenizer

    manifest = json.loads(
        (Path(__file__).parent.parent / "backend/model-manifest.json").read_text()
    )
    for name in ("cadence_kev/api.py", "cadence_kev/model.py"):
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError("Source differs from the pinned public package: " + name)

    # api imports no model; model encoder AST subset retains exactly published
    # functions/constants without importing its heavyweight inference dependencies.
    spec = importlib.util.spec_from_file_location(
        "cadence_tokenizer_api", package / "cadence_kev/api.py"
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    names = {"SPECIAL", "_SPECIAL_RE", "OPT_NONE", "OPT_DECIDE"}
    tree = ast.parse((package / "cadence_kev/model.py").read_text())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef) and node.name in {"user_tokens", "encode"}
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in ast.walk(node)
            )
        )
    ]
    environment = {
        "re": re,
        "ContextOverflow": ValueError,
        "MAX_STATE": 8192,
        "MAX_BRANCH": 8192,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), "pinned-cadence-encoder", "exec"
        ),
        environment,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    class Adapter:
        def convert_tokens_to_ids(self, text):
            return tokenizer.token_to_id(text)

        def __call__(self, text, add_special_tokens=False):
            return type(
                "Encoding",
                (),
                {
                    "input_ids": tokenizer.encode(
                        text, add_special_tokens=add_special_tokens
                    ).ids
                },
            )()

    states = [
        {
            "nested": {"yes": True, "no": False, "nil": None},
            "items": ["x", {"n": 2}, [3, 4], []],
        },
        {"0": "zero", "2": "two", "11": "eleven", "a": "A", "01": "leading"},
        ["\u001chello", "\ufeffhello", "\u0085a", "\u2003b"],
        {
            "数": "你好，世界！",
            "emoji": "☕ 🧑🏽‍💻 🚀",
            "accent": "café e\u0301",
            "ar": "مرحبا",
        },
        "\r\n\tA\u0085B\u2028C\u2029D\u001cE\ufeffF",
        "<|fim_prefix|><|fim_middle|><|box_start|><|box_end|><|fim_suffix|><|im_start|><|A_12|>",
        {"n": [0.00001, 0.0001, -1.23e-7, 5e-324, 1000000000000000, 9007199254740991]},
        {"empty": {}, "array": [], "null": None},
    ]
    rng = random.Random(1729)
    for _ in range(100):
        states.append({"n": rng.uniform(-9, 9) * 10 ** rng.randint(-320, 5)})
    cases = []
    for index, state in enumerate(states):
        request = {
            "state": state,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose <|fim_suffix|> 请回答。",
                    "criteria": {
                        "11": "eleven",
                        "0": None,
                        "other": "café ☕ <|box_end|>",
                    },
                }
            },
        }
        # Same canonical key ordering as JSON.stringify browser requests, including
        # its numeric-index keys. Python inputs above deliberately already ordered.
        request["questions"]["q"]["criteria"] = {
            "0": None,
            "11": "eleven",
            "other": "café ☕ <|box_end|>",
        }
        record, _ = api.to_record(api.SystemOneRequest.model_validate(request))
        cases.append(
            {
                "name": f"tokenizer-edge-{index}",
                "request": request,
                "nativeRecord": record,
                "encoding": environment["encode"](
                    Adapter(), record, max_state=8192, max_branch=8192, strict=True
                ),
            }
        )
    return {
        "modelInferenceRun": False,
        "source": "Pinned Cadence API and encoder, Rust tokenizers",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.package, args.tokenizer_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "modelInferenceRun": False}))


if __name__ == "__main__":
    main()
