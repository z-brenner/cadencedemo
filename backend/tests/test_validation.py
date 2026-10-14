import json
from pathlib import Path
import pytest
from backend.validation import compact_json_length

CASES = json.loads(
    (Path(__file__).parents[2] / "tests/fixtures/compact-json-lengths.json").read_text()
)


@pytest.mark.parametrize("case", CASES)
def test_json_budget_matches_native_javascript_fixtures(case):
    assert compact_json_length(json.loads(case["source"])) == case["units"]


@pytest.mark.parametrize(
    "source", ["[" * 500 + "0" + "]" * 500, '{"x":' * 500 + "0" + "}" * 500]
)
def test_deep_json_budget_does_not_add_recursive_frames(source):
    assert compact_json_length(json.loads(source)) == len(source)
