import json
from pathlib import Path

from evaluation.judge import JUDGE_PROMPT, RESPONSE_SCHEMA, average, load_traces


def test_traces_file_has_ten_incidents():
    traces = load_traces(Path("evaluation/traces.json"))
    assert len(traces) == 10
    assert all(item.get("incident_id") and item.get("trace") for item in traces)
    # The assignment's error-recovery cases: a 400 then a corrected call, a failed rollback, a rejected restart, and two failures.
    recovered = [item for item in traces if any((step.get("result") or {}).get("status") == 400 for step in item["trace"] if "result" in step)]
    assert len(recovered) >= 6


def test_average_error_recovery_score():
    assert average([{"score": 5}, {"score": 1}, {"score": 4}]) == 3.33


def test_judge_prompt_is_error_recovery_only():
    assert "Evaluate only ERROR RECOVERY" in JUDGE_PROMPT
    assert "Score from 1 to 5" in JUDGE_PROMPT
    assert RESPONSE_SCHEMA["properties"]["score"]["minimum"] == 1
    assert RESPONSE_SCHEMA["required"] == ["score", "explanation", "evidence"]


def test_load_traces_rejects_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"traces": []}), encoding="utf-8")
    try:
        load_traces(path)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
