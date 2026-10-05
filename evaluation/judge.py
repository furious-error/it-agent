"""LLM-as-judge for Error Recovery only.

    python -m evaluation.judge
    python -m evaluation.judge --traces evaluation/traces.json --out evaluation/scores.json

Scores one historical trace at a time with a separate Gemini call (not the
investigator). The rubric is the assignment's 1-5 Error Recovery scale.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import ConfigError, load_settings
from app.telemetry import EVALUATE_ERROR_RECOVERY, flush, init_tracing, observation

DEFAULT_TRACES = Path(__file__).resolve().parent / "traces.json"

JUDGE_PROMPT = """\
You are evaluating an AI SRE agent.

Evaluate only ERROR RECOVERY.

Score from 1 to 5.

1 = Agent failed to recognize or recover from the error.

2 = Agent recognized the error but responded poorly.

3 = Agent recovered partially with unnecessary actions.

4 = Agent correctly interpreted the error and recovered.

5 = Agent efficiently diagnosed the error, corrected its behavior,
and continued safely.

Provide:
- score
- explanation
- evidence from the trace
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "explanation", "evidence"],
}


def load_traces(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = payload.get("traces")
    if not isinstance(traces, list) or len(traces) < 1:
        raise ValueError(f"{path} must contain a non-empty 'traces' list")
    return traces


def grade_trace(client: genai.Client, model: str, item: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(
        {"incident_id": item.get("incident_id"), "scenario": item.get("scenario"), "trace": item.get("trace")},
        indent=2,
    )
    with observation(
        as_type="evaluator",
        name=EVALUATE_ERROR_RECOVERY,
        model=model,
        input={"incident_id": item.get("incident_id"), "scenario": item.get("scenario")},
        metadata={"metric": "error_recovery"},
    ) as obs:
        response = _generate_with_retry(
            client,
            model,
            body,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        graded = json.loads(response.text or "{}")
        score = int(graded["score"])
        if score < 1 or score > 5:
            raise ValueError(f"judge returned score {score} outside 1-5")
        result = {
            "incident_id": item.get("incident_id"),
            "scenario": item.get("scenario"),
            "score": score,
            "explanation": graded.get("explanation", ""),
            "evidence": graded.get("evidence") or [],
        }
        obs.update(output=result)
        return result


def _retry_wait(exc: APIError, attempt: int) -> float:
    details = getattr(exc, "details", None) or {}
    error = details.get("error") if isinstance(details, dict) else None
    for item in (error or {}).get("details") or []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(1.0, float(delay[:-1]))
            except ValueError:
                break
    if getattr(exc, "code", None) == 429:
        return 35.0
    return float(2 ** attempt)


def _generate_with_retry(client: genai.Client, model: str, body: str, config: types.GenerateContentConfig, attempts: int = 6):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.models.generate_content(model=model, contents=body, config=config)
        except APIError as exc:
            last = exc
            if getattr(exc, "code", None) not in {429, 503} or attempt == attempts - 1:
                break
            time.sleep(_retry_wait(exc, attempt))
    assert last is not None
    raise last


def average(scores: list[dict[str, Any]]) -> float:
    return round(statistics.fmean(item["score"] for item in scores), 2)


def run(traces_path: Path, out_path: Path | None) -> dict[str, Any]:
    settings = load_settings()
    init_tracing()
    client = genai.Client(api_key=settings.api_key)
    traces = load_traces(traces_path)
    graded = []
    for index, item in enumerate(traces):
        if index:
            time.sleep(13)
        graded.append(grade_trace(client, settings.model, item))
    report = {
        "metric": "error_recovery",
        "scale": "1-5",
        "count": len(graded),
        "average_error_recovery_score": average(graded),
        "scores": graded,
    }
    flush()
    if out_path:
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade historical traces on Error Recovery.")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--out", type=Path, default=Path("evaluation/scores.json"))
    args = parser.parse_args(argv)
    try:
        report = run(args.traces, args.out)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Judge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Average Error Recovery Score: {report['average_error_recovery_score']} ({report['count']} traces)")
    for item in report["scores"]:
        print(f"  {item['score']}  {item['incident_id']}  {item['scenario']}")
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
