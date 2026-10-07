"""HTTP API for one investigation per request.

    uvicorn app.api:app --host 0.0.0.0 --port 8000

POST /webhook starts a run and returns immediately. The graph keeps going in
the background until it finishes or pauses for a high-risk tool. Poll
GET /runs/{run_id}. POST /approval resumes the same checkpoint.
POST /judge grades Error Recovery traces in the background. Poll
GET /judge/{job_id}.

A natural-language report with no incident_id is stored as a new incident
before the model runs. Ids come from INCIDENT_SEQ_FILE, so two reports do not
both become INC-011.
"""

from __future__ import annotations

import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from evaluation.judge import DEFAULT_TRACES, grade_all, load_traces

from .config import ConfigError, load_settings
from .runtime import (
    begin_run,
    execute_investigation,
    normalize_decision,
    read_judge,
    read_status,
    resume_investigation,
    write_judge,
    write_status,
)
from .telemetry import init_tracing

app = FastAPI(title="IT agent")


class WebhookRequest(BaseModel):
    message: str | None = None
    incident_id: str | None = None


class ApprovalRequest(BaseModel):
    run_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)


class JudgeTrace(BaseModel):
    incident_id: str | None = None
    scenario: str | None = None
    trace: list = Field(min_length=1)


class JudgeRequest(BaseModel):
    traces: list[JudgeTrace] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook", status_code=202)
def webhook(body: WebhookRequest, background: BackgroundTasks) -> dict[str, str | None]:
    if not (body.message and body.message.strip()) and not body.incident_id:
        raise HTTPException(status_code=422, detail="Provide a message, an incident_id, or both.")
    _require_settings()
    run_id = uuid.uuid4().hex[:12]
    try:
        record = begin_run(body.message, body.incident_id, run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    background.add_task(execute_investigation, run_id, body.message, record["incident_id"])
    return {"run_id": run_id, "incident_id": record["incident_id"], "status": "running"}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    record = read_status(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return record


@app.post("/approval", status_code=202)
def approval(body: ApprovalRequest, background: BackgroundTasks) -> dict[str, str | None]:
    record = read_status(body.run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown run '{body.run_id}'.")
    if record.get("status") == "running":
        raise HTTPException(status_code=409, detail="This run is still in progress.")
    if record.get("status") != "waiting_for_approval":
        raise HTTPException(status_code=409, detail=f"This run is {record.get('status')}, not waiting for approval.")
    try:
        decision = normalize_decision(body.decision)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _require_settings()
    record["status"] = "running"
    record["message"] = None
    write_status(record)
    background.add_task(resume_investigation, body.run_id, decision)
    return {"run_id": body.run_id, "incident_id": record.get("incident_id"), "status": "running"}


@app.post("/judge", status_code=202)
def judge(body: JudgeRequest, background: BackgroundTasks) -> dict[str, str | int | None]:
    """Grade Error Recovery. Omit traces to score the ten historical investigations."""
    _require_settings()
    try:
        traces = _judge_traces(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    write_judge(
        {
            "job_id": job_id,
            "status": "running",
            "metric": "error_recovery",
            "scale": "1-5",
            "count": len(traces),
            "average_error_recovery_score": None,
            "scores": [],
            "error": None,
        }
    )
    background.add_task(execute_judge, job_id, traces)
    return {"job_id": job_id, "status": "running", "count": len(traces)}


@app.get("/judge/{job_id}")
def get_judge(job_id: str) -> dict:
    record = read_judge(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown judge job '{job_id}'.")
    return record


def execute_judge(job_id: str, traces: list[dict]) -> None:
    try:
        report = grade_all(traces)
        report["job_id"] = job_id
        report["status"] = "final"
        report["error"] = None
        write_judge(report)
    except Exception as exc:
        current = read_judge(job_id) or {"job_id": job_id, "scores": []}
        current["status"] = "failed"
        current["error"] = f"{type(exc).__name__}: {exc}"
        write_judge(current)


def _judge_traces(body: JudgeRequest) -> list[dict]:
    if body.traces is None:
        if not DEFAULT_TRACES.is_file():
            raise FileNotFoundError(f"Historical traces are missing: {DEFAULT_TRACES}")
        return load_traces(DEFAULT_TRACES)
    if not body.traces:
        raise ValueError("traces must be a non-empty list, or omit the field to grade the historical set.")
    return [item.model_dump() for item in body.traces]


def _require_settings() -> None:
    try:
        load_settings()
        init_tracing()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
