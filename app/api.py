"""HTTP API for one investigation per request.

    uvicorn app.api:app --host 0.0.0.0 --port 8000

POST /webhook starts a run and returns immediately. The graph keeps going in
the background until it finishes or pauses for a high-risk tool. Poll
GET /runs/{run_id}. POST /approval resumes the same checkpoint.

A natural-language report with no incident_id is stored as a new incident
before the model runs. Ids come from INCIDENT_SEQ_FILE, so two reports do not
both become INC-011.
"""

from __future__ import annotations

import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ConfigError, load_settings
from .runtime import (
    begin_run,
    execute_investigation,
    normalize_decision,
    read_status,
    resume_investigation,
    write_status,
)
from .telemetry import init_tracing

app = FastAPI(title="AcmeCloud IT agent")


class WebhookRequest(BaseModel):
    message: str | None = None
    incident_id: str | None = None


class ApprovalRequest(BaseModel):
    run_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)


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


def _require_settings() -> None:
    try:
        load_settings()
        init_tracing()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
