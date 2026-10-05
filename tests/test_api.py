from fastapi.testclient import TestClient

from app.api import app
from app.runtime import read_judge, read_status, write_status


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("INCIDENT_SEQ_FILE", str(tmp_path / "seq.sqlite"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("JUDGE_DIR", str(tmp_path / "judge"))
    monkeypatch.setattr("app.api.execute_investigation", lambda run_id, message, incident_id: None)
    monkeypatch.setattr("app.api.resume_investigation", lambda run_id, decision: None)


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_assigns_a_new_incident_for_each_report(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    client = TestClient(app)
    first = client.post("/webhook", json={"message": "Customers say email notifications are delayed."})
    second = client.post("/webhook", json={"message": "orders-api is returning HTTP 500 errors."})
    named = client.post("/webhook", json={"incident_id": "INC-001"})
    assert first.status_code == 202
    assert second.status_code == 202
    assert named.status_code == 202
    assert first.json()["incident_id"] == "INC-011"
    assert second.json()["incident_id"] == "INC-012"
    assert named.json()["incident_id"] == "INC-001"
    assert first.json()["status"] == "running"
    saved = read_status(first.json()["run_id"])
    assert saved["incident_id"] == "INC-011"


def test_webhook_rejects_an_empty_report(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post("/webhook", json={})
    assert response.status_code == 422


def test_webhook_unknown_incident(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post("/webhook", json={"incident_id": "INC-999"})
    assert response.status_code == 404


def test_approval_requires_a_paused_run(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    client = TestClient(app)
    missing = client.post("/approval", json={"run_id": "missing", "decision": "approve"})
    assert missing.status_code == 404

    write_status(
        {
            "run_id": "paused1",
            "incident_id": "INC-011",
            "status": "waiting_for_approval",
            "message": "rollback_deployment requires approval.",
            "final_response": None,
            "approval": {"actions": [{"name": "rollback_deployment", "arguments": {}}]},
            "stop_reason": None,
            "error": None,
        }
    )
    accepted = client.post("/approval", json={"run_id": "paused1", "decision": "approve"})
    assert accepted.status_code == 202
    assert accepted.json()["status"] == "running"
    assert read_status("paused1")["status"] == "running"

    write_status(
        {
            "run_id": "done1",
            "incident_id": "INC-004",
            "status": "final",
            "message": "recovered",
            "final_response": "recovered",
            "approval": None,
            "stop_reason": "final",
            "error": None,
        }
    )
    finished = client.post("/approval", json={"run_id": "done1", "decision": "reject"})
    assert finished.status_code == 409
    assert "final" in finished.json()["detail"]


def test_judge_historical_set_is_scored_in_the_background(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    def fake(job_id, traces):
        assert len(traces) == 10
        from app.runtime import write_judge

        write_judge(
            {
                "job_id": job_id,
                "status": "final",
                "metric": "error_recovery",
                "scale": "1-5",
                "count": len(traces),
                "average_error_recovery_score": 4.2,
                "scores": [{"incident_id": traces[0]["incident_id"], "score": 5, "explanation": "recovered", "evidence": []}],
                "error": None,
            }
        )

    monkeypatch.setattr("app.api.execute_judge", fake)
    client = TestClient(app)
    started = client.post("/judge", json={})
    assert started.status_code == 202
    assert started.json()["count"] == 10
    report = client.get(f"/judge/{started.json()['job_id']}")
    assert report.status_code == 200
    assert report.json()["average_error_recovery_score"] == 4.2
    assert read_judge(started.json()["job_id"])["status"] == "final"


def test_judge_accepts_one_posted_trace(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    seen = {}

    def fake(job_id, traces):
        seen["traces"] = traces

    monkeypatch.setattr("app.api.execute_judge", fake)
    client = TestClient(app)
    body = {
        "traces": [
            {
                "incident_id": "INC-006",
                "scenario": "bad timestamp",
                "trace": [{"tool": "search_logs", "arguments": {"start_time": "invalid"}}, {"result": {"status": 400, "error": "Invalid timestamp"}}],
            }
        ]
    }
    started = client.post("/judge", json=body)
    assert started.status_code == 202
    assert started.json()["count"] == 1
    assert seen["traces"][0]["incident_id"] == "INC-006"
    missing = client.get("/judge/does-not-exist")
    assert missing.status_code == 404


def test_judge_rejects_an_empty_trace_list(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post("/judge", json={"traces": []})
    assert response.status_code == 422
