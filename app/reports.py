"""Open an incident record for a free-form user report.

Seeded tickets already have an id (INC-001 and so on). A natural-language
report has none, so the runner allocates the next shared id and writes the
record into that run's database before the graph starts. The model then reads
it with get_incident, the same way it reads a seeded ticket.
"""

from __future__ import annotations

from pathlib import Path

from mcp_server.data_store import DataStore


def record_user_report(db_path: str | Path, message: str) -> str:
    """Insert one open incident and return its id. The database must already be seeded."""
    text = message.strip()
    if not text:
        raise ValueError("message is empty")
    store = DataStore(db_path=str(db_path))
    try:
        incident_id = store.next_incident_id()
        now = store.now_str()
        title = text.splitlines()[0].strip()
        if len(title) > 120:
            title = title[:117] + "..."
        mentioned = [name for name in store.service_names() if name.lower() in text.lower()]
        service = mentioned[0] if len(mentioned) == 1 else None
        environment = None
        if service:
            row = store.get_service(service)
            environment = row.get("environment") if row else None
        incident = {
            "incident_id": incident_id,
            "title": title,
            "severity": "SEV-3",
            "service": service,
            "environment": environment or "production",
            "status": "open",
            "started_at": now,
            "detected_at": now,
            "detected_by": "user",
            "reporter": "user",
            "description": text,
            "root_cause": None,
            "resolution": None,
            "resolved_at": None,
            "notes": [],
            "timeline": [{"timestamp": now, "event": "Incident opened from a user report"}],
        }
        store.insert_incident(incident)
        store.record_audit_event("user", "record_user_report", incident_id, "success", {"service": service})
        return incident_id
    finally:
        store.close()
