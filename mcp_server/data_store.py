"""SQLite-backed store for the simulated AcmeCloud infrastructure.

JSON files under ``data/`` are the seed; on construction they are loaded into a
SQLite database (in-memory by default, or a file path for persistence). All
reads and writes go through this class so the tool layer never touches JSON or
SQL directly.

Entity tables (services, incidents, deployments, databases, alerts,
dependencies) store a few indexed columns for filtering plus the full record as
a JSON ``payload``. Logs and metrics are stored as flat rows because they are
queried by range.

The store also owns the *simulated clock*: ``now`` starts at the value in
``data/meta.json`` and advances by one minute every time a remediation action
is applied, so "verify recovery" calls can observe fresh data.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

SEED_FILES = {
    "services": "services.json",
    "incidents": "incidents.json",
    "deployments": "deployments.json",
    "databases": "databases.json",
    "alerts": "alerts.json",
    "dependencies": "dependencies.json",
    "logs": "logs.json",
    "metrics": "metrics.json",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS services (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    service TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id TEXT PRIMARY KEY,
    service TEXT NOT NULL,
    deployed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS databases (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    service TEXT,
    status TEXT NOT NULL,
    fired_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dependencies (
    service TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    service TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_service_ts ON logs (service, timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs (timestamp);
CREATE TABLE IF NOT EXISTS metrics (
    service TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    request_rate INTEGER,
    error_rate REAL,
    p95_latency_ms INTEGER,
    PRIMARY KEY (service, timestamp)
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    outcome TEXT NOT NULL,
    details TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def allocate_incident_id(seq_file: str | Path, *, floor: int = 0) -> str:
    """Atomically take the next id. ``floor`` is the highest number already stored in one database."""
    path = Path(seq_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE IF NOT EXISTS seq (id INTEGER PRIMARY KEY CHECK (id = 1), n INTEGER NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO seq (id, n) VALUES (1, 0)")
        current = int(conn.execute("SELECT n FROM seq WHERE id = 1").fetchone()[0])
        nxt = max(current, int(floor)) + 1
        conn.execute("UPDATE seq SET n = ? WHERE id = 1", (nxt,))
        conn.execute("COMMIT")
        return f"INC-{nxt:03d}"
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def format_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(TIME_FORMAT)


def parse_time(value: str) -> datetime:
    """Parse an ISO 8601 timestamp (``Z`` or offset) into an aware UTC datetime.

    Raises ``ValueError`` for anything that is not a valid timestamp.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_time(value: str) -> str:
    return format_time(parse_time(value))


class DataStore:
    """Load the simulated infrastructure into SQLite and expose typed accessors."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR, db_path: str = ":memory:") -> None:
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        # Tool calls may arrive on a different thread than the one that opened the database.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if not self._is_seeded():
            self._seed()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        """Drop all data and reload from the seed files."""
        tables = ["services", "incidents", "deployments", "databases", "alerts", "dependencies", "logs", "metrics", "audit_events", "meta"]
        with self.conn:
            for table in tables:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("DELETE FROM sqlite_sequence WHERE name='audit_events'")
        self._seed()

    def _is_seeded(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM services").fetchone()
        return bool(row and row["n"])

    def _read_seed(self, name: str) -> list[dict[str, Any]]:
        path = self.data_dir / SEED_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"Seed file missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _seed(self) -> None:
        with self.conn:
            for s in self._read_seed("services"):
                self.conn.execute("INSERT INTO services (name, status, payload) VALUES (?, ?, ?)", (s["name"], s["status"], json.dumps(s)))
            for i in self._read_seed("incidents"):
                self.conn.execute(
                    "INSERT INTO incidents (incident_id, service, status, started_at, payload) VALUES (?, ?, ?, ?, ?)",
                    (i["incident_id"], i.get("service"), i["status"], i["started_at"], json.dumps(i)),
                )
            for d in self._read_seed("deployments"):
                self.conn.execute(
                    "INSERT INTO deployments (deployment_id, service, deployed_at, status, payload) VALUES (?, ?, ?, ?, ?)",
                    (d["deployment_id"], d["service"], d["deployed_at"], d["status"], json.dumps(d)),
                )
            for db in self._read_seed("databases"):
                self.conn.execute("INSERT INTO databases (name, status, payload) VALUES (?, ?, ?)", (db["name"], db["status"], json.dumps(db)))
            for a in self._read_seed("alerts"):
                self.conn.execute(
                    "INSERT INTO alerts (alert_id, service, status, fired_at, payload) VALUES (?, ?, ?, ?, ?)",
                    (a["alert_id"], a.get("service"), a["status"], a["fired_at"], json.dumps(a)),
                )
            for dep in self._read_seed("dependencies"):
                self.conn.execute("INSERT INTO dependencies (service, payload) VALUES (?, ?)", (dep["service"], json.dumps(dep)))
            self.conn.executemany(
                "INSERT INTO logs (log_id, timestamp, service, level, message, source) VALUES (?, ?, ?, ?, ?, ?)",
                [(l["log_id"], l["timestamp"], l["service"], l["level"], l["message"], l.get("source")) for l in self._read_seed("logs")],
            )
            self.conn.executemany(
                "INSERT INTO metrics (service, timestamp, cpu_percent, memory_percent, request_rate, error_rate, p95_latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (m["service"], m["timestamp"], m.get("cpu_percent"), m.get("memory_percent"), m.get("request_rate"), m.get("error_rate"), m.get("p95_latency_ms"))
                    for m in self._read_seed("metrics")
                ],
            )
            meta_path = self.data_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            now = meta.get("simulated_now") or self._latest_metric_timestamp() or format_time(datetime.now(timezone.utc))
            self.conn.execute("INSERT INTO meta (key, value) VALUES ('now', ?)", (now,))

    def _latest_metric_timestamp(self) -> str | None:
        row = self.conn.execute("SELECT MAX(timestamp) AS t FROM metrics").fetchone()
        return row["t"] if row else None

    # ------------------------------------------------------------------ #
    # simulated clock
    # ------------------------------------------------------------------ #
    @property
    def now(self) -> datetime:
        row = self.conn.execute("SELECT value FROM meta WHERE key='now'").fetchone()
        return parse_time(row["value"])

    def now_str(self) -> str:
        return format_time(self.now)

    def advance_clock(self, seconds: int = 60) -> datetime:
        new = self.now + timedelta(seconds=seconds)
        with self.conn:
            self.conn.execute("UPDATE meta SET value=? WHERE key='now'", (format_time(new),))
        return new

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _payloads(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [json.loads(r["payload"]) for r in rows]

    @staticmethod
    def _payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return json.loads(row["payload"]) if row else None

    # ------------------------------------------------------------------ #
    # services
    # ------------------------------------------------------------------ #
    def list_services(self) -> list[dict[str, Any]]:
        return self._payloads(self.conn.execute("SELECT payload FROM services ORDER BY name"))

    def get_service(self, name: str) -> dict[str, Any] | None:
        return self._payload(self.conn.execute("SELECT payload FROM services WHERE name=?", (name,)).fetchone())

    def service_names(self) -> list[str]:
        return [r["name"] for r in self.conn.execute("SELECT name FROM services ORDER BY name")]

    def save_service(self, service: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("UPDATE services SET status=?, payload=? WHERE name=?", (service["status"], json.dumps(service), service["name"]))

    # ------------------------------------------------------------------ #
    # incidents
    # ------------------------------------------------------------------ #
    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        return self._payload(self.conn.execute("SELECT payload FROM incidents WHERE incident_id=?", (incident_id,)).fetchone())

    def list_incidents(self, status: str | None = None, service: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT payload FROM incidents WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if service:
            sql += " AND service=?"
            params.append(service)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return self._payloads(self.conn.execute(sql, params))

    def next_incident_id(self) -> str:
        """Next ``INC-NNN``. A shared ``INCIDENT_SEQ_FILE`` keeps ids unique across run databases."""
        floor = self._max_incident_number()
        seq = os.environ.get("INCIDENT_SEQ_FILE")
        if seq:
            return allocate_incident_id(seq, floor=floor)
        return f"INC-{floor + 1:03d}"

    def _max_incident_number(self) -> int:
        rows = self.conn.execute("SELECT incident_id FROM incidents").fetchall()
        nums = [int(str(r["incident_id"]).split("-")[-1]) for r in rows if str(r["incident_id"]).split("-")[-1].isdigit()]
        return max(nums) if nums else 0

    def insert_incident(self, incident: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO incidents (incident_id, service, status, started_at, payload) VALUES (?, ?, ?, ?, ?)",
                (incident["incident_id"], incident.get("service"), incident["status"], incident["started_at"], json.dumps(incident)),
            )

    def save_incident(self, incident: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE incidents SET service=?, status=?, payload=? WHERE incident_id=?",
                (incident.get("service"), incident["status"], json.dumps(incident), incident["incident_id"]),
            )

    # ------------------------------------------------------------------ #
    # deployments
    # ------------------------------------------------------------------ #
    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        return self._payload(self.conn.execute("SELECT payload FROM deployments WHERE deployment_id=?", (deployment_id,)).fetchone())

    def list_deployments(self, service: str | None = None, limit: int = 5, since: str | None = None, until: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT payload FROM deployments WHERE 1=1", []
        if service:
            sql += " AND service=?"
            params.append(service)
        if since:
            sql += " AND deployed_at >= ?"
            params.append(since)
        if until:
            sql += " AND deployed_at <= ?"
            params.append(until)
        sql += " ORDER BY deployed_at DESC LIMIT ?"
        params.append(limit)
        return self._payloads(self.conn.execute(sql, params))

    def next_deployment_id(self) -> str:
        rows = self.conn.execute("SELECT deployment_id FROM deployments").fetchall()
        nums = [int(r["deployment_id"].split("-")[-1]) for r in rows if r["deployment_id"].split("-")[-1].isdigit()]
        return f"deploy-{(max(nums) + 1 if nums else 1)}"

    def insert_deployment(self, deployment: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO deployments (deployment_id, service, deployed_at, status, payload) VALUES (?, ?, ?, ?, ?)",
                (deployment["deployment_id"], deployment["service"], deployment["deployed_at"], deployment["status"], json.dumps(deployment)),
            )

    def save_deployment(self, deployment: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE deployments SET status=?, payload=? WHERE deployment_id=?",
                (deployment["status"], json.dumps(deployment), deployment["deployment_id"]),
            )

    # ------------------------------------------------------------------ #
    # logs
    # ------------------------------------------------------------------ #
    def search_logs(
        self,
        service: str | None = None,
        level: str | None = None,
        keyword: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (matching logs in ascending time order, total match count).

        When more than ``limit`` rows match, the *most recent* ``limit`` rows are returned.
        Time bounds must already be normalized to ``TIME_FORMAT``.
        """
        where, params = ["1=1"], []
        if service:
            where.append("service=?")
            params.append(service)
        if level:
            where.append("level=?")
            params.append(level.upper())
        if keyword:
            where.append("LOWER(message) LIKE ?")
            params.append(f"%{keyword.lower()}%")
        if start_time:
            where.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where.append("timestamp <= ?")
            params.append(end_time)
        clause = " AND ".join(where)
        total = self.conn.execute(f"SELECT COUNT(*) AS n FROM logs WHERE {clause}", params).fetchone()["n"]
        rows = self.conn.execute(
            f"SELECT log_id, timestamp, service, level, message, source FROM logs WHERE {clause} ORDER BY timestamp DESC, log_id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [dict(r) for r in reversed(rows)], total

    def insert_log(self, service: str, level: str, message: str, source: str | None = None, timestamp: str | None = None) -> dict[str, Any]:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM logs").fetchone()
        entry = {
            "log_id": f"log-{row['n'] + 1:05d}",
            "timestamp": timestamp or self.now_str(),
            "service": service,
            "level": level.upper(),
            "message": message,
            "source": source or "simulator",
        }
        with self.conn:
            self.conn.execute(
                "INSERT INTO logs (log_id, timestamp, service, level, message, source) VALUES (?, ?, ?, ?, ?, ?)",
                (entry["log_id"], entry["timestamp"], entry["service"], entry["level"], entry["message"], entry["source"]),
            )
        return entry

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def get_metrics(self, service: str, start_time: str | None = None, end_time: str | None = None, limit: int = 120) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM metrics WHERE service=?", [service]
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in reversed(rows)]

    def latest_metric(self, service: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM metrics WHERE service=? ORDER BY timestamp DESC LIMIT 1", (service,)).fetchone()
        return dict(row) if row else None

    def insert_metric(self, sample: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO metrics (service, timestamp, cpu_percent, memory_percent, request_rate, error_rate, p95_latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sample["service"], sample["timestamp"], sample.get("cpu_percent"), sample.get("memory_percent"), sample.get("request_rate"), sample.get("error_rate"), sample.get("p95_latency_ms")),
            )

    # ------------------------------------------------------------------ #
    # databases
    # ------------------------------------------------------------------ #
    def list_databases(self) -> list[dict[str, Any]]:
        return self._payloads(self.conn.execute("SELECT payload FROM databases ORDER BY name"))

    def get_database(self, name: str) -> dict[str, Any] | None:
        return self._payload(self.conn.execute("SELECT payload FROM databases WHERE name=?", (name,)).fetchone())

    def save_database(self, database: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("UPDATE databases SET status=?, payload=? WHERE name=?", (database["status"], json.dumps(database), database["name"]))

    # ------------------------------------------------------------------ #
    # alerts
    # ------------------------------------------------------------------ #
    def list_alerts(self, service: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT payload FROM alerts WHERE 1=1", []
        if service:
            sql += " AND service=?"
            params.append(service)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY fired_at DESC LIMIT ?"
        params.append(limit)
        return self._payloads(self.conn.execute(sql, params))

    def save_alert(self, alert: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("UPDATE alerts SET status=?, payload=? WHERE alert_id=?", (alert["status"], json.dumps(alert), alert["alert_id"]))

    # ------------------------------------------------------------------ #
    # dependencies
    # ------------------------------------------------------------------ #
    def get_dependencies(self, service: str) -> list[dict[str, Any]]:
        record = self._payload(self.conn.execute("SELECT payload FROM dependencies WHERE service=?", (service,)).fetchone())
        return list(record["depends_on"]) if record else []

    def get_dependents(self, name: str) -> list[str]:
        out = []
        for record in self._payloads(self.conn.execute("SELECT payload FROM dependencies ORDER BY service")):
            if any(dep["name"] == name for dep in record["depends_on"]):
                out.append(record["service"])
        return out

    # ------------------------------------------------------------------ #
    # audit
    # ------------------------------------------------------------------ #
    def record_audit_event(self, actor: str, action: str, target: str | None, outcome: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "timestamp": self.now_str(),
            "actor": actor,
            "action": action,
            "target": target,
            "outcome": outcome,
            "details": details or {},
        }
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO audit_events (timestamp, actor, action, target, outcome, details) VALUES (?, ?, ?, ?, ?, ?)",
                (event["timestamp"], actor, action, target, outcome, json.dumps(event["details"])),
            )
            event["event_id"] = cur.lastrowid
        return event

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM audit_events ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["details"] = json.loads(d["details"]) if d["details"] else {}
            out.append(d)
        return out
