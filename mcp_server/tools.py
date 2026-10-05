"""Tool functions over the simulated infrastructure.

Three categories, mirroring the risk model enforced by ``app.hooks``:

* read-only diagnostics  - get_incident, list_incidents, get_service_health, get_recent_deployments,
                           get_deployment, search_logs, get_metrics, get_database_status, get_alerts,
                           get_dependencies
* low-risk actions       - add_incident_note, create_incident, update_incident
* high-risk actions      - restart_service, rollback_deployment, scale_service, disable_feature

Every tool returns a JSON-serialisable dict. Input problems and simulated
infrastructure failures raise ``ToolError`` carrying an HTTP-like status; the
``execute_tool`` router converts those into ``{"status": ..., "error": ...}``
so the model always receives a structured error it can reason about.

High-risk tools do not fix anything by magic: whether an action heals a
service is decided by the hidden ``simulation`` block in the service seed
(e.g. restarting orders-api does not help because the bug is in the deployed
version; rolling back deploy-8501 fails because the previous image was pruned).
"""

from __future__ import annotations

import inspect
import threading
from datetime import timedelta
from typing import Any, Callable

from .data_store import DataStore, format_time, normalize_time, parse_time

MAX_LOG_WINDOW = timedelta(hours=24)
MAX_LOG_LIMIT = 500
MAX_METRIC_LIMIT = 240
MAX_REPLICAS = 20

INCIDENT_STATUSES = {"open", "investigating", "identified", "monitoring", "resolved", "closed"}
SEVERITIES = {"SEV-1", "SEV-2", "SEV-3", "SEV-4"}
LOG_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}

READ_ONLY_TOOLS = {
    "get_incident", "list_incidents", "get_service_health", "get_recent_deployments", "get_deployment",
    "search_logs", "get_metrics", "get_database_status", "get_alerts", "get_dependencies",
}
LOW_RISK_TOOLS = {"add_incident_note", "create_incident", "update_incident"}
HIGH_RISK_TOOLS = {"restart_service", "rollback_deployment", "scale_service", "disable_feature"}


class ToolError(Exception):
    """A structured tool failure. ``status`` follows HTTP semantics (400, 404, 409, 500...)."""

    def __init__(self, status: int, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "error": self.message}
        if self.details:
            out["details"] = self.details
        return out


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(400, f"'{field}' must be a non-empty string.")
    return value.strip()


def _parse_timestamp(value: Any, field: str) -> str:
    try:
        return normalize_time(_require_str(value, field))
    except ToolError:
        raise
    except (ValueError, TypeError) as exc:
        raise ToolError(
            400,
            f"Invalid time range: '{field}' is not a valid ISO 8601 timestamp (got {value!r}). Use the form 2026-09-30T10:00:00Z.",
            {"field": field, "value": value, "reason": str(exc)},
        ) from exc


def _parse_window(start_time: Any, end_time: Any, store: DataStore, default_span: timedelta, max_span: timedelta | None = None) -> tuple[str, str]:
    """Resolve an optional [start, end] window relative to the simulated clock."""
    end = _parse_timestamp(end_time, "end_time") if end_time is not None else store.now_str()
    start = _parse_timestamp(start_time, "start_time") if start_time is not None else format_time(parse_time(end) - default_span)
    if parse_time(start) >= parse_time(end):
        raise ToolError(400, f"Invalid time range: start_time ({start}) must be before end_time ({end}).", {"start_time": start, "end_time": end})
    if max_span is not None and parse_time(end) - parse_time(start) > max_span:
        raise ToolError(
            400,
            f"Invalid time range: window exceeds the maximum of {int(max_span.total_seconds() // 3600)} hours. Narrow start_time/end_time.",
            {"start_time": start, "end_time": end},
        )
    return start, end


def _parse_limit(value: Any, maximum: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(400, f"'limit' must be an integer between 1 and {maximum}.")
    if value < 1 or value > maximum:
        raise ToolError(400, f"'limit' must be between 1 and {maximum} (got {value}).")
    return value


def _public(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip hidden simulation fields before returning a record to the model."""
    if record is None:
        return None
    return {k: v for k, v in record.items() if k != "simulation"}


# --------------------------------------------------------------------------- #
# toolkit
# --------------------------------------------------------------------------- #
class ToolKit:
    """All agent-callable tools bound to one ``DataStore``."""

    def __init__(self, store: DataStore | None = None, actor: str = "sre-agent") -> None:
        self.store = store or DataStore()
        self.actor = actor
        self._call_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _service_or_404(self, name: Any) -> dict[str, Any]:
        name = _require_str(name, "service")
        service = self.store.get_service(name)
        if not service:
            raise ToolError(404, f"Unknown service '{name}'.", {"known_services": self.store.service_names()})
        return service

    def _incident_or_404(self, incident_id: Any) -> dict[str, Any]:
        incident_id = _require_str(incident_id, "incident_id")
        incident = self.store.get_incident(incident_id)
        if not incident:
            raise ToolError(404, f"Unknown incident '{incident_id}'.")
        return incident

    def _deployment_or_404(self, deployment_id: Any) -> dict[str, Any]:
        deployment_id = _require_str(deployment_id, "deployment_id")
        deployment = self.store.get_deployment(deployment_id)
        if not deployment:
            raise ToolError(404, f"Unknown deployment '{deployment_id}'.")
        return deployment

    def _health_snapshot(self, service: dict[str, Any]) -> dict[str, Any]:
        latest = self.store.latest_metric(service["name"])
        snapshot = {
            "service": service["name"],
            "environment": service["environment"],
            "version": service["version"],
            "status": service["status"],
            "replicas": service["replicas"],
            "healthy_replicas": service["healthy_replicas"],
            "owner_team": service.get("owner_team"),
            "feature_flags": service.get("feature_flags", {}),
            "error_rate": latest["error_rate"] if latest else None,
            "p95_latency_ms": latest["p95_latency_ms"] if latest else None,
            "request_rate": latest["request_rate"] if latest else None,
            "metrics_as_of": latest["timestamp"] if latest else None,
        }
        if latest and parse_time(latest["timestamp"]) < self.store.now - timedelta(minutes=3):
            snapshot["metrics_stale"] = True
        if service.get("rollout"):
            snapshot["rollout"] = service["rollout"]
        return snapshot

    def _heal_service(self, name: str, reason: str) -> None:
        """Mark a service healthy: replicas ready, fresh baseline metrics, alerts resolved, dependents re-evaluated."""
        service = self.store.get_service(name)
        if not service:
            return
        service["status"] = "healthy"
        service["healthy_replicas"] = service["replicas"]
        service.pop("rollout", None)
        self.store.save_service(service)

        baseline = service.get("simulation", {}).get("baseline")
        if baseline:
            self.store.insert_metric({"service": name, "timestamp": self.store.now_str(), **baseline})
        self.store.insert_log(name, "INFO", f"Service recovered: {reason}", source="simulator")

        for alert in self.store.list_alerts(service=name, status="firing"):
            alert["status"] = "resolved"
            alert["resolved_at"] = self.store.now_str()
            self.store.save_alert(alert)

        # A datastore with the same name (redis-cache) recovers with the service; a
        # dependency database that the service itself was degrading (an exhausted
        # pool) recovers once the client stops misbehaving.
        affected_dbs = [self.store.get_database(name)] + [self.store.get_database(d["name"]) for d in self.store.get_dependencies(name)]
        for db in affected_dbs:
            if db and db["status"] != "healthy":
                db["status"] = "healthy"
                db["last_heartbeat"] = self.store.now_str()
                db["connections_active"] = min(db.get("connections_active") or 0, 25)
                db.pop("warnings", None)
                self.store.save_database(db)

        for other in self.store.list_services():
            if other.get("simulation", {}).get("blocked_by") == name and other["status"] != "healthy":
                self._heal_service(other["name"], f"dependency {name} recovered")

    # ------------------------------------------------------------------ #
    # read-only diagnostics
    # ------------------------------------------------------------------ #
    def get_incident(self, incident_id: str) -> dict[str, Any]:
        """Fetch an incident record including its notes and timeline."""
        return self._incident_or_404(incident_id)

    def list_incidents(self, status: str | None = None, service: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """List incidents, optionally filtered by status and/or service. Newest first."""
        if status is not None:
            status = _require_str(status, "status").lower()
            if status not in INCIDENT_STATUSES:
                raise ToolError(400, f"Unknown incident status '{status}'. Valid: {sorted(INCIDENT_STATUSES)}.")
        if service is not None:
            service = _require_str(service, "service")
        limit = _parse_limit(limit, 100, 20)
        incidents = self.store.list_incidents(status=status, service=service, limit=limit)
        summary = [
            {k: i.get(k) for k in ("incident_id", "title", "severity", "service", "status", "started_at")}
            for i in incidents
        ]
        return {"count": len(summary), "incidents": summary}

    def get_service_health(self, service: str) -> dict[str, Any]:
        """Current health of a service: status, replicas, latest error rate / latency, rollout state."""
        return self._health_snapshot(self._service_or_404(service))

    def get_recent_deployments(self, service: str | None = None, limit: int | None = None, since: str | None = None) -> dict[str, Any]:
        """Most recent deployments (all services or one), newest first."""
        if service is not None:
            self._service_or_404(service)
        limit = _parse_limit(limit, 50, 5)
        since_norm = _parse_timestamp(since, "since") if since is not None else None
        deployments = [_public(d) for d in self.store.list_deployments(service=service, limit=limit, since=since_norm)]
        for d in deployments:
            d.pop("changes", None)  # full diff is available via get_deployment
        return {"count": len(deployments), "deployments": deployments}

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Full deployment record including the list of changes it introduced."""
        return _public(self._deployment_or_404(deployment_id))

    def search_logs(
        self,
        service: str | None = None,
        level: str | None = None,
        keyword: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search log lines. Window defaults to the last 60 minutes; maximum window 24 hours; max 500 lines.

        Timestamps must be ISO 8601 (e.g. 2026-09-30T10:00:00Z). Returns the most recent matches in ascending order.
        """
        if service is not None:
            self._service_or_404(service)
        if level is not None:
            level = _require_str(level, "level").upper()
            if level not in LOG_LEVELS:
                raise ToolError(400, f"Unknown log level '{level}'. Valid: {sorted(LOG_LEVELS)}.")
        if keyword is not None:
            keyword = _require_str(keyword, "keyword")
        limit = _parse_limit(limit, MAX_LOG_LIMIT, 50)
        start, end = _parse_window(start_time, end_time, self.store, default_span=timedelta(minutes=60), max_span=MAX_LOG_WINDOW)
        logs, total = self.store.search_logs(service=service, level=level, keyword=keyword, start_time=start, end_time=end, limit=limit)
        return {
            "query": {"service": service, "level": level, "keyword": keyword, "start_time": start, "end_time": end, "limit": limit},
            "total_matches": total,
            "returned": len(logs),
            "truncated": total > len(logs),
            "logs": logs,
        }

    def get_metrics(self, service: str, start_time: str | None = None, end_time: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """Per-minute metrics for a service (cpu, memory, request rate, error rate, p95 latency) with a summary."""
        svc = self._service_or_404(service)
        limit = _parse_limit(limit, MAX_METRIC_LIMIT, 60)
        start, end = _parse_window(start_time, end_time, self.store, default_span=timedelta(minutes=20))
        points = self.store.get_metrics(svc["name"], start_time=start, end_time=end, limit=limit)
        summary: dict[str, Any] = {"points": len(points)}
        if points:
            errs = [p["error_rate"] for p in points if p["error_rate"] is not None]
            lats = [p["p95_latency_ms"] for p in points if p["p95_latency_ms"] is not None]
            summary.update(
                first_timestamp=points[0]["timestamp"],
                last_timestamp=points[-1]["timestamp"],
                error_rate={"first": errs[0], "min": min(errs), "max": max(errs), "latest": errs[-1]} if errs else None,
                p95_latency_ms={"first": lats[0], "min": min(lats), "max": max(lats), "latest": lats[-1]} if lats else None,
            )
        else:
            summary["note"] = "No metric samples in this window (scrape target may be down)."
        return {"service": svc["name"], "window": {"start_time": start, "end_time": end}, "summary": summary, "metrics": points}

    def get_database_status(self, database: str | None = None) -> dict[str, Any]:
        """Status of one database (by name) or all databases: connections, latency, replication lag, warnings."""
        if database is None:
            dbs = self.store.list_databases()
            return {"count": len(dbs), "databases": dbs}
        database = _require_str(database, "database")
        db = self.store.get_database(database)
        if not db:
            raise ToolError(404, f"Unknown database '{database}'.", {"known_databases": [d["name"] for d in self.store.list_databases()]})
        return db

    def get_alerts(self, service: str | None = None, status: str | None = None) -> dict[str, Any]:
        """Alerts, optionally filtered by service and status ('firing' or 'resolved')."""
        if service is not None:
            service = _require_str(service, "service")
        if status is not None:
            status = _require_str(status, "status").lower()
            if status not in {"firing", "resolved"}:
                raise ToolError(400, f"Unknown alert status '{status}'. Valid: ['firing', 'resolved'].")
        alerts = self.store.list_alerts(service=service, status=status)
        return {"count": len(alerts), "alerts": alerts}

    def get_dependencies(self, service: str) -> dict[str, Any]:
        """Upstream dependencies of a service (with their current status) and the services that depend on it."""
        svc = self._service_or_404(service)
        depends_on = []
        for dep in self.store.get_dependencies(svc["name"]):
            entry = dict(dep)
            target = self.store.get_service(dep["name"]) or self.store.get_database(dep["name"])
            entry["status"] = target["status"] if target else "unknown"
            depends_on.append(entry)
        dependents = []
        for name in self.store.get_dependents(svc["name"]):
            other = self.store.get_service(name)
            dependents.append({"name": name, "status": other["status"] if other else "unknown"})
        return {"service": svc["name"], "depends_on": depends_on, "depended_on_by": dependents}

    # ------------------------------------------------------------------ #
    # low-risk actions
    # ------------------------------------------------------------------ #
    def add_incident_note(self, incident_id: str, note: str, author: str | None = None) -> dict[str, Any]:
        """Append an investigation note to an incident."""
        incident = self._incident_or_404(incident_id)
        note = _require_str(note, "note")
        entry = {"timestamp": self.store.now_str(), "author": author or self.actor, "note": note}
        incident.setdefault("notes", []).append(entry)
        self.store.save_incident(incident)
        self.store.record_audit_event(self.actor, "add_incident_note", incident["incident_id"], "success", {"note": note})
        return {"incident_id": incident["incident_id"], "note": entry, "note_count": len(incident["notes"])}

    def create_incident(self, title: str, service: str, severity: str, description: str) -> dict[str, Any]:
        """Open a new incident for a service."""
        title = _require_str(title, "title")
        svc = self._service_or_404(service)
        severity = _require_str(severity, "severity").upper()
        if severity not in SEVERITIES:
            raise ToolError(400, f"Invalid severity '{severity}'. Valid: {sorted(SEVERITIES)}.")
        description = _require_str(description, "description")
        now = self.store.now_str()
        incident = {
            "incident_id": self.store.next_incident_id(),
            "title": title,
            "severity": severity,
            "service": svc["name"],
            "environment": svc["environment"],
            "status": "open",
            "started_at": now,
            "detected_at": now,
            "detected_by": self.actor,
            "reporter": self.actor,
            "description": description,
            "root_cause": None,
            "resolution": None,
            "resolved_at": None,
            "notes": [],
            "timeline": [{"timestamp": now, "event": f"Incident created by {self.actor}"}],
        }
        self.store.insert_incident(incident)
        self.store.record_audit_event(self.actor, "create_incident", incident["incident_id"], "success", {"title": title, "service": svc["name"]})
        return incident

    def update_incident(
        self,
        incident_id: str,
        status: str | None = None,
        root_cause: str | None = None,
        resolution: str | None = None,
        severity: str | None = None,
    ) -> dict[str, Any]:
        """Update incident status, root cause, resolution and/or severity. Setting status 'resolved' records resolved_at."""
        incident = self._incident_or_404(incident_id)
        changes: dict[str, Any] = {}
        if status is not None:
            status = _require_str(status, "status").lower()
            if status not in INCIDENT_STATUSES:
                raise ToolError(400, f"Unknown incident status '{status}'. Valid: {sorted(INCIDENT_STATUSES)}.")
            changes["status"] = status
        if root_cause is not None:
            changes["root_cause"] = _require_str(root_cause, "root_cause")
        if resolution is not None:
            changes["resolution"] = _require_str(resolution, "resolution")
        if severity is not None:
            severity = _require_str(severity, "severity").upper()
            if severity not in SEVERITIES:
                raise ToolError(400, f"Invalid severity '{severity}'. Valid: {sorted(SEVERITIES)}.")
            changes["severity"] = severity
        if not changes:
            raise ToolError(400, "Nothing to update: provide at least one of status, root_cause, resolution, severity.")

        now = self.store.now_str()
        previous_status = incident["status"]
        incident.update(changes)
        if changes.get("status") in {"resolved", "closed"} and not incident.get("resolved_at"):
            incident["resolved_at"] = now
        if changes.get("status") and changes["status"] != previous_status:
            incident.setdefault("timeline", []).append({"timestamp": now, "event": f"Status changed {previous_status} -> {changes['status']} by {self.actor}"})
        self.store.save_incident(incident)
        self.store.record_audit_event(self.actor, "update_incident", incident["incident_id"], "success", changes)
        return incident

    # ------------------------------------------------------------------ #
    # high-risk actions (simulated)
    # ------------------------------------------------------------------ #
    def restart_service(self, service: str, reason: str | None = None) -> dict[str, Any]:
        """Rolling restart of all replicas of a production service. HIGH RISK."""
        svc = self._service_or_404(service)
        before = self._health_snapshot(svc)
        self.store.advance_clock(60)
        sim = svc.get("simulation", {})
        heals = bool(sim.get("remediations", {}).get("restart_service"))
        self.store.insert_log(svc["name"], "INFO", f"Rolling restart initiated by {self.actor}: {reason or 'no reason given'}", source="orchestrator")

        if heals:
            self._heal_service(svc["name"], "rolling restart completed")
            outcome_note = "All replicas restarted and passed readiness checks."
        else:
            # Pods come back, then fail for the same reason as before.
            self.store.insert_log(svc["name"], "WARN", "Replicas restarted but readiness is degrading again for the same reason as before the restart", source="orchestrator")
            outcome_note = "Replicas restarted, but service health did not improve. The underlying cause persists."
            if sim.get("blocked_by"):
                outcome_note += f" Upstream dependency '{sim['blocked_by']}' is still unhealthy."

        after = self._health_snapshot(self.store.get_service(svc["name"]))
        self.store.record_audit_event(self.actor, "restart_service", svc["name"], "success", {"reason": reason, "health_improved": heals})
        return {"action": "restart_service", "service": svc["name"], "result": "completed", "health_improved": heals, "note": outcome_note, "before": before, "after": after}

    def rollback_deployment(self, deployment_id: str, reason: str | None = None) -> dict[str, Any]:
        """Roll a service back to the version before the given deployment. HIGH RISK. May fail if the previous artifact is unavailable."""
        deployment = self._deployment_or_404(deployment_id)
        svc = self._service_or_404(deployment["service"])
        if deployment["status"] == "rolled_back":
            raise ToolError(409, f"Deployment '{deployment['deployment_id']}' has already been rolled back (by {deployment.get('rolled_back_by', 'unknown')}).")
        if deployment.get("type") == "rollback":
            raise ToolError(409, f"Deployment '{deployment['deployment_id']}' is itself a rollback; roll back the original release instead.")

        self.store.advance_clock(60)
        self.store.insert_log(svc["name"], "INFO", f"Rollback of {deployment['deployment_id']} ({deployment['version']} -> {deployment['previous_version']}) initiated by {self.actor}", source="ci-pipeline")

        error = deployment.get("simulation", {}).get("rollback_error")
        if error:
            deployment["rollback_attempts"] = deployment.get("rollback_attempts", 0) + 1
            self.store.save_deployment(deployment)
            self.store.insert_log(svc["name"], "ERROR", error, source="ci-pipeline")
            self.store.record_audit_event(self.actor, "rollback_deployment", deployment["deployment_id"], "failure", {"reason": reason, "error": error})
            raise ToolError(500, error, {"deployment_id": deployment["deployment_id"], "service": svc["name"], "previous_version": deployment["previous_version"]})

        rollback = {
            "deployment_id": self.store.next_deployment_id(),
            "service": svc["name"],
            "version": deployment["previous_version"],
            "previous_version": deployment["version"],
            "environment": deployment["environment"],
            "deployed_at": self.store.now_str(),
            "completed_at": self.store.now_str(),
            "status": "successful",
            "type": "rollback",
            "deployed_by": self.actor,
            "commit": None,
            "change_summary": f"Rollback of {deployment['deployment_id']}: {reason or 'no reason given'}",
            "changes": [f"Revert to {deployment['previous_version']}"],
            "rollback_of": deployment["deployment_id"],
        }
        self.store.insert_deployment(rollback)
        deployment["status"] = "rolled_back"
        deployment["rolled_back_by"] = rollback["deployment_id"]
        self.store.save_deployment(deployment)

        svc["version"] = deployment["previous_version"]
        svc.pop("rollout", None)
        self.store.save_service(svc)
        self.store.insert_log(svc["name"], "INFO", f"Deployment {rollback['deployment_id']} completed: {svc['replicas']}/{svc['replicas']} pods running {svc['version']}", source="ci-pipeline")

        heals = svc.get("simulation", {}).get("remediations", {}).get("rollback_deployment") == deployment["deployment_id"]
        if heals:
            self._heal_service(svc["name"], f"rolled back {deployment['deployment_id']} to {svc['version']}")
            note = "Rollback completed and the service passed readiness checks."
        else:
            note = "Rollback completed, but service health did not improve. This deployment was probably not the cause."

        self.store.record_audit_event(self.actor, "rollback_deployment", deployment["deployment_id"], "success", {"reason": reason, "rollback_deployment_id": rollback["deployment_id"], "health_improved": heals})
        return {
            "action": "rollback_deployment",
            "service": svc["name"],
            "rolled_back": deployment["deployment_id"],
            "rollback_deployment_id": rollback["deployment_id"],
            "version": svc["version"],
            "result": "completed",
            "health_improved": heals,
            "note": note,
            "after": self._health_snapshot(self.store.get_service(svc["name"])),
        }

    def scale_service(self, service: str, replicas: int, reason: str | None = None) -> dict[str, Any]:
        """Change the replica count of a service (1-20). HIGH RISK."""
        svc = self._service_or_404(service)
        if isinstance(replicas, bool) or not isinstance(replicas, int):
            raise ToolError(400, f"'replicas' must be an integer between 1 and {MAX_REPLICAS}.")
        if replicas < 1 or replicas > MAX_REPLICAS:
            raise ToolError(400, f"'replicas' must be between 1 and {MAX_REPLICAS} (got {replicas}).")
        if replicas == svc["replicas"]:
            raise ToolError(409, f"Service '{svc['name']}' already has {replicas} replicas.")

        self.store.advance_clock(60)
        previous, previous_healthy = svc["replicas"], svc["healthy_replicas"]
        svc["replicas"] = replicas
        if svc["status"] == "healthy":
            svc["healthy_replicas"] = replicas
        else:
            ratio = previous_healthy / previous if previous else 0
            svc["healthy_replicas"] = min(replicas, max(0, round(replicas * ratio)))
        self.store.save_service(svc)
        self.store.insert_log(svc["name"], "INFO", f"Scaled {previous} -> {replicas} replicas by {self.actor}: {reason or 'no reason given'}", source="orchestrator")
        self.store.record_audit_event(self.actor, "scale_service", svc["name"], "success", {"from": previous, "to": replicas, "reason": reason})
        return {
            "action": "scale_service",
            "service": svc["name"],
            "result": "completed",
            "previous_replicas": previous,
            "replicas": replicas,
            "healthy_replicas": svc["healthy_replicas"],
            "note": None if svc["status"] == "healthy" else "Service is not healthy; scaling changes capacity but does not address the underlying fault.",
            "after": self._health_snapshot(svc),
        }

    def disable_feature(self, service: str, feature_flag: str, reason: str | None = None) -> dict[str, Any]:
        """Turn off a feature flag on a service. HIGH RISK."""
        svc = self._service_or_404(service)
        feature_flag = _require_str(feature_flag, "feature_flag")
        flags = svc.get("feature_flags", {})
        if feature_flag not in flags:
            raise ToolError(404, f"Service '{svc['name']}' has no feature flag '{feature_flag}'.", {"known_flags": sorted(flags)})
        if flags[feature_flag] is False:
            raise ToolError(409, f"Feature flag '{feature_flag}' on '{svc['name']}' is already disabled.")

        self.store.advance_clock(60)
        flags[feature_flag] = False
        svc["feature_flags"] = flags
        self.store.save_service(svc)
        self.store.insert_log(svc["name"], "INFO", f"Feature flag {feature_flag} disabled by {self.actor}: {reason or 'no reason given'}", source="feature-flags")

        heals = svc.get("simulation", {}).get("remediations", {}).get("disable_feature") == feature_flag
        if heals:
            self._heal_service(svc["name"], f"feature flag {feature_flag} disabled")
            note = "Feature disabled; error rate returned to baseline."
        else:
            note = "Feature disabled, but service health did not improve."
        self.store.record_audit_event(self.actor, "disable_feature", f"{svc['name']}:{feature_flag}", "success", {"reason": reason, "health_improved": heals})
        return {
            "action": "disable_feature",
            "service": svc["name"],
            "feature_flag": feature_flag,
            "result": "completed",
            "health_improved": heals,
            "note": note,
            "after": self._health_snapshot(self.store.get_service(svc["name"])),
        }

    # ------------------------------------------------------------------ #
    # registry / router
    # ------------------------------------------------------------------ #
    def registry(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {name: getattr(self, name) for name in sorted(READ_ONLY_TOOLS | LOW_RISK_TOOLS | HIGH_RISK_TOOLS)}

    def catalog(self) -> list[dict[str, Any]]:
        """Describe every tool: name, risk level, description, parameters. Used for tool discovery."""
        out = []
        for name, fn in self.registry().items():
            sig = inspect.signature(fn)
            params = []
            for p in sig.parameters.values():
                params.append({"name": p.name, "required": p.default is inspect.Parameter.empty, "type": _annotation_name(p.annotation)})
            out.append({"name": name, "risk": tool_risk(name), "description": (fn.__doc__ or "").strip(), "parameters": params})
        return out

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route a tool call. Returns {"status": 200, "result": ...} or {"status": 4xx/5xx, "error": ...}."""
        arguments = arguments or {}
        tool = self.registry().get(name)
        if tool is None:
            return {"status": 404, "error": f"Unknown tool: {name}", "details": {"known_tools": sorted(self.registry())}}
        if not isinstance(arguments, dict):
            return {"status": 400, "error": "Tool arguments must be a JSON object."}
        try:
            inspect.signature(tool).bind(**arguments)
        except TypeError as exc:
            return {"status": 400, "error": f"Invalid arguments for {name}: {exc}"}
        try:
            return {"status": 200, "result": tool(**arguments)}
        except ToolError as exc:
            return exc.to_dict()
        except Exception as exc:  # defensive: never leak a Python traceback to the model
            return {"status": 500, "error": f"{type(exc).__name__}: {exc}"}


def tool_risk(name: str) -> str:
    if name in HIGH_RISK_TOOLS:
        return "high"
    if name in LOW_RISK_TOOLS:
        return "low"
    if name in READ_ONLY_TOOLS:
        return "read_only"
    return "unknown"


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    return str(annotation).replace("typing.", "")


_default_toolkit: ToolKit | None = None


def default_toolkit() -> ToolKit:
    """Process-wide ToolKit over an in-memory store seeded from ``data/``."""
    global _default_toolkit
    if _default_toolkit is None:
        _default_toolkit = ToolKit()
    return _default_toolkit


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Module-level router using the default toolkit."""
    return default_toolkit().execute(name, arguments)
