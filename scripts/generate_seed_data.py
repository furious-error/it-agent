"""Deterministically generate the simulated AcmeCloud infrastructure seed data.

Run:  python scripts/generate_seed_data.py

Writes JSON files into ``data/``. The files are committed, so this script only
needs to be re-run when the scenario design changes.

Simulated clock: "now" is 2026-09-30T10:20:00Z. Metrics cover 09:30-10:20 at
one-minute resolution.

Scenario map (see evaluation/scenarios.json for ground truth):
  INC-001 orders-api            bad deployment (deploy-8472, DB port change)
  INC-002 payments-api          postgres connection pool exhausted (leak in v4.2.0)
  INC-003 users-api             redis-cache down (OOM killed, corrupt AOF)
  INC-004 api-gateway           false alarm (traffic burst, already recovered)
  INC-005 auth-service          deployment stuck (readiness probe, missing env var)
  INC-006 notification-service  healthy; exercises tool-error recovery
  INC-007 orders-api            historical, resolved via approved rollback
  INC-008 payments-api          historical, restart rejected; real cause cert expiry
  INC-009 inventory-api         bad deployment whose rollback FAILS; feature flag fix
  INC-010 search-api            logs say DB timeout, DB metrics healthy (bad config)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RNG = random.Random(42)

DAY = datetime(2026, 9, 30, tzinfo=timezone.utc)
WINDOW_START = DAY.replace(hour=9, minute=30)
NOW = DAY.replace(hour=10, minute=20)
MINUTES = 51  # 09:30 .. 10:20 inclusive


def ts(hour: int, minute: int, second: int = 0, day: datetime = DAY) -> str:
    return day.replace(hour=hour, minute=minute, second=second).strftime("%Y-%m-%dT%H:%M:%SZ")


def minute_ts(m: int) -> str:
    return (WINDOW_START + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ")


def jitter(value: float, pct: float = 0.05, digits: int = 3) -> float:
    return round(value * (1 + RNG.uniform(-pct, pct)), digits)


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
# ``simulation`` is hidden from tool output. It drives how remediation tools
# behave: which actions actually heal the service, and which upstream service
# a degraded service is blocked by.
SERVICES = [
    {
        "service_id": "svc-users",
        "name": "users-api",
        "environment": "production",
        "version": "v5.0.2",
        "status": "degraded",
        "replicas": 3,
        "healthy_replicas": 3,
        "owner_team": "identity",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"session_cache": True},
        "simulation": {
            "baseline": {"cpu_percent": 45, "memory_percent": 55, "request_rate": 900, "error_rate": 0.004, "p95_latency_ms": 120},
            "blocked_by": "redis-cache",
            "remediations": {"restart_service": False, "rollback_deployment": None, "disable_feature": None},
        },
    },
    {
        "service_id": "svc-orders",
        "name": "orders-api",
        "environment": "production",
        "version": "v2.8.1",
        "status": "degraded",
        "replicas": 4,
        "healthy_replicas": 2,
        "owner_team": "commerce",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"new_checkout_flow": False},
        "simulation": {
            "baseline": {"cpu_percent": 48, "memory_percent": 60, "request_rate": 1200, "error_rate": 0.01, "p95_latency_ms": 180},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": "deploy-8472", "disable_feature": None},
        },
    },
    {
        "service_id": "svc-payments",
        "name": "payments-api",
        "environment": "production",
        "version": "v4.2.0",
        "status": "degraded",
        "replicas": 3,
        "healthy_replicas": 3,
        "owner_team": "payments",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"provider_retry_wrapper": True},
        "simulation": {
            "baseline": {"cpu_percent": 40, "memory_percent": 58, "request_rate": 350, "error_rate": 0.005, "p95_latency_ms": 250},
            "blocked_by": None,
            # Restart clears the leaked connections (a temporary fix the agent may take),
            # rollback removes the leaking code, disabling the retry wrapper also works.
            "remediations": {"restart_service": True, "rollback_deployment": "deploy-8480", "disable_feature": "provider_retry_wrapper"},
        },
    },
    {
        "service_id": "svc-auth",
        "name": "auth-service",
        "environment": "production",
        "version": "v3.1.4",
        "status": "deploying",
        "replicas": 3,
        "healthy_replicas": 3,
        "owner_team": "identity",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {},
        "rollout": {
            "deployment_id": "deploy-8490",
            "target_version": "v3.2.0",
            "updated_replicas": 1,
            "ready_replicas": 0,
            "status": "stuck",
            "message": "Progress deadline exceeded: 0/3 new pods ready after 10m",
        },
        "simulation": {
            "baseline": {"cpu_percent": 35, "memory_percent": 50, "request_rate": 1500, "error_rate": 0.002, "p95_latency_ms": 60},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": "deploy-8490", "disable_feature": None},
        },
    },
    {
        "service_id": "svc-notification",
        "name": "notification-service",
        "environment": "production",
        "version": "v2.3.1",
        "status": "healthy",
        "replicas": 2,
        "healthy_replicas": 2,
        "owner_team": "platform",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"sms_channel": True},
        "simulation": {
            "baseline": {"cpu_percent": 30, "memory_percent": 45, "request_rate": 200, "error_rate": 0.003, "p95_latency_ms": 90},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": None, "disable_feature": None},
        },
    },
    {
        "service_id": "svc-postgres",
        "name": "postgres-db",
        "environment": "production",
        "version": "15.6",
        "status": "healthy",
        "replicas": 3,
        "healthy_replicas": 3,
        "owner_team": "data-platform",
        "runtime": "vm/acme-prod-db",
        "feature_flags": {},
        "simulation": {
            "baseline": {"cpu_percent": 38, "memory_percent": 72, "request_rate": 4200, "error_rate": 0.0, "p95_latency_ms": 8},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": None, "disable_feature": None},
        },
    },
    {
        "service_id": "svc-redis",
        "name": "redis-cache",
        "environment": "production",
        "version": "7.2.4",
        "status": "down",
        "replicas": 1,
        "healthy_replicas": 0,
        "owner_team": "data-platform",
        "runtime": "vm/acme-prod-cache",
        "feature_flags": {},
        "simulation": {
            "baseline": {"cpu_percent": 25, "memory_percent": 70, "request_rate": 8000, "error_rate": 0.0, "p95_latency_ms": 1},
            "blocked_by": None,
            # Restart runs the AOF repair on boot and brings the node back.
            "remediations": {"restart_service": True, "rollback_deployment": None, "disable_feature": None},
        },
    },
    {
        "service_id": "svc-gateway",
        "name": "api-gateway",
        "environment": "production",
        "version": "v1.12.0",
        "status": "healthy",
        "replicas": 6,
        "healthy_replicas": 6,
        "owner_team": "platform",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"rate_limiting_v2": True},
        "simulation": {
            "baseline": {"cpu_percent": 50, "memory_percent": 48, "request_rate": 5200, "error_rate": 0.002, "p95_latency_ms": 90},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": None, "disable_feature": None},
        },
    },
    {
        "service_id": "svc-inventory",
        "name": "inventory-api",
        "environment": "production",
        "version": "v1.5.0",
        "status": "degraded",
        "replicas": 3,
        "healthy_replicas": 3,
        "owner_team": "commerce",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"async_stock_sync": True, "bulk_import": False},
        "simulation": {
            "baseline": {"cpu_percent": 42, "memory_percent": 52, "request_rate": 300, "error_rate": 0.005, "p95_latency_ms": 140},
            "blocked_by": None,
            # rollback of deploy-8501 fails (artifact pruned) - see deployments.
            "remediations": {"restart_service": False, "rollback_deployment": "deploy-8501", "disable_feature": "async_stock_sync"},
        },
    },
    {
        "service_id": "svc-search",
        "name": "search-api",
        "environment": "production",
        "version": "v1.9.3",
        "status": "degraded",
        "replicas": 2,
        "healthy_replicas": 2,
        "owner_team": "discovery",
        "runtime": "kubernetes/acme-prod",
        "feature_flags": {"semantic_ranking": False},
        "simulation": {
            "baseline": {"cpu_percent": 38, "memory_percent": 50, "request_rate": 600, "error_rate": 0.003, "p95_latency_ms": 110},
            "blocked_by": None,
            "remediations": {"restart_service": False, "rollback_deployment": "deploy-8497", "disable_feature": None},
        },
    },
]


# --------------------------------------------------------------------------- #
# Deployments
# --------------------------------------------------------------------------- #
YESTERDAY = DAY - timedelta(days=1)
TWO_DAYS_AGO = DAY - timedelta(days=2)

DEPLOYMENTS = [
    # --- historical -------------------------------------------------------- #
    {
        "deployment_id": "deploy-8441",
        "service": "payments-api",
        "version": "v4.1.7",
        "previous_version": "v4.1.6",
        "environment": "production",
        "deployed_at": ts(14, 0, day=TWO_DAYS_AGO),
        "completed_at": ts(14, 4, day=TWO_DAYS_AGO),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "a81f2c0",
        "change_summary": "Minor logging improvements in provider client.",
        "changes": ["Add structured logging for PaymentProvider responses"],
    },
    {
        "deployment_id": "deploy-8455",
        "service": "users-api",
        "version": "v5.0.2",
        "previous_version": "v5.0.1",
        "environment": "production",
        "deployed_at": ts(11, 0, day=YESTERDAY),
        "completed_at": ts(11, 3, day=YESTERDAY),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "c0ffee1",
        "change_summary": "Profile endpoint pagination fix.",
        "changes": ["Fix off-by-one in /api/users?page= pagination"],
    },
    {
        "deployment_id": "deploy-8460",
        "service": "orders-api",
        "version": "v2.7.9",
        "previous_version": "v2.7.8",
        "environment": "production",
        "deployed_at": ts(8, 0, day=YESTERDAY),
        "completed_at": ts(8, 3, day=YESTERDAY),
        "status": "rolled_back",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "9d1e77b",
        "change_summary": "Introduce order status enum; caused HTTP 500 on legacy statuses (INC-007).",
        "changes": ["Replace free-text order status with OrderStatus enum", "Strict validation of status on read path"],
        "rolled_back_by": "deploy-8461",
    },
    {
        "deployment_id": "deploy-8461",
        "service": "orders-api",
        "version": "v2.7.8",
        "previous_version": "v2.7.9",
        "environment": "production",
        "deployed_at": ts(8, 35, day=YESTERDAY),
        "completed_at": ts(8, 37, day=YESTERDAY),
        "status": "successful",
        "type": "rollback",
        "deployed_by": "sre-oncall (approved by: priya.n)",
        "commit": "4b2a9f0",
        "change_summary": "Rollback of deploy-8460 for INC-007.",
        "changes": ["Revert to v2.7.8"],
        "rollback_of": "deploy-8460",
    },
    {
        "deployment_id": "deploy-8465",
        "service": "orders-api",
        "version": "v2.8.0",
        "previous_version": "v2.7.8",
        "environment": "production",
        "deployed_at": ts(16, 0, day=YESTERDAY),
        "completed_at": ts(16, 3, day=YESTERDAY),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "e3c1a88",
        "change_summary": "Re-land order status enum with backward-compatible legacy status mapping.",
        "changes": ["OrderStatus enum with legacy fallback", "Add contract tests for legacy statuses"],
    },
    {
        "deployment_id": "deploy-8468",
        "service": "api-gateway",
        "version": "v1.12.0",
        "previous_version": "v1.11.4",
        "environment": "production",
        "deployed_at": ts(18, 0, day=YESTERDAY),
        "completed_at": ts(18, 6, day=YESTERDAY),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "77ab3d2",
        "change_summary": "Enable rate_limiting_v2 and upgrade envoy base image.",
        "changes": ["rate_limiting_v2 flag on", "envoy 1.31 base image"],
    },
    # --- today ------------------------------------------------------------- #
    {
        "deployment_id": "deploy-8470",
        "service": "notification-service",
        "version": "v2.3.1",
        "previous_version": "v2.3.0",
        "environment": "production",
        "deployed_at": ts(7, 0),
        "completed_at": ts(7, 2),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "5e5e5e5",
        "change_summary": "Template rendering cache.",
        "changes": ["Cache compiled email templates"],
    },
    {
        "deployment_id": "deploy-8480",
        "service": "payments-api",
        "version": "v4.2.0",
        "previous_version": "v4.1.7",
        "environment": "production",
        "deployed_at": ts(8, 30),
        "completed_at": ts(8, 34),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "b7d44e1",
        "change_summary": "Add retry wrapper around PaymentProvider client (3 attempts, exponential backoff).",
        "changes": [
            "New ProviderRetryWrapper (feature flag provider_retry_wrapper)",
            "Charge handler acquires DB connection before calling provider",
            "Bump provider client timeout 2s -> 8s",
        ],
    },
    {
        "deployment_id": "deploy-8497",
        "service": "search-api",
        "version": "v1.9.3",
        "previous_version": "v1.9.2",
        "environment": "production",
        "deployed_at": ts(9, 50),
        "completed_at": ts(9, 51),
        "status": "successful",
        "type": "config",
        "deployed_by": "arjun.m",
        "commit": "config-2219",
        "change_summary": "Config-only change: point read queries at dedicated replica.",
        "changes": [
            "DB_READ_HOST: pg-replica-02.internal -> pg-replica-old.internal",
            "DB_CONNECT_TIMEOUT_MS: 5000 -> 10000",
        ],
    },
    {
        "deployment_id": "deploy-8490",
        "service": "auth-service",
        "version": "v3.2.0",
        "previous_version": "v3.1.4",
        "environment": "production",
        "deployed_at": ts(9, 55),
        "completed_at": None,
        "status": "in_progress",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "f00dbab",
        "change_summary": "Rotate to v2 JWT signing key; requires JWT_SIGNING_KEY_V2 secret.",
        "changes": [
            "Load JWT_SIGNING_KEY_V2 at startup (required)",
            "Dual-sign tokens during rotation window",
        ],
        "rollout": {"updated_replicas": 1, "ready_replicas": 0, "progress_deadline_exceeded": True},
    },
    {
        "deployment_id": "deploy-8501",
        "service": "inventory-api",
        "version": "v1.5.0",
        "previous_version": "v1.4.9",
        "environment": "production",
        "deployed_at": ts(9, 58),
        "completed_at": ts(10, 0),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "0ddba11",
        "change_summary": "Async stock synchronisation worker (feature flag async_stock_sync).",
        "changes": [
            "StockSyncWorker writes stock_levels in background batches",
            "Enable async_stock_sync flag by default",
        ],
        "simulation": {
            "rollback_error": "Rollback failed: artifact registry.acmecloud.internal/inventory-api:v1.4.9 not found (pruned by image retention policy)"
        },
    },
    {
        "deployment_id": "deploy-8472",
        "service": "orders-api",
        "version": "v2.8.1",
        "previous_version": "v2.8.0",
        "environment": "production",
        "deployed_at": ts(10, 0),
        "completed_at": ts(10, 3, 30),
        "status": "successful",
        "type": "release",
        "deployed_by": "ci-pipeline",
        "commit": "deadb33f",
        "change_summary": "Migrate OrderRepository to transaction-mode connection pooling via PgBouncer.",
        "changes": [
            "DB_POOL_MODE: session -> transaction",
            "DB_PORT: 5432 -> 6432 (PgBouncer)",
            "Remove per-request connection warmup",
        ],
    },
]


# --------------------------------------------------------------------------- #
# Databases
# --------------------------------------------------------------------------- #
DATABASES = [
    {"database_id": "db-orders", "name": "orders-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-primary.internal", "port": 5432,
     "status": "healthy", "connections_active": 42, "connections_max": 200, "p95_query_ms": 9, "replication_lag_ms": 0, "disk_used_percent": 61, "last_heartbeat": ts(10, 19, 55)},
    {"database_id": "db-payments", "name": "payments-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-primary.internal", "port": 5432,
     "status": "degraded", "connections_active": 200, "connections_max": 200, "p95_query_ms": 12, "replication_lag_ms": 0, "disk_used_percent": 58, "last_heartbeat": ts(10, 19, 55),
     "warnings": ["Connection limit reached for role payments_app (200/200)", "163 connections idle in transaction > 5m"]},
    {"database_id": "db-users", "name": "users-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-primary.internal", "port": 5432,
     "status": "healthy", "connections_active": 71, "connections_max": 200, "p95_query_ms": 14, "replication_lag_ms": 0, "disk_used_percent": 55, "last_heartbeat": ts(10, 19, 55),
     "warnings": ["Query volume +240% since 09:52 (session lookups falling back from cache)"]},
    {"database_id": "db-auth", "name": "auth-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-primary.internal", "port": 5432,
     "status": "healthy", "connections_active": 23, "connections_max": 100, "p95_query_ms": 6, "replication_lag_ms": 0, "disk_used_percent": 40, "last_heartbeat": ts(10, 19, 55)},
    {"database_id": "db-inventory", "name": "inventory-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-primary.internal", "port": 5432,
     "status": "healthy", "connections_active": 38, "connections_max": 100, "p95_query_ms": 31, "replication_lag_ms": 0, "disk_used_percent": 66, "last_heartbeat": ts(10, 19, 55),
     "warnings": ["14 deadlocks detected on table stock_levels since 10:00"]},
    {"database_id": "db-search", "name": "search-db", "engine": "postgres", "cluster": "postgres-db", "host": "pg-replica-02.internal", "port": 5432,
     "status": "healthy", "connections_active": 3, "connections_max": 100, "p95_query_ms": 7, "replication_lag_ms": 40, "disk_used_percent": 52, "last_heartbeat": ts(10, 19, 55),
     "notes": ["Read replica. Connections dropped from 18 to 3 at 09:51."]},
    {"database_id": "db-redis", "name": "redis-cache", "engine": "redis", "cluster": "redis-cache", "host": "redis-cache.internal", "port": 6379,
     "status": "down", "connections_active": 0, "connections_max": 10000, "p95_query_ms": None, "replication_lag_ms": None, "disk_used_percent": 88, "last_heartbeat": ts(9, 51, 2),
     "warnings": ["No heartbeat since 09:51:02", "Last exit: OOMKilled (137)", "AOF appendonly.aof failed validation on restart attempt"]},
]


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
DEPENDENCIES = [
    {"service": "api-gateway", "depends_on": [
        {"name": "users-api", "type": "service"}, {"name": "orders-api", "type": "service"}, {"name": "payments-api", "type": "service"},
        {"name": "auth-service", "type": "service"}, {"name": "search-api", "type": "service"}, {"name": "inventory-api", "type": "service"}]},
    {"service": "users-api", "depends_on": [{"name": "users-db", "type": "database"}, {"name": "redis-cache", "type": "cache"}]},
    {"service": "orders-api", "depends_on": [
        {"name": "orders-db", "type": "database"}, {"name": "payments-api", "type": "service"},
        {"name": "inventory-api", "type": "service"}, {"name": "notification-service", "type": "service"}]},
    {"service": "payments-api", "depends_on": [{"name": "payments-db", "type": "database"}, {"name": "payment-provider", "type": "external"}]},
    {"service": "auth-service", "depends_on": [{"name": "auth-db", "type": "database"}]},
    {"service": "notification-service", "depends_on": [{"name": "notification-queue", "type": "queue"}, {"name": "email-provider", "type": "external"}]},
    {"service": "inventory-api", "depends_on": [{"name": "inventory-db", "type": "database"}]},
    {"service": "search-api", "depends_on": [{"name": "search-db", "type": "database"}]},
    {"service": "postgres-db", "depends_on": []},
    {"service": "redis-cache", "depends_on": []},
]


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
ALERTS = [
    {"alert_id": "ALT-3001", "name": "HighErrorRate", "service": "orders-api", "severity": "critical", "status": "firing",
     "fired_at": ts(10, 4), "resolved_at": None, "incident_id": "INC-001",
     "summary": "orders-api 5xx error rate 35% > threshold 5% for 2m"},
    {"alert_id": "ALT-3002", "name": "HighLatencyP95", "service": "payments-api", "severity": "warning", "status": "firing",
     "fired_at": ts(9, 53), "resolved_at": None, "incident_id": "INC-002",
     "summary": "payments-api p95 latency 1.2s > threshold 800ms for 5m"},
    {"alert_id": "ALT-3003", "name": "RedisDown", "service": "redis-cache", "severity": "critical", "status": "firing",
     "fired_at": ts(9, 52), "resolved_at": None, "incident_id": "INC-003",
     "summary": "redis-cache exporter target down for 1m"},
    {"alert_id": "ALT-3004", "name": "HighErrorRate", "service": "users-api", "severity": "warning", "status": "firing",
     "fired_at": ts(9, 55), "resolved_at": None, "incident_id": "INC-003",
     "summary": "users-api 5xx error rate 18% > threshold 5% for 2m"},
    {"alert_id": "ALT-3005", "name": "HighLatencyP95", "service": "api-gateway", "severity": "warning", "status": "resolved",
     "fired_at": ts(9, 58), "resolved_at": ts(10, 1), "incident_id": "INC-004",
     "summary": "api-gateway p95 latency 1.3s > threshold 500ms for 1m"},
    {"alert_id": "ALT-3006", "name": "DeploymentStuck", "service": "auth-service", "severity": "warning", "status": "firing",
     "fired_at": ts(10, 5, 30), "resolved_at": None, "incident_id": "INC-005",
     "summary": "deploy-8490 progress deadline exceeded (0/3 ready)"},
    {"alert_id": "ALT-3007", "name": "HighErrorRate", "service": "inventory-api", "severity": "warning", "status": "firing",
     "fired_at": ts(10, 3), "resolved_at": None, "incident_id": "INC-009",
     "summary": "inventory-api 5xx error rate 12% > threshold 5% for 2m"},
    {"alert_id": "ALT-3008", "name": "DatabaseConnectionTimeouts", "service": "search-api", "severity": "critical", "status": "firing",
     "fired_at": ts(9, 53), "resolved_at": None, "incident_id": "INC-010",
     "summary": "search-api DB connection timeouts > 100/min"},
    {"alert_id": "ALT-3009", "name": "EmailProviderThrottled", "service": "notification-service", "severity": "info", "status": "resolved",
     "fired_at": ts(9, 34), "resolved_at": ts(9, 41), "incident_id": None,
     "summary": "email-provider returned 429 for 7m; backlog drained"},
]


# --------------------------------------------------------------------------- #
# Incidents
# --------------------------------------------------------------------------- #
INCIDENTS = [
    {
        "incident_id": "INC-001",
        "title": "orders-api returning HTTP 500",
        "severity": "SEV-1",
        "service": "orders-api",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(10, 2),
        "detected_at": ts(10, 4),
        "detected_by": "alert:ALT-3001",
        "reporter": "pagerduty",
        "description": "Orders API returning HTTP 500 errors to customers. Checkout conversion dropped sharply. Started roughly 10 minutes before this incident was opened.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [
            {"timestamp": ts(10, 4), "event": "Alert HighErrorRate fired for orders-api"},
            {"timestamp": ts(10, 4, 30), "event": "Incident INC-001 created automatically"},
        ],
    },
    {
        "incident_id": "INC-002",
        "title": "payments-api latency increasing",
        "severity": "SEV-2",
        "service": "payments-api",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(9, 50),
        "detected_at": ts(9, 53),
        "detected_by": "alert:ALT-3002",
        "reporter": "pagerduty",
        "description": "p95 latency on payments-api has been climbing steadily for the last 30 minutes; some charge requests now time out with HTTP 504.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [{"timestamp": ts(9, 53), "event": "Alert HighLatencyP95 fired for payments-api"}],
    },
    {
        "incident_id": "INC-003",
        "title": "users-api intermittently returning 503",
        "severity": "SEV-2",
        "service": "users-api",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(9, 52),
        "detected_at": ts(9, 55),
        "detected_by": "alert:ALT-3004",
        "reporter": "pagerduty",
        "description": "Roughly one in five requests to users-api fail with HTTP 503. Users are being logged out unexpectedly.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [
            {"timestamp": ts(9, 52), "event": "Alert RedisDown fired for redis-cache"},
            {"timestamp": ts(9, 55), "event": "Alert HighErrorRate fired for users-api"},
        ],
    },
    {
        "incident_id": "INC-004",
        "title": "API latency alert triggered",
        "severity": "SEV-3",
        "service": "api-gateway",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(9, 58),
        "detected_at": ts(9, 58),
        "detected_by": "alert:ALT-3005",
        "reporter": "pagerduty",
        "description": "HighLatencyP95 alert fired for api-gateway. Automatic incident; impact not yet assessed.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [
            {"timestamp": ts(9, 58), "event": "Alert HighLatencyP95 fired for api-gateway"},
            {"timestamp": ts(10, 1), "event": "Alert HighLatencyP95 resolved"},
        ],
    },
    {
        "incident_id": "INC-005",
        "title": "auth-service deployment stuck",
        "severity": "SEV-3",
        "service": "auth-service",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(9, 55),
        "detected_at": ts(10, 5, 30),
        "detected_by": "alert:ALT-3006",
        "reporter": "ci-pipeline",
        "description": "Deployment deploy-8490 (v3.2.0) of auth-service has not completed. New pods are not becoming ready. Old pods still serving traffic.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [{"timestamp": ts(10, 5, 30), "event": "Alert DeploymentStuck fired for auth-service"}],
    },
    {
        "incident_id": "INC-006",
        "title": "Customers report delayed email notifications",
        "severity": "SEV-3",
        "service": "notification-service",
        "environment": "production",
        "status": "open",
        "started_at": ts(9, 35),
        "detected_at": ts(10, 8),
        "detected_by": "support-ticket:TCK-5521",
        "reporter": "support-team",
        "description": "Several customers reported that order confirmation emails arrived 10-15 minutes late this morning. Please confirm whether there is an ongoing problem with notification-service.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [{"timestamp": ts(10, 8), "event": "Incident created from support ticket TCK-5521"}],
    },
    {
        "incident_id": "INC-007",
        "title": "orders-api HTTP 500 after release v2.7.9",
        "severity": "SEV-1",
        "service": "orders-api",
        "environment": "production",
        "status": "resolved",
        "started_at": ts(8, 3, day=YESTERDAY),
        "detected_at": ts(8, 6, day=YESTERDAY),
        "detected_by": "alert:HighErrorRate",
        "reporter": "pagerduty",
        "description": "orders-api returned HTTP 500 for orders with legacy status values immediately after deploy-8460.",
        "root_cause": "deploy-8460 (v2.7.9) introduced a strict OrderStatus enum that rejected legacy status strings still present in orders-db.",
        "resolution": "Rolled back to v2.7.8 via deploy-8461 after human approval. Fix re-landed as v2.8.0 with legacy mapping.",
        "resolved_at": ts(8, 40, day=YESTERDAY),
        "notes": [
            {"timestamp": ts(8, 20, day=YESTERDAY), "author": "sre-agent", "note": "Errors correlate with deploy-8460; logs show OrderStatus validation errors."},
            {"timestamp": ts(8, 33, day=YESTERDAY), "author": "priya.n", "note": "Rollback approved."},
        ],
        "timeline": [
            {"timestamp": ts(8, 6, day=YESTERDAY), "event": "Alert HighErrorRate fired"},
            {"timestamp": ts(8, 33, day=YESTERDAY), "event": "Rollback of deploy-8460 approved by priya.n"},
            {"timestamp": ts(8, 37, day=YESTERDAY), "event": "deploy-8461 completed"},
            {"timestamp": ts(8, 40, day=YESTERDAY), "event": "Service healthy; incident resolved"},
        ],
    },
    {
        "incident_id": "INC-008",
        "title": "payments-api checkout failures",
        "severity": "SEV-2",
        "service": "payments-api",
        "environment": "production",
        "status": "resolved",
        "started_at": ts(2, 0, day=TWO_DAYS_AGO),
        "detected_at": ts(2, 4, day=TWO_DAYS_AGO),
        "detected_by": "alert:HighErrorRate",
        "reporter": "pagerduty",
        "description": "All charge requests failing with provider errors.",
        "root_cause": "The mTLS client certificate used to call payment-provider expired at 02:00 UTC.",
        "resolution": "Rotated client certificate. A proposed restart of payments-api was rejected by the on-call engineer because logs showed a TLS handshake failure, not a service fault.",
        "resolved_at": ts(2, 48, day=TWO_DAYS_AGO),
        "notes": [
            {"timestamp": ts(2, 15, day=TWO_DAYS_AGO), "author": "sre-agent", "note": "Proposed restart_service(payments-api)."},
            {"timestamp": ts(2, 16, day=TWO_DAYS_AGO), "author": "marco.b", "note": "Rejected: logs show 'certificate has expired' from provider handshake. Restart will not help."},
            {"timestamp": ts(2, 30, day=TWO_DAYS_AGO), "author": "sre-agent", "note": "Confirmed client cert expiry; escalated to payments team for cert rotation."},
        ],
        "timeline": [
            {"timestamp": ts(2, 4, day=TWO_DAYS_AGO), "event": "Alert fired"},
            {"timestamp": ts(2, 16, day=TWO_DAYS_AGO), "event": "Restart proposal rejected"},
            {"timestamp": ts(2, 45, day=TWO_DAYS_AGO), "event": "Certificate rotated"},
            {"timestamp": ts(2, 48, day=TWO_DAYS_AGO), "event": "Incident resolved"},
        ],
    },
    {
        "incident_id": "INC-009",
        "title": "inventory-api HTTP 500 on stock updates",
        "severity": "SEV-2",
        "service": "inventory-api",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(10, 1),
        "detected_at": ts(10, 3),
        "detected_by": "alert:ALT-3007",
        "reporter": "pagerduty",
        "description": "PUT /api/inventory/{sku} intermittently returns HTTP 500. Warehouse integrations are retrying and amplifying load.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [{"timestamp": ts(10, 3), "event": "Alert HighErrorRate fired for inventory-api"}],
    },
    {
        "incident_id": "INC-010",
        "title": "search-api database connection timeouts",
        "severity": "SEV-2",
        "service": "search-api",
        "environment": "production",
        "status": "investigating",
        "started_at": ts(9, 51),
        "detected_at": ts(9, 53),
        "detected_by": "alert:ALT-3008",
        "reporter": "pagerduty",
        "description": "search-api logs are full of database connection timeouts and roughly half of search requests fail. The database team reports postgres is healthy.",
        "root_cause": None,
        "resolution": None,
        "resolved_at": None,
        "notes": [],
        "timeline": [{"timestamp": ts(9, 53), "event": "Alert DatabaseConnectionTimeouts fired for search-api"}],
    },
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metric_profile(service: str, m: int) -> dict | None:
    """Return a metric sample for service at minute offset m (0 = 09:30), or None if the scrape failed."""
    b = next(s for s in SERVICES if s["name"] == service)["simulation"]["baseline"]
    cpu, mem, rps, err, p95 = b["cpu_percent"], b["memory_percent"], b["request_rate"], b["error_rate"], b["p95_latency_ms"]

    if service == "orders-api":  # deploy at 10:00 (m=30), errors ramp
        ramp = {30: (0.01, 180), 31: (0.02, 260), 32: (0.08, 600), 33: (0.21, 1200), 34: (0.35, 1700)}
        if m in ramp:
            err, p95 = ramp[m]
        elif m > 34:
            err, p95, cpu, mem = 0.37, 1800, 71, 68
    elif service == "payments-api":  # pool exhaustion ramp from 09:45 (m=15)
        if m >= 15:
            k = m - 15
            p95 = 250 + k * 115
            err = min(0.06, 0.005 + k * 0.0016)
            cpu = 40 + k * 0.4
    elif service == "users-api":  # redis down at 09:51 (m=21), impact from 09:52
        if m >= 22:
            err, p95, cpu = 0.18, 650, 60
    elif service == "redis-cache":
        if m >= 21:
            return None  # exporter down
        mem = 85 + (m / 20) * 14  # climbing to ~99%
    elif service == "api-gateway":  # burst 09:57-10:00 (m=27..30), recovered
        burst = {27: (15000, 900), 28: (15600, 1300), 29: (14800, 1250), 30: (9000, 400)}
        if m in burst:
            rps, p95 = burst[m]
            cpu = 82
    elif service == "notification-service":  # throttled 09:33-09:40 (m=3..10), drained by 09:43
        if 3 <= m <= 10:
            rps, p95 = 40, 400
        elif 11 <= m <= 13:
            rps, p95 = 900, 150
    elif service == "inventory-api":  # deploy 09:58, errors from 10:00 (m=30)
        ramp = {30: (0.02, 220), 31: (0.06, 400), 32: (0.12, 650), 33: (0.18, 820), 34: (0.22, 900)}
        if m in ramp:
            err, p95 = ramp[m]
            cpu = 65
        elif m > 34:
            err, p95, cpu = 0.22, 900, 78
    elif service == "search-api":  # config deploy 09:50, timeouts from 09:51 (m=21)
        if m >= 21:
            err, p95, rps = 0.45, 10200, 450
    elif service == "postgres-db":
        if m >= 20:
            cpu = 42

    return {
        "service": service,
        "timestamp": minute_ts(m),
        "cpu_percent": round(jitter(cpu, 0.04, 1), 1),
        "memory_percent": round(jitter(mem, 0.02, 1), 1),
        "request_rate": int(jitter(rps, 0.05, 0)),
        "error_rate": round(jitter(err, 0.08, 4), 4) if err else 0.0,
        "p95_latency_ms": int(jitter(p95, 0.06, 0)),
    }


def build_metrics() -> list[dict]:
    out = []
    for svc in SERVICES:
        for m in range(MINUTES):
            sample = metric_profile(svc["name"], m)
            if sample:
                out.append(sample)
    out.sort(key=lambda r: (r["timestamp"], r["service"]))
    return out


# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #
def build_logs() -> list[dict]:
    L: list[tuple[str, str, str, str, str]] = []  # (timestamp, service, level, message, source)

    def add(t: str, service: str, level: str, message: str, source: str | None = None):
        L.append((t, service, level, message, source or f"{service}-pod"))

    # --- INC-007 / INC-008 history (sparse) -------------------------------- #
    add(ts(2, 1, 10, day=TWO_DAYS_AGO), "payments-api", "ERROR", "PaymentProvider TLS handshake failed: certificate has expired (client cert CN=payments-api.acmecloud)")
    add(ts(2, 1, 11, day=TWO_DAYS_AGO), "payments-api", "ERROR", "HTTP 502 POST /api/payments/charge - provider unavailable")
    add(ts(2, 46, 0, day=TWO_DAYS_AGO), "payments-api", "INFO", "Client certificate rotated; provider handshake OK")
    add(ts(8, 0, 0, day=YESTERDAY), "orders-api", "INFO", "Deployment deploy-8460 started (v2.7.8 -> v2.7.9)", "ci-pipeline")
    add(ts(8, 4, 12, day=YESTERDAY), "orders-api", "ERROR", "HTTP 500 GET /api/orders/551 - ValueError: 'SHIPPED_PARTIAL' is not a valid OrderStatus")
    add(ts(8, 35, 0, day=YESTERDAY), "orders-api", "INFO", "Deployment deploy-8461 started (rollback v2.7.9 -> v2.7.8)", "ci-pipeline")
    add(ts(8, 37, 0, day=YESTERDAY), "orders-api", "INFO", "Deployment deploy-8461 completed: 4/4 pods running v2.7.8", "ci-pipeline")
    add(ts(16, 3, 0, day=YESTERDAY), "orders-api", "INFO", "Deployment deploy-8465 completed: 4/4 pods running v2.8.0", "ci-pipeline")

    # --- today: early deployments ----------------------------------------- #
    add(ts(7, 0), "notification-service", "INFO", "Deployment deploy-8470 started (v2.3.0 -> v2.3.1)", "ci-pipeline")
    add(ts(7, 2), "notification-service", "INFO", "Deployment deploy-8470 completed: 2/2 pods running v2.3.1", "ci-pipeline")
    add(ts(8, 30), "payments-api", "INFO", "Deployment deploy-8480 started (v4.1.7 -> v4.2.0)", "ci-pipeline")
    add(ts(8, 34), "payments-api", "INFO", "Deployment deploy-8480 completed: 3/3 pods running v4.2.0", "ci-pipeline")
    add(ts(8, 34, 5), "payments-api", "INFO", "Feature flag provider_retry_wrapper=true; ProviderRetryWrapper enabled (max_attempts=3)")

    # --- INC-006 notification-service: transient throttling, recovered ---- #
    add(ts(9, 33, 10), "notification-service", "WARN", "email-provider responded 429 Too Many Requests; backing off 30s")
    add(ts(9, 34, 2), "notification-service", "WARN", "email-provider responded 429 Too Many Requests; backing off 60s")
    add(ts(9, 36, 0), "notification-service", "WARN", "Queue depth 6,200 (threshold 5,000) - deliveries delayed")
    add(ts(9, 38, 0), "notification-service", "WARN", "Queue depth 12,400 - deliveries delayed by ~9m")
    add(ts(9, 40, 30), "notification-service", "INFO", "email-provider accepting requests again; resuming full send rate")
    add(ts(9, 43, 5), "notification-service", "INFO", "Backlog drained; queue depth 0; delivery latency back to 1.8s")

    # --- INC-002 payments-api: pool exhaustion ----------------------------- #
    add(ts(9, 40, 12), "payments-api", "WARN", "Connection pool utilization 85% (170/200) pool=payments-db")
    add(ts(9, 44, 8), "payments-api", "WARN", "ProviderRetryWrapper: attempt 2/3 for txn_7f31 after timeout (DB connection held during retry)")
    add(ts(9, 47, 41), "payments-api", "WARN", "Connection pool utilization 95% (190/200) pool=payments-db")
    add(ts(9, 51, 3), "payments-api", "WARN", "Connection pool utilization 100% (200/200) pool=payments-db")
    add(ts(9, 52, 15), "payments-api", "ERROR", "Timed out waiting for connection from pool after 5000ms (pool=payments-db, active=200, idle=0, max=200)")
    add(ts(9, 52, 16), "payments-api", "ERROR", "HTTP 504 POST /api/payments/charge - PoolTimeoutError")
    add(ts(9, 55, 0), "postgres-db", "WARN", "payments_db: connection limit reached for role payments_app (200/200); 158 connections 'idle in transaction'", "pg-primary")
    add(ts(9, 58, 44), "payments-api", "ERROR", "Timed out waiting for connection from pool after 5000ms (pool=payments-db, active=200, idle=0, max=200)")
    add(ts(10, 3, 21), "payments-api", "ERROR", "HTTP 504 POST /api/payments/charge - PoolTimeoutError")
    add(ts(10, 9, 2), "payments-api", "WARN", "ProviderRetryWrapper: attempt 3/3 for txn_8a02 after timeout (DB connection held during retry)")
    add(ts(10, 12, 57), "payments-api", "ERROR", "Timed out waiting for connection from pool after 5000ms (pool=payments-db, active=200, idle=0, max=200)")
    add(ts(10, 17, 30), "postgres-db", "WARN", "payments_db: connection limit reached for role payments_app (200/200); 163 connections 'idle in transaction'", "pg-primary")

    # --- INC-010 search-api: config points at decommissioned host --------- #
    add(ts(9, 50, 0), "search-api", "INFO", "Deployment deploy-8497 started (config change: DB_READ_HOST, DB_CONNECT_TIMEOUT_MS)", "arjun.m")
    add(ts(9, 50, 40), "search-api", "INFO", "Pod search-api-6b7c-r1 starting v1.9.3; DB_READ_HOST=pg-replica-old.internal DB_CONNECT_TIMEOUT_MS=10000")
    add(ts(9, 51, 0), "search-api", "INFO", "Deployment deploy-8497 completed: 2/2 pods running v1.9.3", "arjun.m")
    add(ts(9, 51, 11), "search-api", "ERROR", "database connection timeout after 10000ms connecting to pg-replica-old.internal:5432")
    add(ts(9, 51, 12), "search-api", "ERROR", "HTTP 500 GET /api/search?q=shoes - DBConnectionTimeout")
    add(ts(9, 53, 40), "search-api", "ERROR", "database connection timeout after 10000ms connecting to pg-replica-old.internal:5432")
    add(ts(9, 56, 2), "search-api", "WARN", "Circuit breaker 'search-db' half-open; probing")
    add(ts(10, 1, 19), "search-api", "ERROR", "database connection timeout after 10000ms connecting to pg-replica-old.internal:5432")
    add(ts(10, 7, 45), "search-api", "ERROR", "HTTP 500 GET /api/search?q=jacket - DBConnectionTimeout")
    add(ts(10, 14, 3), "search-api", "ERROR", "database connection timeout after 10000ms connecting to pg-replica-old.internal:5432")
    add(ts(10, 15, 0), "postgres-db", "INFO", "pg-replica-02: 3 active connections, replication lag 40ms, no slow queries", "pg-replica-02")

    # --- INC-003 redis-cache down -> users-api 503 ------------------------- #
    add(ts(9, 50, 40), "redis-cache", "WARN", "used_memory 3.98GB approaching maxmemory 4.00GB; eviction policy noeviction", "redis-cache-0")
    add(ts(9, 51, 2), "redis-cache", "ERROR", "OOM command not allowed when used memory > 'maxmemory'", "redis-cache-0")
    add(ts(9, 51, 5), "redis-cache", "ERROR", "Process terminated: OOMKilled (exit code 137)", "redis-cache-0")
    add(ts(9, 51, 20), "redis-cache", "ERROR", "Failed to start: Bad file format reading the append only file appendonly.aof: truncated record at offset 4211880960", "redis-cache-0")
    add(ts(9, 51, 25), "redis-cache", "ERROR", "Process exited (1); restart backoff 5m", "redis-cache-0")
    add(ts(9, 51, 30), "users-api", "ERROR", "Redis connection refused redis-cache.internal:6379")
    add(ts(9, 52, 1), "users-api", "ERROR", "HTTP 503 GET /api/users/me - SessionStoreUnavailable")
    add(ts(9, 52, 2), "users-api", "WARN", "Falling back to database session lookup (slow path) for 40% of requests")
    add(ts(9, 57, 44), "users-api", "ERROR", "HTTP 503 GET /api/users/me - SessionStoreUnavailable")
    add(ts(10, 4, 10), "users-api", "ERROR", "Redis connection refused redis-cache.internal:6379")
    add(ts(10, 11, 28), "users-api", "ERROR", "HTTP 503 POST /api/users/login - SessionStoreUnavailable")
    add(ts(10, 16, 25), "redis-cache", "ERROR", "Failed to start: Bad file format reading the append only file appendonly.aof: truncated record at offset 4211880960", "redis-cache-0")

    # --- INC-005 auth-service stuck rollout -------------------------------- #
    add(ts(9, 55, 0), "auth-service", "INFO", "Deployment deploy-8490 started (v3.1.4 -> v3.2.0)", "ci-pipeline")
    add(ts(9, 55, 30), "auth-service", "INFO", "Pod auth-service-5c6b-a1 starting v3.2.0", "auth-service-5c6b-a1")
    add(ts(9, 55, 35), "auth-service", "ERROR", "Startup check failed: required environment variable JWT_SIGNING_KEY_V2 is not set", "auth-service-5c6b-a1")
    add(ts(9, 55, 40), "auth-service", "WARN", "Readiness probe failed: GET /healthz/ready -> 503 (signing key not loaded)", "auth-service-5c6b-a1")
    add(ts(9, 56, 10), "auth-service", "WARN", "Readiness probe failed: GET /healthz/ready -> 503 (signing key not loaded)", "auth-service-5c6b-a1")
    add(ts(9, 58, 40), "auth-service", "WARN", "Readiness probe failed: GET /healthz/ready -> 503 (signing key not loaded)", "auth-service-5c6b-a1")
    add(ts(10, 2, 40), "auth-service", "WARN", "Readiness probe failed: GET /healthz/ready -> 503 (signing key not loaded)", "auth-service-5c6b-a1")
    add(ts(10, 5, 30), "auth-service", "ERROR", "Deployment deploy-8490 progress deadline exceeded (0/3 new pods ready after 10m); rollout paused", "ci-pipeline")
    add(ts(10, 10, 0), "auth-service", "INFO", "Serving traffic from 3/3 pods on v3.1.4; token issuance nominal", "auth-service-7f8d-b2")

    # --- INC-004 api-gateway latency burst, recovered --------------------- #
    add(ts(9, 57, 5), "api-gateway", "WARN", "Upstream latency spike: p95 900ms; request rate 15.0k rps (2.9x baseline); top client: batch-export-job")
    add(ts(9, 58, 0), "api-gateway", "WARN", "Upstream latency spike: p95 1300ms; request rate 15.6k rps; rate_limiting_v2 throttled 1,840 requests from batch-export-job")
    add(ts(10, 0, 30), "api-gateway", "INFO", "Request rate returning to baseline (9.0k rps)")
    add(ts(10, 1, 0), "api-gateway", "INFO", "Latency back to baseline: p95 95ms; 5xx rate 0.2%")

    # --- INC-009 inventory-api bad deploy, rollback will fail ------------- #
    add(ts(9, 58, 0), "inventory-api", "INFO", "Deployment deploy-8501 started (v1.4.9 -> v1.5.0)", "ci-pipeline")
    add(ts(9, 59, 10), "inventory-api", "INFO", "Pod inventory-api-4e2a-s1 starting v1.5.0; feature flag async_stock_sync=true; StockSyncWorker started (batch=500)")
    add(ts(10, 0, 0), "inventory-api", "INFO", "Deployment deploy-8501 completed: 3/3 pods running v1.5.0", "ci-pipeline")
    add(ts(10, 0, 48), "inventory-api", "ERROR", "StockSyncWorker: deadlock detected while updating stock_levels (sku batch 2200-2700); retrying")
    add(ts(10, 1, 3), "inventory-api", "ERROR", "HTTP 500 PUT /api/inventory/SKU-88213 - DeadlockDetected: could not serialize access due to concurrent update")
    add(ts(10, 2, 30), "postgres-db", "WARN", "inventory_db: deadlock detected; process 48112 waits for ShareLock on transaction 991203 (table stock_levels)", "pg-primary")
    add(ts(10, 4, 55), "inventory-api", "ERROR", "HTTP 500 PUT /api/inventory/SKU-10477 - DeadlockDetected: could not serialize access due to concurrent update")
    add(ts(10, 6, 12), "inventory-api", "WARN", "Warehouse client retry storm detected: 3.1x normal PUT volume")
    add(ts(10, 10, 40), "inventory-api", "ERROR", "StockSyncWorker: deadlock detected while updating stock_levels (sku batch 5000-5500); retrying")
    add(ts(10, 15, 7), "inventory-api", "ERROR", "HTTP 500 PUT /api/inventory/SKU-33190 - DeadlockDetected: could not serialize access due to concurrent update")

    # --- INC-001 orders-api bad deploy ------------------------------------- #
    add(ts(10, 0, 0), "orders-api", "INFO", "Deployment deploy-8472 started (v2.8.0 -> v2.8.1)", "ci-pipeline")
    add(ts(10, 1, 2), "orders-api", "INFO", "Pod orders-api-7d9f-x1 starting v2.8.1", "orders-api-7d9f-x1")
    add(ts(10, 1, 5), "orders-api", "INFO", "Loading database config: DB_HOST=pg-primary.internal DB_PORT=6432 DB_POOL_MODE=transaction", "orders-api-7d9f-x1")
    add(ts(10, 1, 8), "orders-api", "ERROR", "Database connection refused: pg-primary.internal:6432 (ECONNREFUSED)", "orders-api-7d9f-x1")
    add(ts(10, 1, 9), "orders-api", "ERROR", "Failed to initialize OrderRepository: could not acquire connection", "orders-api-7d9f-x1")
    add(ts(10, 1, 10), "orders-api", "WARN", "Readiness degraded: database=unavailable (serving with cached reads only)", "orders-api-7d9f-x1")
    add(ts(10, 2, 14), "orders-api", "ERROR", "HTTP 500 GET /api/orders/9183 - OrderRepositoryError: connection refused", "orders-api-7d9f-x1")
    add(ts(10, 2, 50), "orders-api", "ERROR", "HTTP 500 POST /api/orders - OrderRepositoryError: connection refused", "orders-api-7d9f-x2")
    add(ts(10, 3, 30), "orders-api", "INFO", "Deployment deploy-8472 completed: 4/4 pods running v2.8.1", "ci-pipeline")
    add(ts(10, 4, 12), "orders-api", "ERROR", "Database connection refused: pg-primary.internal:6432 (ECONNREFUSED)", "orders-api-7d9f-x3")
    add(ts(10, 5, 21), "orders-api", "ERROR", "HTTP 500 GET /api/orders/9201 - OrderRepositoryError: connection refused", "orders-api-7d9f-x1")
    add(ts(10, 8, 2), "orders-api", "WARN", "Readiness flapping: 2/4 pods failing readiness (database=unavailable)", "ci-pipeline")
    add(ts(10, 11, 47), "orders-api", "ERROR", "HTTP 500 POST /api/orders - OrderRepositoryError: connection refused", "orders-api-7d9f-x4")
    add(ts(10, 14, 0), "postgres-db", "INFO", "pg-primary: orders_db 42 active connections on port 5432; no listener configured on 6432", "pg-primary")
    add(ts(10, 17, 9), "orders-api", "ERROR", "HTTP 500 GET /api/orders/9340 - OrderRepositoryError: connection refused", "orders-api-7d9f-x2")

    # --- background INFO noise for every service --------------------------- #
    for svc in SERVICES:
        name = svc["name"]
        for m in range(0, MINUTES, 5):
            t = (WINDOW_START + timedelta(minutes=m, seconds=RNG.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%SZ")
            if name in ("postgres-db", "redis-cache"):
                if name == "redis-cache" and m >= 21:
                    continue
                msg = f"checkpoint complete; {RNG.randint(1, 9)} wal segments recycled" if name == "postgres-db" else f"DB saved on disk; {RNG.randint(100, 900)}k keys"
                src = "pg-primary" if name == "postgres-db" else "redis-cache-0"
            else:
                msg = f"Health check OK; {RNG.randint(80, 999)} req/s over last 60s"
                src = f"{name}-pod"
            add(t, name, "INFO", msg, src)

    L.sort(key=lambda r: r[0])
    return [
        {"log_id": f"log-{i + 1:05d}", "timestamp": t, "service": s, "level": lvl, "message": msg, "source": src}
        for i, (t, s, lvl, msg, src) in enumerate(L)
    ]


# --------------------------------------------------------------------------- #
def write(name: str, payload) -> None:
    path = DATA_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    count = len(payload) if isinstance(payload, list) else 1
    print(f"wrote {path.relative_to(DATA_DIR.parent)} ({count} records)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write("services.json", SERVICES)
    write("deployments.json", DEPLOYMENTS)
    write("databases.json", DATABASES)
    write("dependencies.json", DEPENDENCIES)
    write("alerts.json", ALERTS)
    write("incidents.json", INCIDENTS)
    write("metrics.json", build_metrics())
    write("logs.json", build_logs())
    write("meta.json", {"simulated_now": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "company": "AcmeCloud", "environment": "production"})


if __name__ == "__main__":
    main()
