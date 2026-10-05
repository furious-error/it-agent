from mcp_server.data_store import DataStore

from app.reports import record_user_report


def test_natural_language_reports_get_distinct_incident_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_SEQ_FILE", str(tmp_path / "seq.sqlite"))
    first_db = tmp_path / "a.db"
    second_db = tmp_path / "b.db"
    DataStore(db_path=str(first_db)).close()
    DataStore(db_path=str(second_db)).close()

    first = record_user_report(first_db, "orders-api is returning HTTP 500 errors.")
    second = record_user_report(second_db, "both orders-api and payments-api look degraded.")

    assert first == "INC-011"
    assert second == "INC-012"
    store = DataStore(db_path=str(first_db))
    try:
        incident = store.get_incident(first)
    finally:
        store.close()
    assert incident["service"] == "orders-api"
    assert incident["status"] == "open"
    assert incident["description"] == "orders-api is returning HTTP 500 errors."

    store = DataStore(db_path=str(second_db))
    try:
        incident = store.get_incident(second)
    finally:
        store.close()
    assert incident["service"] is None
