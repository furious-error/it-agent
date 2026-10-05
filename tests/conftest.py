import pytest

from mcp_server.data_store import DataStore
from mcp_server.tools import ToolKit


@pytest.fixture
def store() -> DataStore:
    s = DataStore()  # fresh in-memory copy of the seed data per test
    yield s
    s.close()


@pytest.fixture
def tk(store: DataStore) -> ToolKit:
    return ToolKit(store)
