"""SQLite checkpoint storage.

Compaction shortens the transcript. Checkpointing is separate: it writes the
graph state after every step so a restarted process can resume, including while
it is waiting for approval.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def open_checkpointer(path: str | Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return connection, saver
