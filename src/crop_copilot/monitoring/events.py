from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


class EventLogger:
    """Append operational metadata only; never stores images or question text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def log(self, event_type: str, request_id: str, **metadata: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "request_id": request_id,
            **metadata,
        }
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

