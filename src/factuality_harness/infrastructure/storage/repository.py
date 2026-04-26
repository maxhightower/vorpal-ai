"""Audit-trace persistence.

Two implementations:
 - ``InMemoryAuditRepository`` for tests.
 - ``JsonAuditRepository`` writing one file per run under a configurable directory.

A SQLAlchemy-backed repository can be added later behind the same protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ...domain.audit import AuditTrace


class AuditRepository(Protocol):
    def save(self, trace: AuditTrace) -> None: ...

    def get(self, audit_id: str) -> AuditTrace | None: ...

    def list_ids(self) -> list[str]: ...


class InMemoryAuditRepository:
    def __init__(self) -> None:
        self._traces: dict[str, AuditTrace] = {}

    def save(self, trace: AuditTrace) -> None:
        self._traces[trace.audit_id] = trace

    def get(self, audit_id: str) -> AuditTrace | None:
        return self._traces.get(audit_id)

    def list_ids(self) -> list[str]:
        return list(self._traces.keys())


class JsonAuditRepository:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, audit_id: str) -> Path:
        return self._dir / f"{audit_id}.json"

    def save(self, trace: AuditTrace) -> None:
        self._path_for(trace.audit_id).write_text(trace.to_json(), encoding="utf-8")

    def get(self, audit_id: str) -> AuditTrace | None:
        path = self._path_for(audit_id)
        if not path.exists():
            return None
        return AuditTrace.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("audit_*.json"))


_: AuditRepository = InMemoryAuditRepository()
