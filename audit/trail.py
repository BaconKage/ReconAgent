"""Append-only audit trail.

Every reconciliation decision - matched, split, duplicate or refused - is written
here as one JSON line, whether or not a language model was involved. Two reasons
this is not optional in a finance tool:

1. **Accountability.** For any transaction a reviewer can ask "why did this end
   up like that", and the answer is a record of what the engine actually did:
   which rules ran, which declined, what evidence was in front of it.

2. **It is the substrate for Q&A.** The question-answering layer reads this file
   and nothing else. That is what makes it retrieval rather than a fresh guess -
   it can only report decisions that were genuinely made.

Append-only is deliberate: entries are never rewritten, so the file is a history
rather than a current-state snapshot.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_TRAIL_DIR = Path("audit_trail")


def _jsonable(value: Any) -> Any:
    """Convert domain objects to something json.dump can handle."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


class AuditTrail:
    """One file per reconciliation run."""

    def __init__(self, run_id: str | None = None, directory: Path | str = DEFAULT_TRAIL_DIR):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.run_id}.jsonl"
        self._count = 0

    # -- writing -------------------------------------------------------

    def append(self, entry: dict[str, Any]) -> str:
        entry_id = entry.get("entry_id") or f"ae_{uuid.uuid4().hex[:12]}"
        payload = {
            "entry_id": entry_id,
            "run_id": self.run_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **_jsonable(entry),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=False) + "\n")
        self._count += 1
        return entry_id

    def append_decision(self, result, *, inputs: dict[str, Any] | None = None) -> str:
        """Record one engine decision, before any reasoning has run."""
        return self.append({
            "record_id": result.record_id,
            "status": result.status,
            "match_type": result.match_type,
            "confidence": result.confidence,
            "exception_reason": result.exception_reason,
            "linked_ids": result.linked_ids,
            "rule_trace": result.rule_trace,
            "near_misses": [nm.to_dict() for nm in result.near_misses],
            "inputs": inputs or {},
            "reasoning": None,
        })

    def attach_reasoning(self, record_id: str, reasoning: dict[str, Any]) -> str:
        """Record a reasoning result as a new entry rather than editing the old one.

        The trail is append-only, so an investigation is a *later observation*
        about a decision, not a mutation of it. Readers reconcile the two by
        record_id, which keeps the ordering of what was known when.
        """
        return self.append({
            "record_id": record_id,
            "entry_kind": "reasoning",
            "reasoning": reasoning,
        })

    # -- reading -------------------------------------------------------

    def __len__(self) -> int:
        return self._count

    @classmethod
    def read(cls, path: Path | str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return entries
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @classmethod
    def latest(cls, directory: Path | str = DEFAULT_TRAIL_DIR) -> Path | None:
        d = Path(directory)
        trails = sorted(d.glob("*.jsonl")) if d.exists() else []
        return trails[-1] if trails else None

    @classmethod
    def iter_records(cls, path: Path | str) -> Iterator[dict[str, Any]]:
        yield from cls.read(path)


def merge_by_record(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold an append-only trail into one view per record.

    Later entries layer onto earlier ones: the engine decision arrives first, a
    reasoning entry may arrive after. Nothing is overwritten on disk - this is
    only how a reader assembles the current picture.
    """
    merged: dict[str, dict[str, Any]] = {}
    for e in entries:
        rid = e.get("record_id")
        if not rid:
            continue
        cur = merged.setdefault(rid, {"record_id": rid})
        for k, v in e.items():
            if k in ("entry_id", "run_id", "logged_at", "entry_kind"):
                continue
            if v is None and k in cur and cur[k] is not None:
                continue          # never let a later null erase a real value
            cur[k] = v
    return merged
