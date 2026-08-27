"""Content-addressed cache of reasoning traces.

Why this exists: a judge cloning this repo should be able to run the full demo -
including the agent's explanations - without an Anthropic API key. Traces are
keyed by a hash of the exact evidence the model was shown, and committed to the
repository. With a key, the run is live and refreshes the cache; without one, it
replays.

Keying on the evidence rather than on a record ID matters. If the underlying data
or the engine's finding changes, the hash changes and the stale explanation is
never served for a case it no longer describes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "traces.json"


def evidence_key(bundle: dict[str, Any], instructions: str = "") -> str:
    """Stable hash of an evidence bundle plus the instructions it was judged under.

    `sort_keys` plus separators makes the encoding canonical, so the same
    evidence always produces the same key regardless of dict ordering.

    `instructions` folds the system prompt into the key. An answer is a function
    of the evidence *and* what the model was told to do with it, so editing the
    prompt must invalidate the cache. Without this, a prompt change would appear
    to take effect while every existing case silently replayed an answer written
    under the old rules - and the difference would be invisible, because a
    cached trace is indistinguishable from a fresh one once stored.
    """
    blob = json.dumps(bundle, sort_keys=True, separators=(",", ":"), default=str)
    if instructions:
        blob = f"{hashlib.sha256(instructions.encode('utf-8')).hexdigest()[:16]}|{blob}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TraceCache:
    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        self._dirty = False
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # A corrupt cache must not take the run down - it is an
                # optimisation, not a source of truth.
                self._data = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
