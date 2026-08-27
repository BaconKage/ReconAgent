"""Proof that matching does not depend on the LLM - or on anything else.

The central architectural claim of this project is that correctness lives in the
deterministic engine and the language model only explains what that engine
already decided. A claim like that is worth nothing if it is only stated in a
README, because it decays the moment someone adds a convenient import.

These tests make it mechanical:

* `core/` may not import any LLM SDK, HTTP client, or the `agent` package.
* `core/` may not read ground truth - it must not be able to see the answers.
* Reconciliation must complete with the network physically disabled.
* Reconciliation must complete with `anthropic` unimportable.

If someone later wires an LLM into the matching path, the suite fails and says so.
"""

from __future__ import annotations

import ast
import builtins
import socket
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "core"

FORBIDDEN_MODULES = {
    "anthropic", "openai", "google", "cohere", "mistralai", "ollama",
    "langchain", "llama_index", "transformers",
    "requests", "httpx", "urllib3", "aiohttp", "socket", "urllib",
    "agent",
}

CORE_MODULES = sorted(p for p in CORE.glob("*.py") if p.name != "__init__.py")


def imported_names(path: Path) -> set[str]:
    """Top-level module names imported by a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: p.name)
def test_core_module_imports_nothing_forbidden(path: Path):
    offending = imported_names(path) & FORBIDDEN_MODULES
    assert not offending, (
        f"{path.name} imports {sorted(offending)}. The matching engine must stay "
        f"free of model SDKs and network clients."
    )


@pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: p.name)
def test_core_module_cannot_see_ground_truth(path: Path):
    """The engine must never read the answer key, even accidentally."""
    text = path.read_text(encoding="utf-8").lower()
    assert "ground_truth" not in text, (
        f"{path.name} references ground truth. The engine is evaluated against "
        f"that file; it must not be able to read it."
    )


def test_reconciliation_runs_with_the_network_disabled():
    """Nothing in the matching path may reach out over the wire.

    `socket.socket` is replaced with something that raises, so any HTTP client,
    telemetry call or model request would fail loudly rather than quietly work
    on the developer's machine and inflate the reported throughput.
    """
    from core.loader import load_batch
    from core.matcher import reconcile

    batch = load_batch(Path(__file__).resolve().parents[1] / "data" / "dev")

    original = socket.socket

    class Blocked(socket.socket):
        def __init__(self, *a, **kw):
            raise AssertionError("core/ attempted a network connection")

    socket.socket = Blocked
    try:
        report = reconcile(batch)
    finally:
        socket.socket = original

    assert report.results
    assert report.by_status().get("matched", 0) > 0


def test_reconciliation_runs_when_the_anthropic_sdk_is_unimportable():
    """A judge with no API key - and no SDK installed - still gets full matching."""
    from core.loader import load_batch
    from core.matcher import reconcile

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"anthropic", "openai"}:
            raise ImportError(f"{name} is deliberately unavailable in this test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        batch = load_batch(Path(__file__).resolve().parents[1] / "data" / "dev")
        report = reconcile(batch)
    finally:
        builtins.__import__ = real_import

    assert report.by_status().get("matched", 0) > 0
    assert report.by_status().get("matched_split", 0) > 0


def test_matching_outcomes_are_identical_across_repeated_runs():
    """Determinism, on the real dataset rather than a fixture.

    This is what makes it defensible to say the LLM cannot change a match: the
    engine's output is a pure function of the CSVs and the config.
    """
    from core.loader import load_batch
    from core.matcher import reconcile

    root = Path(__file__).resolve().parents[1] / "data" / "dev"
    a = reconcile(load_batch(root))
    b = reconcile(load_batch(root))

    def signature(rep):
        return sorted((r.record_id, r.status, r.match_type, r.confidence,
                       tuple(sorted(r.group_key))) for r in rep.results)

    assert signature(a) == signature(b)
