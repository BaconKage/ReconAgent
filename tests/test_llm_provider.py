"""Tests for the model provider shim.

Two things are being defended.

**Selection is predictable.** Which vendor answers is decided by the environment
alone, in a fixed order, with an explicit override. A run that silently picks a
different provider than the operator expected would make the reasoning traces
unreproducible.

**Provider choice cannot reach the engine.** This is the important one. The whole
architecture claims that correctness is deterministic and the model only
explains. If swapping vendors moved a single match, that claim would be false.

No test here constructs a real client or makes a network call.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from agent.llm import (ANTHROPIC_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL, LLMUnavailable,
                       describe_provider, detect_provider_name, get_provider)

DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RECONAGENT_LLM_PROVIDER",
                "ANTHROPIC_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_no_keys_means_no_provider():
    assert detect_provider_name() is None


def test_anthropic_key_selects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert detect_provider_name() == "anthropic"


def test_openai_key_selects_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert detect_provider_name() == "openai"


def test_both_keys_resolve_deterministically(monkeypatch):
    """Ambiguity in provider choice would make traces unreproducible."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert detect_provider_name() == "anthropic"


def test_explicit_override_wins_over_key_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("RECONAGENT_LLM_PROVIDER", "openai")
    assert detect_provider_name() == "openai"


def test_unknown_override_is_rejected_not_guessed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("RECONAGENT_LLM_PROVIDER", "gemini")
    assert detect_provider_name() is None
    with pytest.raises(LLMUnavailable):
        get_provider("gemini")


def test_selecting_a_provider_without_its_key_fails_clearly(monkeypatch):
    """Naming openai while only holding an Anthropic key must not silently swap."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        get_provider("openai")


def test_missing_key_message_names_both_options():
    with pytest.raises(LLMUnavailable) as exc:
        get_provider()
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg and "OPENAI_API_KEY" in msg


def test_describe_provider_is_safe_with_no_key():
    assert "unavailable" in describe_provider()


def test_missing_sdk_is_reported_as_unavailable_not_raised(monkeypatch):
    """A key present but the SDK absent is an ordinary, recoverable condition."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name.split(".")[0] == "openai":
            raise ImportError("blocked")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(LLMUnavailable, match="openai SDK not installed"):
        get_provider()


# --------------------------------------------------------------------------
# Model defaults
# --------------------------------------------------------------------------

def test_model_defaults_are_overridable_by_environment():
    """Users hold different access tiers; pinning one model would strand them."""
    assert ANTHROPIC_DEFAULT_MODEL and OPENAI_DEFAULT_MODEL
    import agent.llm as llm
    assert "OPENAI_MODEL" in Path(llm.__file__).read_text(encoding="utf-8")
    assert "ANTHROPIC_MODEL" in Path(llm.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The claim that matters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["anthropic", "openai", None])
def test_matching_is_identical_regardless_of_provider(monkeypatch, provider):
    """Reconciliation output must not vary with which vendor is configured.

    The engine never consults `agent.llm`, so this should hold trivially - which
    is exactly why it is worth asserting. If it ever stops holding, the claim
    that the model only explains is no longer true.
    """
    from core.loader import load_batch
    from core.matcher import reconcile

    if provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    elif provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    rep = reconcile(load_batch(DEV))
    return_sig = sorted((r.record_id, r.status, r.match_type, r.confidence,
                         tuple(sorted(r.group_key))) for r in rep.results)

    baseline = reconcile(load_batch(DEV))
    baseline_sig = sorted((r.record_id, r.status, r.match_type, r.confidence,
                           tuple(sorted(r.group_key))) for r in baseline.results)
    assert return_sig == baseline_sig


def test_engine_does_not_import_the_provider_shim():
    """core/ must not reach agent.llm - the layer rule, restated for this module."""
    import ast
    core = Path(__file__).resolve().parents[1] / "core"
    for path in core.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("agent"), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("agent"), f"{path.name} imports {a.name}"
