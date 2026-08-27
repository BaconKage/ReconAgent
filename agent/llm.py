"""Model provider shim.

The reasoning layer needs exactly two things from a language model: a
schema-constrained JSON completion, and a plain-text completion. Both are behind
this interface, so the rest of the codebase never imports a vendor SDK and never
learns which one is in use.

Supported: Anthropic (Claude) and OpenAI. The active provider is chosen from
whichever API key is present, or forced with RECONAGENT_LLM_PROVIDER.

This is not vendor-neutrality for its own sake. It exists because the whole
architecture rests on the model being replaceable: matching is deterministic and
the model only explains what the engine already decided. If swapping providers
were hard, that claim would be weaker than it sounds. Swapping providers changes
the wording of explanations and nothing else - not one match, not one metric.

Note that `core/` may not import this module, or any SDK it wraps.
`tests/test_layer_separation.py` enforces that by parsing the AST.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Defaults are overridable so a user can pick a tier they actually have access to.
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
OPENAI_DEFAULT_MODEL = "gpt-5.6-terra"

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Checked in order; the first hit wins and never overrides a real env var.
ENV_FILES = (".env.local", ".env")

_env_loaded = False


def _load_env_files() -> None:
    """Read .env.local / .env once, without overriding the real environment.

    Skipped entirely under pytest: the suite must never pick up a developer's
    real key from a file on disk and turn itself into live API spend.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in ENV_FILES:
        path = REPO_ROOT / name
        if path.exists():
            load_dotenv(path, override=False)


def _usable(value: str | None) -> bool:
    """Whether an env value is a real credential rather than template filler.

    Copying `.env.example` and filling in one line is the normal way people set
    this up, which leaves the other provider's placeholder sitting there. Reading
    it as a real key would select the wrong vendor and fail on auth with a
    confusing message, so obvious placeholders count as absent.
    """
    if not value:
        return False
    v = value.strip().strip("'\"")
    if not v or "..." in v or v.endswith("-") or " " in v:
        return False
    return v.lower() not in {
        "none", "null", "changeme", "your-api-key", "your_api_key",
        "sk-ant", "sk-", "todo",
    }


def available_keys() -> dict[str, bool]:
    """Which provider credentials look genuinely present."""
    _load_env_files()
    return {
        "anthropic": _usable(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": _usable(os.environ.get("OPENAI_API_KEY")),
    }


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMUnavailable(Exception):
    """No usable provider. Never fatal - the caller degrades to cached traces."""


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *,
                      max_tokens: int, schema_name: str) -> LLMResponse: ...

    def complete_text(self, system: str, user: str, *, max_tokens: int) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable("anthropic SDK not installed") from exc
        self._sdk = anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)
        self.client = anthropic.Anthropic()

    def complete_json(self, system, user, schema, *, max_tokens, schema_name="result"):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": system,
                # Identical across every batch in a run, so caching it turns N
                # batches into one paid prefix.
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return LLMResponse(text=text, model=self.model, provider=self.name)

    def complete_text(self, system, user, *, max_tokens):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return LLMResponse(text=text, model=self.model, provider=self.name)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

class OpenAIProvider:
    """OpenAI via the Responses API.

    Structured output is expressed as `text={"format": {...}}` on
    `responses.create`, not as `response_format` on `chat.completions`. Strict
    mode requires every property to appear in `required` and
    `additionalProperties: false` at each level - which the reasoning schema
    already satisfies, since it was written to be unambiguous rather than to
    satisfy any one vendor.
    """
    name = "openai"

    def __init__(self, model: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise LLMUnavailable("openai SDK not installed (pip install openai)") from exc
        self._sdk = openai
        self.model = model or os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
        self.client = openai.OpenAI()

    def complete_json(self, system, user, schema, *, max_tokens, schema_name="result"):
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            text={"format": {"type": "json_schema", "name": schema_name,
                             "strict": True, "schema": schema}},
            max_output_tokens=max_tokens,
        )
        return LLMResponse(text=_openai_text(response), model=self.model, provider=self.name)

    def complete_text(self, system, user, *, max_tokens):
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            max_output_tokens=max_tokens,
        )
        return LLMResponse(text=_openai_text(response), model=self.model, provider=self.name)


def _openai_text(response: Any) -> str:
    """Pull the final answer text out of a Responses API result.

    Deliberately NOT `response.output_text`. That accessor concatenates the text
    of *every* message item in the response, and a reasoning model routinely
    emits more than one - interleaved with `reasoning` items. For prose the
    concatenation is merely odd; for a JSON schema response it produces two valid
    objects glued together, which fails to parse with "Extra data: line 1 column
    N". The failure is intermittent, because whether a second message appears
    depends on the prompt.

    The last message item is the model's final answer, so that is what is
    returned. Content parts *within* that one message are joined, since a single
    message legitimately splits across parts.
    """
    messages: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        parts = [
            getattr(block, "text", "") or ""
            for block in (getattr(item, "content", None) or [])
            if getattr(block, "type", None) in ("output_text", "text")
        ]
        if parts:
            messages.append("".join(parts))
    if messages:
        return messages[-1]
    # Older or unexpected shapes: fall back rather than returning nothing.
    return getattr(response, "output_text", "") or ""


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}


def detect_provider_name() -> str | None:
    """Which provider to use, from the environment.

    An explicit RECONAGENT_LLM_PROVIDER wins. Otherwise the first *usable* key,
    checked in a fixed order so behaviour is reproducible when both are set.
    Placeholder values left over from `.env.example` do not count as present.
    """
    _load_env_files()
    forced = (os.environ.get("RECONAGENT_LLM_PROVIDER") or "").strip().lower()
    if forced:
        return forced if forced in PROVIDERS else None
    keys = available_keys()
    if keys["anthropic"]:
        return "anthropic"
    if keys["openai"]:
        return "openai"
    return None


def get_provider(name: str | None = None) -> LLMProvider:
    """Build the active provider, or raise LLMUnavailable with a usable reason."""
    name = name or detect_provider_name()
    if name is None:
        raise LLMUnavailable(
            "no API key found - set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    if name not in PROVIDERS:
        raise LLMUnavailable(
            f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")
    key_var = "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENAI_API_KEY"
    if not available_keys()[name]:
        present = os.environ.get(key_var)
        raise LLMUnavailable(
            f"{name} selected but {key_var} looks like a placeholder"
            if present else f"{name} selected but {key_var} is not set")
    try:
        return PROVIDERS[name]()
    except LLMUnavailable:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise LLMUnavailable(f"{name} client init failed: {exc}") from exc


def describe_provider() -> str:
    """One-line status for the CLI and the UI."""
    try:
        p = get_provider()
    except LLMUnavailable as exc:
        return f"unavailable ({exc})"
    return f"{p.name} / {p.model}"
