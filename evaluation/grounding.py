"""Record-ID extraction, shared by the grounding check and the agent evaluator.

Both `verify_grounding.py` and `evaluation/agent_eval.py` need to ask the same
question - "which records does this text claim something about?" - and they must
ask it the same way, or their numbers stop being comparable. Two copies of a
regex is exactly how that drifts.

Nothing here reads ground truth, so this module carries no special privilege; it
lives under `evaluation/` only because that is where its two callers meet.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: The three record-ID shapes this domain uses. Anything matching one of these in
#: model-written text is a factual claim about a record, and is checkable.
ID_PATTERN = re.compile(r"\b(?:pay_[a-z0-9]+|order_[a-z0-9]+|BNK_\d+)\b")


def ids_in(obj: Any) -> set[str]:
    """Every record ID appearing anywhere in a nested structure.

    Serialising and regexing rather than walking the shape deliberately: the
    point is to catch an ID *wherever* it appears, including inside a narration
    string or a nested tool observation, and a structural walk would need
    updating every time either shape changed.
    """
    if obj is None:
        return set()
    blob = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return set(ID_PATTERN.findall(blob))


def permitted_ids(bundle: dict, trace: dict) -> set[str]:
    """What the model was allowed to know about, from its own point of view.

    The evidence bundle it was handed, plus whatever a read-only tool returned to
    it during the investigation. Tool *parameters* are excluded on purpose: those
    are the model's own choice, so they are a claim to be checked rather than a
    source to be trusted.
    """
    seen = ids_in(bundle)
    for step in trace.get("investigation_trace") or []:
        seen |= ids_in(step.get("observation"))
    return seen


def written_ids(trace: dict) -> dict[str, set[str]]:
    """What the model asserted, split by where it asserted it.

    Counting `tool_params` is what makes this stricter than a check on the final
    answer alone: a model that invents a plausible bank row and then asks a tool
    about it has hallucinated, even if the tool returns nothing and the invented
    ID never reaches the conclusion. That is where it surfaces first.
    """
    out = {
        "hypothesis": ids_in(trace.get("hypothesis")),
        "evidence_cited": set(trace.get("evidence_cited") or []),
        "thought": set(),
        "tool_params": set(),
    }
    for step in trace.get("investigation_trace") or []:
        out["thought"] |= ids_in(step.get("thought"))
        out["tool_params"] |= ids_in(step.get("params"))
    return out


def ungrounded_in(bundle: dict, trace: dict) -> list[tuple[str, str]]:
    """(field, record_id) for every ID the model wrote but was never shown."""
    allowed = permitted_ids(bundle, trace)
    return [(field, rid)
            for field, ids in written_ids(trace).items()
            for rid in sorted(ids) if rid not in allowed]
