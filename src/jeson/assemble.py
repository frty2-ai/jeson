"""Assemble the JSON answer from per-leaf predictions, in code: valid by construction, never parsed or repaired.

A prediction per leaf key is a head's decoded value (a bool, a declared option, a level, a span string), or None
when the leaf is nullable and null won. The assembler checks each value against its leaf, so a head that returns
something outside its type is a bug raised here, not malformed JSON downstream.

Leaves no head answered (a kind without a head yet, or free text) are not invented: their keys are listed under
"_unanswered" so a caller can route them elsewhere.
"""
from __future__ import annotations

from .schema import Compiled, Leaf

UNANSWERED = "_unanswered"
_MISSING = object()


def check(leaf: Leaf, value):
    """The value if it is a valid answer for the leaf; raises ValueError otherwise."""
    if value is None:
        if not leaf.nullable:
            raise ValueError(f"{leaf.key}: null from a leaf that is not nullable")
        return None
    ok = {
        "noul": lambda v: isinstance(v, bool),
        "choice": lambda v: v in leaf.options,
        "score": lambda v: v in leaf.options,
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "span": lambda v: isinstance(v, str),
        "box": lambda v: isinstance(v, (list, tuple)) and len(v) == 4,
        "multi_label": lambda v: isinstance(v, (list, tuple)) and all(x in leaf.options for x in v),
        "multi_span": lambda v: isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v),
    }[leaf.kind]
    if not ok(value):
        raise ValueError(f"{leaf.key}: {value!r} is not a valid {leaf.kind}")
    return list(value) if isinstance(value, tuple) else value


def assemble(compiled: Compiled, predictions: dict) -> dict:
    """The nested JSON object. Keys without a prediction (and free-text keys) are listed under "_unanswered"."""
    out: dict = {}
    unanswered = list(compiled.needs_generation)
    for leaf in compiled.leaves:
        value = predictions.get(leaf.key, _MISSING)
        if value is _MISSING:
            unanswered.append(leaf.key)
            continue
        node = out
        for part in leaf.path[:-1]:
            node = node.setdefault(part, {})
        node[leaf.path[-1]] = check(leaf, value)
    if unanswered:
        out[UNANSWERED] = unanswered
    return out


def flatten(d: dict, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}: the leaf keys of a nested answer."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
