"""Compile a JSON Schema into typed leaves. Each leaf is answered by a typed head, never by generated text.

The mapping is made from the schema alone and is deterministic:

  boolean                                        noul         true / false
  string (or number) with enum                   choice       one declared value
  enum with "x-ordered": true                    score        one declared level, trained as ordered
  integer with minimum and maximum, <= 21 levels score        minimum..maximum
  number, or integer without a small range       number       (no head yet: listed as unanswered)
  array of 4 numbers named *_box / "box"         box          (no head yet)
  array of enums                                 multi_label  (no head yet)
  string with "x-primitive": "span"              span         text copied from the input (experimental)
  array of span strings                          multi_span   (no head yet)
  object                                         its properties, recursively
  anything else (free text, arrays of objects)   needs generation: never answered, listed in the output

Any leaf can be forced with "x-primitive": one of PRIMITIVES, or "generate". A leaf whose type allows null
(type ["x", "null"], or anyOf with {"type": "null"}) is nullable: null becomes one of its answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRIMITIVES = ("noul", "choice", "score", "number", "span", "box", "multi_label", "multi_span")
TYPED = ("noul", "choice", "score")  # the kinds the readout and the heads answer today
MAX_SCORE_LEVELS = 21


@dataclass(frozen=True)
class Leaf:
    path: tuple[str, ...]          # ("applicant", "name"): the JSON pointer without the slashes
    kind: str                      # one of PRIMITIVES
    title: str
    description: str
    options: tuple = ()            # choice / multi_label: declared values; score: the levels
    nullable: bool = False

    @property
    def key(self) -> str:
        return ".".join(self.path)

    @property
    def text(self) -> str:
        """What a head is conditioned on: name, title and description, in that order."""
        name = self.path[-1].replace("_", " ")
        head = name if self.title.lower() == name.lower() or not self.title else f"{name} ({self.title})"
        return f"{head}: {self.description}" if self.description else head

    @property
    def values(self) -> tuple:
        """Every answer this leaf may take: booleans or its options, plus None when nullable."""
        base = (True, False) if self.kind == "noul" else tuple(self.options)
        return base + ((None,) if self.nullable else ())


@dataclass
class Compiled:
    leaves: list[Leaf] = field(default_factory=list)
    needs_generation: list[str] = field(default_factory=list)   # keys no typed head answers (free text, ...)
    required: dict[tuple[str, ...], list[str]] = field(default_factory=dict)  # object path -> required names

    def by_kind(self, *kinds: str) -> list[Leaf]:
        return [leaf for leaf in self.leaves if leaf.kind in kinds]

    def typed(self) -> list[Leaf]:
        return self.by_kind(*TYPED)


def _types(spec: dict) -> tuple[set[str], bool]:
    """The non-null types a spec allows, and whether null is allowed."""
    t = spec.get("type")
    types = set(t) if isinstance(t, list) else ({t} if t else set())
    for alt in spec.get("anyOf") or spec.get("oneOf") or []:
        if isinstance(alt, dict):
            types |= _types(alt)[0] | ({"null"} if alt.get("type") == "null" else set())
            if "enum" in alt and "type" not in alt:
                types.add("string")
    null = "null" in types or (None in (spec.get("enum") or []))
    return types - {"null"}, null


def _enum(spec: dict) -> list | None:
    if "enum" in spec:
        return [v for v in spec["enum"] if v is not None]
    for alt in spec.get("anyOf") or spec.get("oneOf") or []:
        if isinstance(alt, dict) and "enum" in alt:
            return [v for v in alt["enum"] if v is not None]
    return None


def _is_box(name: str, spec: dict) -> bool:
    items = spec.get("items") or {}
    numeric = isinstance(items, dict) and items.get("type") in ("integer", "number")
    four = spec.get("minItems", 4) == 4 and spec.get("maxItems", 4) == 4
    return numeric and four and (name.endswith("_box") or name == "box" or "bounding box" in spec.get("description", "").lower())


def _kind(name: str, spec: dict) -> str:
    forced = spec.get("x-primitive")
    if forced:
        if forced not in PRIMITIVES + ("generate",):
            raise ValueError(f"{name}: unknown x-primitive {forced!r}")
        return forced
    types, _ = _types(spec)
    enum = _enum(spec)
    if types == {"boolean"}:
        return "noul"
    if enum is not None and types <= {"string", "integer", "number"}:
        return "score" if spec.get("x-ordered") else "choice"
    if types and types <= {"integer"} and "minimum" in spec and "maximum" in spec \
            and spec["maximum"] - spec["minimum"] + 1 <= MAX_SCORE_LEVELS:
        return "score"
    if types and types <= {"integer", "number"}:
        return "number"
    if types == {"array"}:
        if _is_box(name, spec):
            return "box"
        items = spec.get("items") or {}
        if isinstance(items, dict):
            if _enum(items) is not None:
                return "multi_label"
            if items.get("x-primitive") == "span":
                return "multi_span"
    return "generate"


def compile_schema(schema: dict) -> Compiled:
    """Walk the schema once; every property becomes a typed leaf or a needs-generation key."""
    if "$ref" in schema or "$defs" in schema or "definitions" in schema:
        raise NotImplementedError("$ref / $defs are not supported yet: inline the definitions (see docs/schema.md)")
    out = Compiled()

    def walk(spec: dict, path: tuple[str, ...]) -> None:
        props = spec.get("properties")
        if props is not None and "object" in (_types(spec)[0] or {"object"}):
            out.required[path] = list(spec.get("required", []))
            for name, sub in props.items():
                walk(sub, path + (name,))
            return
        name = path[-1] if path else ""
        kind = _kind(name, spec)
        if kind == "generate":
            out.needs_generation.append(".".join(path))
            return
        _, nullable = _types(spec)
        if kind in ("choice", "multi_label"):
            options = tuple(_enum(spec) if kind == "choice" else _enum(spec["items"]))
        elif kind == "score":
            options = tuple(_enum(spec) or range(spec["minimum"], spec["maximum"] + 1))
        else:
            options = ()
        out.leaves.append(Leaf(path, kind, spec.get("title", ""), spec.get("description", ""), options, nullable))

    walk(schema, ())
    return out
