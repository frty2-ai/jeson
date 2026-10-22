import pytest

from jeson import compile_schema


def kinds(schema):
    c = compile_schema(schema)
    return {leaf.key: leaf.kind for leaf in c.leaves}, c


def test_every_primitive():
    k, c = kinds({"type": "object", "properties": {
        "signed": {"type": "boolean"},
        "currency": {"type": "string", "enum": ["INR", "USD"]},
        "grade": {"type": "string", "enum": ["low", "mid", "high"], "x-ordered": True},
        "rows": {"type": "integer", "minimum": 1, "maximum": 10},
        "total": {"type": "number"},
        "count": {"type": "integer"},
        "logo_box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
        "name": {"type": "string", "x-primitive": "span"},
        "notes": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}},
    }})
    assert k == {"signed": "noul", "currency": "choice", "grade": "score", "rows": "score", "total": "number", "count": "number",
                 "logo_box": "box", "tags": "multi_label", "name": "span"}
    assert c.needs_generation == ["notes", "items"]


def test_nested_paths_and_required():
    c = compile_schema({"type": "object", "required": ["a"], "properties": {
        "a": {"type": "object", "required": ["b"], "properties": {"b": {"type": "boolean"}}}}})
    assert [leaf.key for leaf in c.leaves] == ["a.b"]
    assert c.required == {(): ["a"], ("a",): ["b"]}


def test_nullable_adds_none_as_a_value():
    c = compile_schema({"type": "object", "properties": {
        "x": {"type": ["boolean", "null"]},
        "y": {"anyOf": [{"type": "string", "enum": ["p", "q"]}, {"type": "null"}]}}})
    x, y = c.leaves
    assert x.nullable and x.values == (True, False, None)
    assert y.kind == "choice" and y.values == ("p", "q", None)


def test_score_range_cap():
    _, c = kinds({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 100}}})
    assert c.leaves[0].kind == "number"  # more than 21 levels is a number, not an ordinal


def test_forced_primitive_and_unknown():
    k, _ = kinds({"type": "object", "properties": {"t": {"type": "string", "x-primitive": "generate"}}})
    assert k == {}
    with pytest.raises(ValueError):
        compile_schema({"type": "object", "properties": {"t": {"type": "string", "x-primitive": "nope"}}})


def test_refs_are_refused_clearly():
    with pytest.raises(NotImplementedError):
        compile_schema({"$defs": {"a": {"type": "boolean"}}, "type": "object", "properties": {"x": {"$ref": "#/$defs/a"}}})


def test_leaf_text_carries_name_title_description():
    c = compile_schema({"type": "object", "properties": {"pan_number": {"type": "boolean", "title": "PAN", "description": "A PAN is printed."}}})
    assert c.leaves[0].text == "pan number (PAN): A PAN is printed."
