import pytest

from jeson import assemble, compile_schema, flatten
from jeson.assemble import UNANSWERED

SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["invoice", "receipt"]},
    "seller": {"type": "object", "properties": {"gst": {"type": "boolean"}, "name": {"type": "string", "x-primitive": "span"}}},
    "note": {"type": "string"}}}


def test_nested_output_and_unanswered():
    c = compile_schema(SCHEMA)
    out = assemble(c, {"kind": "invoice", "seller.gst": True})
    assert out == {"kind": "invoice", "seller": {"gst": True}, UNANSWERED: ["note", "seller.name"]}


def test_invalid_values_raise_instead_of_emitting_bad_json():
    c = compile_schema(SCHEMA)
    with pytest.raises(ValueError):
        assemble(c, {"kind": "bill"})
    with pytest.raises(ValueError):
        assemble(c, {"seller.gst": "yes"})
    with pytest.raises(ValueError):
        assemble(c, {"kind": None})  # not nullable


def test_flatten_roundtrip():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
