from jeson import compile_schema
from jeson.readout import _rows, _tokenize


class CharTok:
    """A stand-in tokenizer: one token per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


def test_rows_cover_every_value_and_values_start_after_the_shared_prefix():
    c = compile_schema({"type": "object", "properties": {"ok": {"type": "boolean"}, "c": {"type": ["string", "null"], "enum": ["aa", "ab"]}}})
    rows = list(_rows(c))
    assert [(leaf.key, v) for leaf, v, *_ in rows] == [("ok", True), ("ok", False), ("c", "aa"), ("c", "ab"), ("c", None)]
    ids, start = _tokenize(CharTok(), rows)
    for (_, _, ctx, full), s in zip(rows, start):
        assert full.startswith(ctx) and s >= len(ctx)  # scoring starts at or after the field marker
    # the bare context '{"ok":' ends before the space, so the space is scored with the value (BPE may merge them)
    assert start[0] == len('{"ok":')
