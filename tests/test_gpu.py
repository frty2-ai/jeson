"""Model tests: need a CUDA GPU and download Qwen3-VL-2B (about 4.5 GB). Run with `pytest -m gpu`."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
EX = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def reader():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from jeson import Reader
    return Reader.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


def test_reader_answers_every_typed_field_with_an_allowed_value(reader):
    schema = json.loads((EX / "invoice.schema.json").read_text())
    reading = reader.read(EX / "invoice.jpg", schema)
    leaves = {leaf.key: leaf for leaf in reading.compiled.typed()}
    assert set(reading.dists) == set(leaves)
    for key, dist in reading.dists.items():
        assert set(dist) == set(leaves[key].values)
        assert abs(sum(dist.values()) - 1.0) < 1e-4
    assert "_unanswered" not in reading.json


def test_field_order_does_not_change_the_answers(reader):
    schema = json.loads((EX / "invoice.schema.json").read_text())
    rev = {**schema, "properties": dict(reversed(list(schema["properties"].items())))}
    a = reader.read(EX / "invoice.jpg", schema).answers()
    b = reader.read(EX / "invoice.jpg", rev).answers()
    # the prompt carries the schema, so reordering it changes the prefix: answers may move by bf16 noise, not more
    assert sum(a[k] == b[k] for k in a) >= len(a) - 1
