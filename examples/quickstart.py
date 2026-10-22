"""Read a synthetic invoice against its schema, zero-shot, and compare with the truth it was rendered from.

  pip install -e ".[models]"
  python examples/quickstart.py
"""
import json
from pathlib import Path

from jeson import Reader

HERE = Path(__file__).parent
schema = json.loads((HERE / "invoice.schema.json").read_text())
truth = json.loads((HERE / "invoice.truth.json").read_text())

reader = Reader.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")      # about 4.5 GB on first use; needs a CUDA GPU
reading = reader.read(HERE / "invoice.jpg", schema)

print(json.dumps(reading.json, indent=1))
print(f"\n{reading.latency_ms:.0f} ms, no tokens decoded")
for key, value in reading.answers().items():
    mark = "ok " if value == truth[key] else "-- "
    print(f"{mark}{key:22s} {json.dumps(value):12s} p={reading.confidence()[key]:.2f}   truth {json.dumps(truth[key])}")
