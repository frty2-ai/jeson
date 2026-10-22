"""jeson: typed JSON from images in one forward pass. A JSON schema in, valid JSON out, no token decoding.

    from jeson import Reader, Lock
    reader = Reader.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    reading = reader.read("invoice.jpg", schema)
    reading.json                                      # valid by construction
    lock = Lock.fit([reader.read(p, schema) for p in pages], labels)   # seconds, on a CPU
    reader.extract("new.jpg", schema, lock=lock)
"""
from .assemble import assemble, flatten
from .lock import Lock
from .reading import Reading
from .schema import Compiled, Leaf, compile_schema

__version__ = "0.1.0"


def __getattr__(name):  # the model classes import torch lazily, so the core installs and imports without it
    if name == "Reader":
        from .readout import Reader
        return Reader
    if name == "FastReader":
        from .fast import FastReader
        return FastReader
    raise AttributeError(name)


__all__ = ["Compiled", "FastReader", "Leaf", "Lock", "Reader", "Reading", "assemble", "compile_schema", "flatten", "__version__"]
