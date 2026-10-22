#!/usr/bin/env python3
"""Latency against schema size: the Reader (GPU) and the FastReader (GPU and CPU) on the same pages, with schemas of
5 to 100 typed fields cut from the L3 bank-statement schema (107 fields). Batch 1, p50 over the pages. The fast head
is trained for one epoch on 16 pages first: its weights do not change its speed.

  python benchmarks/speed.py --data data/synth-docs
Out: benchmarks/results/speed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jeson import FastReader, Reader, compile_schema

HERE = Path(__file__).resolve().parent
SIZES = (5, 10, 25, 50, 100)


def first_fields(schema: dict, n: int) -> dict:
    """The schema cut to its first n typed leaves, nesting kept."""
    keep = {leaf.path for leaf in compile_schema(schema).typed()[:n]}

    def walk(spec, path):
        props = {}
        for k, v in spec.get("properties", {}).items():
            p = path + (k,)
            if "properties" in v:
                sub = walk(v, p)
                if sub["properties"]:
                    props[k] = sub
            elif p in keep:
                props[k] = v
        return {**{k: v for k, v in spec.items() if k not in ("properties", "required")}, "properties": props, "required": list(props)}
    return walk(schema, ())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synth-docs"))
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()
    import torch
    rows = [json.loads(line) for line in open(args.data / "cases.jsonl")]
    pages = [r for r in rows if r["level"] == "L3" and r["split"] == "test"][: args.pages]
    base = pages[0]["schema"]
    schemas = {n: first_fields(base, n) for n in SIZES}
    out = {"gpu": torch.cuda.get_device_name(0), "cpu_threads": args.cpu_threads, "pages": len(pages), "reader_ms": {}, "fast_gpu_ms": {}, "fast_cpu_ms": {}}
    reader = Reader.from_pretrained()
    reader.read(pages[0]["file"], schemas[5])
    for n, s in schemas.items():
        out["reader_ms"][n] = round(float(np.median([reader.read(p["file"], s).latency_ms for p in pages])), 1)
        print("reader", n, out["reader_ms"][n], flush=True)
    del reader
    torch.cuda.empty_cache()
    warm = [r for r in rows if r["level"] == "L3" and r["split"] == "train"][:16]
    for dev, key in (("cuda", "fast_gpu_ms"), ("cpu", "fast_cpu_ms")):
        if dev == "cpu":
            torch.set_num_threads(args.cpu_threads)
        fast = FastReader.from_pretrained(device=dev)
        fast.fit([r["file"] for r in warm], base, [r["truth"] for r in warm], epochs=1)
        fast.read(pages[0]["file"], schemas[5])
        for n, s in schemas.items():
            out[key][n] = round(float(np.median([fast.read(p["file"], s).latency_ms for p in pages])), 1)
            print(key, n, out[key][n], flush=True)
        del fast
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "speed.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
