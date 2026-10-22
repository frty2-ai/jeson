"""Command line: read an image against a schema, compile a schema, and fit or apply a lock.

  jeson schema invoice.schema.json                       what each field compiles to
  jeson read page.jpg --schema invoice.schema.json [--lock invoice.lock.npz] [--probs]
  jeson lock fit --schema invoice.schema.json --labels train.jsonl --out invoice.lock.npz
      train.jsonl: one {"image": "path.jpg", "labels": {...}} per line
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _schema(path) -> dict:
    return json.loads(Path(path).read_text())


def cmd_schema(args) -> None:
    from .schema import compile_schema
    c = compile_schema(_schema(args.schema))
    for leaf in c.leaves:
        opts = f" {list(leaf.values)}" if leaf.kind in ("noul", "choice", "score") else ""
        print(f"{leaf.key:40s} {leaf.kind:11s}{' nullable' if leaf.nullable else ''}{opts}")
    for k in c.needs_generation:
        print(f"{k:40s} needs generation (not answered)")


def cmd_read(args) -> None:
    from .lock import Lock
    from .readout import Reader
    reader = Reader.from_pretrained(args.model, device=args.device)
    reading = reader.read(args.image, _schema(args.schema), max_side=args.max_side)
    if args.lock:
        reading = Lock.load(args.lock).apply(reading)
    out = {"json": reading.json, "latency_ms": round(reading.latency_ms or 0, 1)}
    if args.probs:
        out["confidence"] = {k: round(v, 4) for k, v in reading.confidence().items()}
    print(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_lock_fit(args) -> None:
    from .lock import Lock
    from .readout import Reader
    schema = _schema(args.schema)
    rows = [json.loads(line) for line in open(args.labels) if line.strip()]
    reader = Reader.from_pretrained(args.model, device=args.device)
    readings = [reader.read(r["image"], schema, max_side=args.max_side) for r in rows]
    lock = Lock.fit(readings, [r["labels"] for r in rows])
    lock.save(args.out)
    print(f"{len(lock.heads)} heads fitted on {len(rows)} images in {lock.fit_seconds} s -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="jeson", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schema", help="show what each field of a schema compiles to")
    s.add_argument("schema")
    s.set_defaults(fn=cmd_schema)
    for name, fn, helptext in (("read", cmd_read, "read an image against a schema"),):
        r = sub.add_parser(name, help=helptext)
        r.add_argument("image")
        r.add_argument("--schema", required=True)
        r.add_argument("--lock")
        r.add_argument("--probs", action="store_true", help="also print each field's confidence")
        r.set_defaults(fn=fn)
    lk = sub.add_parser("lock", help="fit a per-schema lock")
    lsub = lk.add_subparsers(dest="lock_cmd", required=True)
    f = lsub.add_parser("fit", help="fit a lock from labelled images")
    f.add_argument("--schema", required=True)
    f.add_argument("--labels", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_lock_fit)
    for p in (sub.choices["read"], f):
        p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
        p.add_argument("--device", default="cuda")
        p.add_argument("--max-side", type=int, default=None)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
