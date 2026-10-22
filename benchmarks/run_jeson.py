#!/usr/bin/env python3
"""jeson on the synthetic levels L1-L3: three arms, scored on the same test pages the hosted models answer.

  zero-shot   Reader (Qwen3-VL-2B readout), no training
  locked      Reader + a Lock per template (a lock is per schema; L1 mixes three) fitted on the level's 300 training
              pages (wall time on ONE CPU thread, summed over the level's locks)
  fast        FastReader (SigLIP2 + one shared head) trained on the same 300 pages

Also the no-model prior (the most frequent training value per template and field): read every agreement as its
distance above it. Scored on the first 50 test pages (the prefix every hosted model answered) and on 100.

  python benchmarks/run_jeson.py --data data/synth-docs
Out: benchmarks/results/jeson-synth.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jeson import FastReader, Lock, Reader
from jeson.metrics import agreement, prior

HERE = Path(__file__).resolve().parent


def load(data: Path, level: str):
    rows = [json.loads(line) for line in open(data / "cases.jsonl")]
    rows = [r for r in rows if r["level"] == level]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synth-docs"))
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--vision", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--text", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3"])
    ap.add_argument("--cache", type=Path, default=Path("data/readings"), help="the Reader's outputs, reused on a rerun")
    args = ap.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    out = {"model": args.model, "vision": args.vision, "text": args.text, "device": torch.cuda.get_device_name(0), "levels": {}}
    data = {lv: load(args.data, lv) for lv in args.levels}
    import pickle  # a local cache this script writes itself; never load one from elsewhere
    readings, reader = {}, None
    args.cache.mkdir(parents=True, exist_ok=True)
    for lv, (train, test) in data.items():
        train, test = train[: args.train], test[: args.test]
        cf = args.cache / f"{lv}-{args.model.replace('/', '_')}-{len(train)}-{len(test)}.pkl"
        if cf.exists():
            readings[lv] = pickle.loads(cf.read_bytes())
            print(f"{lv}: readings from {cf}", flush=True)
            continue
        reader = reader or Reader.from_pretrained(args.model)
        reader.read(test[0]["file"], test[0]["schema"], prompt=test[0]["prompt"])  # warm-up
        readings[lv] = tuple([reader.read(r["file"], r["schema"], prompt=r["prompt"]) for r in rows] for rows in (train, test))
        cf.write_bytes(pickle.dumps(readings[lv]))
        print(f"{lv}: read {len(train) + len(test)} pages", flush=True)
    if reader is not None:
        reader.model = None  # free the VLM before loading the fast reader
    torch.cuda.empty_cache()
    fast = FastReader.from_pretrained(args.vision, args.text)
    for lv, (train, test) in data.items():
        train, test = train[: args.train], test[: args.test]
        tr, te = readings[lv]
        locks, lock_s = {}, 0.0
        with threadpool_limits(1):
            for tpl in sorted({r["template"] for r in train}):
                idx = [i for i, r in enumerate(train) if r["template"] == tpl]
                t0 = time.perf_counter()
                locks[tpl] = Lock.fit([tr[i] for i in idx], [train[i]["truth"] for i in idx])
                lock_s += time.perf_counter() - t0
        locked = [locks[r["template"]].apply(x) for r, x in zip(test, te)]
        fast.fit([r["file"] for r in train], [r["schema"] for r in train], [r["truth"] for r in train])
        fr = [fast.read(r["file"], r["schema"]) for r in test]
        res = {"train_pages": len(train), "test_pages": len(test), "fields_per_page": int(np.median([len(r["truth"]) for r in test])),
               "lock_fit_s_one_thread": round(lock_s, 2), "lock_heads": sum(len(lk.heads) for lk in locks.values()),
               "fast_train_gpu_s": fast.config["train_seconds"],
               "latency_ms_p50": {"zero_shot_and_locked": round(float(np.median([x.latency_ms for x in te])), 1),
                                  "fast_end_to_end": round(float(np.median([x.latency_ms for x in fr])), 1)}}
        for n in (50, len(test)):
            truths = [r["truth"] for r in test[:n]]
            res[f"n{n}"] = {"prior": prior([r["truth"] for r in train], truths, [r["template"] for r in train], [r["template"] for r in test[:n]]),
                            "zero_shot": agreement(te[:n], truths), "locked": agreement(locked[:n], truths), "fast": agreement(fr[:n], truths)}
        out["levels"][lv] = res
        print(lv, {k: (v["agreement"] if isinstance(v, dict) and "agreement" in v else v) for k, v in res["n50"].items()}, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / "jeson-synth.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
