"""The lock: fit a schema's own heads on a CPU, in seconds, from a few labelled images.

A Reader's zero-shot answers come from the VLM's own language-model head. A Lock replaces them, field by field, with
a logistic regression on the reader's hidden state at the field marker, fitted on labelled readings of that schema.
Features are L2-normalised, then standardised with a variance floor (sd + 0.1 * median sd): without the floor a
near-constant feature dominates a small fit and one step can take minutes instead of seconds. A field whose labels
never vary in the training set is answered with that constant; a field with fewer than `min_fit` labels keeps the
zero-shot answer.

A lock file is a plain .npz (arrays and JSON metadata, no pickle), so loading one never runs code.

    readings = [reader.read(img, schema) for img in train_images]
    lock = Lock.fit(readings, train_labels)        # labels: one dict per image, leaf key (or nested) -> value
    lock.save("invoice.lock.npz")
    reader.extract("new.jpg", schema, lock=Lock.load("invoice.lock.npz"))
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assemble import flatten
from .reading import Reading

FORMAT = 1


def _norm(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-6)


def schema_hash(compiled) -> str:
    """The typed leaves (key, kind, values) a lock was fitted for; a lock refuses a schema that changed them."""
    sig = [(leaf.key, leaf.kind, [json.dumps(v) for v in leaf.values]) for leaf in compiled.typed()]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:16]


class Lock:
    def __init__(self, heads: dict, schema: str, fit_seconds: float | None = None):
        self.heads = heads              # leaf key -> {"classes": [...], "mu", "sd", "W", "b"} or {"const": value}
        self.schema = schema
        self.fit_seconds = fit_seconds

    @classmethod
    def fit(cls, readings: list[Reading], labels: list[dict], C: float = 0.02, min_fit: int = 4) -> Lock:
        from sklearn.linear_model import LogisticRegression
        if not readings:
            raise ValueError("no readings to fit on")
        compiled = readings[0].compiled
        sig = schema_hash(compiled)
        t0 = time.perf_counter()
        rows = defaultdict(list)
        for rd, lab in zip(readings, labels):
            if schema_hash(rd.compiled) != sig:
                raise ValueError("every reading must come from the same schema")
            flat = flatten(lab) if any(isinstance(v, dict) for v in lab.values()) else lab
            for leaf in rd.compiled.typed():
                if leaf.key in flat and leaf.key in rd.features and flat[leaf.key] in leaf.values:
                    rows[leaf.key].append((_norm(rd.features[leaf.key]), flat[leaf.key]))
        heads = {}
        for key, r in rows.items():
            y = [v for _, v in r]
            if len(r) < min_fit:
                continue
            if len(set(map(json.dumps, y))) == 1:
                heads[key] = {"const": y[0]}
                continue
            X = np.stack([x for x, _ in r])
            sd0 = X.std(0)
            mu, sd = X.mean(0), sd0 + 0.1 * float(np.median(sd0)) + 1e-6
            yk = [json.dumps(v) for v in y]  # sklearn needs one label type; None and bools survive as JSON
            clf = LogisticRegression(C=C, max_iter=1000).fit((X - mu) / sd, yk)
            heads[key] = {"classes": [json.loads(c) for c in clf.classes_], "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                          "W": clf.coef_.astype(np.float32), "b": clf.intercept_.astype(np.float32)}
        return cls(heads, sig, round(time.perf_counter() - t0, 3))

    def apply(self, reading: Reading) -> Reading:
        if schema_hash(reading.compiled) != self.schema:
            raise ValueError("this lock was fitted for a different schema")
        leaves = {leaf.key: leaf for leaf in reading.compiled.typed()}
        new = {}
        for key, h in self.heads.items():
            leaf = leaves.get(key)
            if leaf is None:
                continue
            if "const" in h:
                new[key] = {v: float(v == h["const"]) for v in leaf.values}
                continue
            if key not in reading.features:
                continue
            x = (_norm(reading.features[key]) - h["mu"]) / h["sd"]
            z = h["W"] @ x + h["b"]
            if len(h["classes"]) == 2:  # sklearn's binary form: one row of weights, for classes[1]
                p1 = 1.0 / (1.0 + np.exp(-z[0]))
                probs = [1.0 - p1, p1]
            else:
                e = np.exp(z - z.max())
                probs = (e / e.sum()).tolist()
            pp = {json.dumps(c): float(p) for c, p in zip(h["classes"], probs)}
            new[key] = {v: pp.get(json.dumps(v), 0.0) for v in leaf.values}
        return reading.with_dists(new)

    def save(self, path) -> Path:
        path = Path(path)
        arrays, meta = {}, {"format": FORMAT, "schema": self.schema, "fit_seconds": self.fit_seconds, "heads": []}
        for i, (key, h) in enumerate(self.heads.items()):
            if "const" in h:
                meta["heads"].append({"key": key, "const": h["const"]})
                continue
            meta["heads"].append({"key": key, "classes": h["classes"], "i": i})
            for name in ("mu", "sd", "W", "b"):
                arrays[f"{name}_{i}"] = h[name]
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        return path

    @classmethod
    def load(cls, path) -> Lock:
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(z["meta"].tobytes().decode())
        if meta.get("format") != FORMAT:
            raise ValueError(f"unknown lock format {meta.get('format')}")
        heads = {}
        for h in meta["heads"]:
            if "const" in h:
                heads[h["key"]] = {"const": h["const"]}
            else:
                heads[h["key"]] = {"classes": h["classes"], **{n: z[f"{n}_{h['i']}"] for n in ("mu", "sd", "W", "b")}}
        return cls(heads, meta["schema"], meta.get("fit_seconds"))
