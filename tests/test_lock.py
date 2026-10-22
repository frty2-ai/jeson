import numpy as np
import pytest

from jeson import Lock, Reading, compile_schema

SCHEMA = {"type": "object", "properties": {
    "signed": {"type": "boolean"},
    "colour": {"type": "string", "enum": ["red", "green", "blue"]},
    "stamp": {"type": ["boolean", "null"]},
    "always": {"type": "boolean"}}}


def synthetic(n, seed=0):
    """Readings whose features encode the answer along a few directions, plus noise; zero-shot dists are uniform."""
    rng = np.random.default_rng(seed)
    c = compile_schema(SCHEMA)
    readings, labels = [], []
    for _ in range(n):
        lab = {"signed": bool(rng.random() < 0.5), "colour": str(rng.choice(["red", "green", "blue"])),
               "stamp": [True, False, None][rng.integers(3)], "always": True}
        f = {}
        for leaf in c.typed():
            x = rng.normal(size=64)
            x[leaf.values.index(lab[leaf.key]) * 4] += 6.0
            f[leaf.key] = x
        readings.append(Reading(c, {leaf.key: {v: 1 / len(leaf.values) for v in leaf.values} for leaf in c.typed()}, f))
        labels.append(lab)
    return readings, labels


def test_fit_apply_recovers_answers(tmp_path):
    tr, trl = synthetic(80)
    te, tel = synthetic(40, seed=1)
    lock = Lock.fit(tr, trl)
    assert lock.heads["always"] == {"const": True}
    acc = np.mean([lock.apply(r).answers()[k] == lab[k] for r, lab in zip(te, tel) for k in lab])
    assert acc > 0.95
    loaded = Lock.load(lock.save(tmp_path / "x.lock.npz"))
    for r in te[:5]:
        assert lock.apply(r).dists == pytest.approx(loaded.apply(r).dists) if False else True
        a, b = lock.apply(r).dists, loaded.apply(r).dists
        for k in a:
            for v in a[k]:
                assert a[k][v] == pytest.approx(b[k][v], abs=1e-5)


def test_output_is_valid_json_by_construction():
    tr, trl = synthetic(40)
    out = Lock.fit(tr, trl).apply(tr[0]).json
    assert set(out) == {"signed", "colour", "stamp", "always"} and out["colour"] in ("red", "green", "blue")


def test_too_few_labels_keep_zero_shot():
    tr, trl = synthetic(3)
    assert Lock.fit(tr, trl, min_fit=4).heads == {}


def test_lock_refuses_another_schema():
    tr, trl = synthetic(20)
    lock = Lock.fit(tr, trl)
    other = compile_schema({"type": "object", "properties": {"signed": {"type": "boolean"}}})
    with pytest.raises(ValueError):
        lock.apply(Reading(other, {"signed": {True: 0.5, False: 0.5}}, {"signed": np.zeros(64)}))
