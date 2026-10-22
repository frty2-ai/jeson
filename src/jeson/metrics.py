"""Scoring typed answers against truth: field agreement overall and by primitive, and the no-model prior."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from .reading import Reading


def agreement(readings: list[Reading], truths: list[dict]) -> dict:
    """Share of typed fields whose answer equals the truth, overall and by primitive. Truth dicts are flat
    (leaf key -> value); keys missing from a truth are not scored."""
    tot, by = [0, 0], defaultdict(lambda: [0, 0])
    for rd, truth in zip(readings, truths):
        ans = rd.answers()
        for leaf in rd.compiled.typed():
            if leaf.key not in truth:
                continue
            ok = ans.get(leaf.key, object()) == truth[leaf.key] and type(ans.get(leaf.key)) is type(truth[leaf.key])
            for acc in (tot, by[leaf.kind]):
                acc[0] += ok
                acc[1] += 1
    return {"agreement": round(tot[0] / max(1, tot[1]), 4), "n_fields": tot[1],
            "by_primitive": {k: round(a / n, 4) for k, (a, n) in sorted(by.items())}}


def prior(train_truths: list[dict], test_truths: list[dict], groups_train=None, groups_test=None) -> dict:
    """The no-model baseline: answer every field with its most frequent training value (per group, e.g. per
    template), without looking at the image. Read every model's agreement as its distance above this."""
    gt = groups_train or [None] * len(train_truths)
    gs = groups_test or [None] * len(test_truths)
    freq = defaultdict(Counter)
    for g, t in zip(gt, train_truths):
        for k, v in t.items():
            freq[(g, k)][json.dumps(v)] += 1
    ok = n = 0
    for g, t in zip(gs, test_truths):
        for k, v in t.items():
            if freq[(g, k)]:
                ok += json.loads(freq[(g, k)].most_common(1)[0][0]) == v
                n += 1
    return {"agreement": round(ok / max(1, n), 4), "n_fields": n}
