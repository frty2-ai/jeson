"""Span fields over OCR words: what a span head returns, how it is scored, and where its training target is.

A span is a contiguous run of OCR words (start, end), copied verbatim and joined by single spaces; code then trims
separators the OCR glued to the edge of a word (a trailing comma or colon), which is deterministic, never generated.
Scores: exact match after normalising case and whitespace, and ANLS (1 - normalised edit distance, 0 below 0.5).
"""
from __future__ import annotations

EDGE = ",;:|"


def norm(s: str | None) -> str | None:
    return None if s is None else " ".join(str(s).lower().split())


def clean(s: str) -> str:
    """The span's text with separators glued to its edges removed ('Ltd.,' -> 'Ltd.', 'Ref:' -> 'Ref')."""
    return s.strip().strip(EDGE).strip()


def text_of(words: list, i: int, j: int) -> str:
    """Words i..j copied verbatim. When the run covers several OCR lines, a line with no horizontal overlap with the
    run's widest line is dropped: a label beside a wrapped value ('Terms:' next to its second line) falls between the
    value's lines in reading order but is not part of it. Words are [text, score, box, line, ...]."""
    run = words[i:j + 1]
    if len({w[3] for w in run if len(w) > 3}) > 1:
        ext = {}
        for w in run:
            x0, x1 = ext.get(w[3], (1e9, -1e9))
            ext[w[3]] = (min(x0, w[2][0]), max(x1, w[2][2]))
        r0, r1 = max(ext.values(), key=lambda e: e[1] - e[0])
        keep = {li for li, (x0, x1) in ext.items() if min(x1, r1) - max(x0, r0) > 0}
        run = [w for w in run if w[3] in keep]
    return clean(" ".join(str(w[0]) for w in run))


def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nls(pred: str | None, truth: str | None) -> float:
    """Normalised Levenshtein similarity in [0, 1]; null against null is 1, null against a string is 0."""
    p, t = norm(pred), norm(truth)
    if p is None or t is None:
        return float(p is None and t is None)
    if not p and not t:
        return 1.0
    return 1.0 - lev(p, t) / max(len(p), len(t))


def anls(pred, truth, tau: float = 0.5) -> float:
    s = nls(pred, truth)
    return s if s >= tau else 0.0


def exact(pred, truth) -> bool:
    return norm(pred) == norm(truth)


def best_run(words: list, truth: str, max_len: int = 80) -> tuple[int, int, float]:
    """The contiguous run of words closest to the truth: (start, end, similarity). Exact runs return 1.0. The
    span head's training target and, over all fields, the OCR ceiling."""
    t = norm(truth)
    best = (-1, -1, 0.0)
    if not t or not words:
        return best
    toks = [norm(str(w[0])) for w in words]
    first = t.split()[0]
    starts = [i for i, w in enumerate(toks) if first[:3] in w or w[:3] in first] or range(len(toks))
    for i in starts:
        acc = ""
        for j in range(i, min(len(words), i + max_len)):
            acc = (acc + " " + toks[j]).strip()
            s = nls(clean(acc), t)
            if s > best[2]:
                best = (i, j, s)
            if s == 1.0 or len(acc) > 1.5 * len(t) + 10:
                break
    return best
