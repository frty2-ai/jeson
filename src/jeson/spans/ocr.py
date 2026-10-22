"""OCR a page into words with boxes in reading order, with a second look at the lines the recogniser is unsure of.

Experimental (install the `spans` extra). The engine is RapidOCR: PaddleOCR's PP-OCR detection and recognition
models on ONNX Runtime, Apache-2.0, CPU.

Two passes. The first is plain OCR of the page. The second looks again only at lines scored below LOOK: the crop is
tightened to its text band (the tallest run of inked rows) and read twice, as is and with thin ruling lines painted
out, and the best-scored reading replaces the old one when it scores higher. Why: a detector box that swallows a
form's box border is several times the text's height; the recogniser shrinks it to a fixed height and the text is
lost (a printed sentence in a remarks box read as 'Pho t t te.' at 0.72; tightened, exact at 0.999). Painting out
lines on the whole page is worse: on blurred scans the letters of a bold word merge into a run as long as a line
('Sector' -> 'Sart').

    ocr = PageOCR()
    page = ocr.read("page.jpg")      # {"words": [[text, score, [x0, y0, x1, y1], line], ...], "lines": [...], ...}

Reading order: lines sorted top to bottom (a line joins the previous one when their vertical centres are within
half a line height), then left to right; words keep their order within the line.
"""
from __future__ import annotations

import time

LOOK = 0.9   # lines scored below this get a second look
THIN = 4     # px: ruling lines are at most this thick; text strokes merged into a band are thicker


def rect(q):
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    return [round(float(min(xs)), 1), round(float(min(ys)), 1), round(float(max(xs)), 1), round(float(max(ys)), 1)]


def unlined(grey):
    """Thin horizontal and vertical ruling lines painted out of a grey crop; anything thicker than THIN stays."""
    import cv2
    import numpy as np
    bw = cv2.adaptiveThreshold(~grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    out = grey.copy()
    for k, thin_axis in (((30, 1), 3), ((1, 30), 2)):  # stats column: 3 = height for horizontal, 2 = width for vertical
        m = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, thin_axis] <= THIN:
                keep[lab == i] = 255
        out[cv2.dilate(keep, np.ones((2, 2), np.uint8)) > 0] = 255
    return out


def band(grey):
    """Rows (start, end) of the tallest run of inked rows in a crop: the text line, without borders above or below."""
    import cv2
    on = (cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0).mean(1) > 0.02
    runs, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    return max(runs, key=lambda r: r[1] - r[0]) if runs else (0, len(on))


def order(lines):
    """Indices of lines in reading order: rows by vertical centre, then left to right."""
    idx = sorted(range(len(lines)), key=lambda i: (lines[i][2][1] + lines[i][2][3]) / 2)
    rows, cur = [], []
    for i in idx:
        b = lines[i][2]
        if cur:
            pb = lines[cur[-1]][2]
            if abs((b[1] + b[3]) / 2 - (pb[1] + pb[3]) / 2) < 0.5 * min(b[3] - b[1], pb[3] - pb[1]):
                cur.append(i)
                continue
            rows.append(cur)
        cur = [i]
    if cur:
        rows.append(cur)
    return [i for row in rows for i in sorted(row, key=lambda i: lines[i][2][0])]


def read(engine, img, dx=0.0, dy=0.0):
    """OCR an image (path or array) -> (lines [[text, score, box]], words per line [[text, score, box]]), offset by dx, dy."""
    out = engine(img, return_word_box=True)
    if out.txts is None:
        return [], []
    sh = lambda b: [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]  # noqa: E731
    lines = [[t, round(float(s), 4), sh(rect(b))] for t, b, s in zip(out.txts, out.boxes, out.scores)]
    wr = list(out.word_results or [])
    words = [[[w[0], round(float(w[1]), 4), sh(rect(w[2])) if w[2] is not None else ln[2]] for w in (wr[i] if i < len(wr) and wr[i] else [(ln[0], ln[1], None)])
              if str(w[0]).strip()] for i, ln in enumerate(lines)]
    return lines, words


def assemble(lines, words):
    out_w, out_l = [], []
    for li in order(lines):
        out_l.append(lines[li])
        out_w += [w + [len(out_l) - 1] for w in words[li]]
    return {"words": out_w, "lines": out_l}


def from_cache(res):
    """A cached first pass -> (lines, words per line) in the cache's line order."""
    per = [[] for _ in res["lines"]]
    for w in res["words"]:
        per[w[3]].append(w[:3])
    return [list(x) for x in res["lines"]], per


class PageOCR:
    def __init__(self, threads: int = 4, look: float = LOOK):
        from rapidocr import RapidOCR
        self.engine = RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": threads, "Global.log_level": "error"})
        self.look = look

    def _read(self, img, dx=0.0, dy=0.0):
        return read(self.engine, img, dx, dy)

    def read(self, src, first_pass: dict | None = None) -> dict:
        """OCR one page (a path). `first_pass`: a cached plain reading of the same page, to skip the first pass."""
        import cv2
        t0 = time.perf_counter()
        if first_pass is not None:
            lines, words = from_cache(first_pass)
            t_first = first_pass.get("seconds", 0.0)
        else:
            lines, words = self._read(str(src))
            t_first = time.perf_counter() - t0
        t1 = time.perf_counter()
        grey = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        H, W = grey.shape
        looked = replaced = 0
        for i, (_text, score, b) in enumerate(lines):
            if score >= self.look:
                continue
            looked += 1
            x0, y0 = max(0, int(b[0]) - 6), max(0, int(b[1]) - 8)
            x1, y1 = min(W, int(b[2]) + 6), min(H, int(b[3]) + 8)
            c = grey[y0:y1, x0:x1]
            a, z = band(c)
            ty = max(0, a - 4)
            tight = c[ty:min(c.shape[0], z + 4)]
            best = None
            for img in (tight, unlined(tight)):
                nl, nw = self._read(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), x0, y0 + ty)
                if nl:
                    sc = sum(ln[1] * len(ln[0]) for ln in nl) / max(1, sum(len(ln[0]) for ln in nl))
                    if best is None or sc > best[0]:
                        best = (sc, nl, nw)
            if best is None:
                continue
            score2, nl, nw = best
            if score2 > score:
                lines[i] = [" ".join(ln[0] for ln in nl), round(score2, 4), b]
                words[i] = [w for ws in nw for w in ws]
                replaced += 1
        return {**assemble(lines, words), "seconds": round(t_first + time.perf_counter() - t1, 3), "second_look": [looked, replaced]}
