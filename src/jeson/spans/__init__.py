"""Experimental: strings (words, phrases, sentences) returned as spans of OCR words, copied verbatim, never generated.

`ocr.PageOCR` reads a page into words with boxes; `text` holds the span primitives (copying a run of words, the
exact-match and ANLS scores, and the search for a truth's run that serves as both the OCR ceiling and the training
target). A span head that points into the words is on the roadmap (docs/roadmap.md); it is not shipped yet.
"""
from .text import anls, best_run, clean, exact, nls, norm, text_of

__all__ = ["anls", "best_run", "clean", "exact", "nls", "norm", "text_of"]
