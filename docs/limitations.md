# Known limitations

jeson is early. This page lists what does not work yet, measured where we could measure it, so you can decide
whether it fits your use and where to help.

## Accuracy

- **Repeated sections are the main gap.** On flat schemas the locked Reader matches the fast hosted tier (L1:
  within about two points of Claude Haiku and Qwen 3.7 Flash). On pages with repeated sections (an invoice's line
  items, a statement's transactions) it is 15-19 points behind the best hosted models, and only 8 points above the
  no-model prior on the 107-field statement ([RESULTS.md](../benchmarks/RESULTS.md)). Per field, header fields
  are read almost perfectly, while fields of the k-th row are close to chance and do not get worse further down the
  page: the model reads the page but does not find "row k". A fix (section queries with a box head, then reading
  inside each section) is the first item on the [roadmap](roadmap.md).
- **The FastReader trails the locked Reader on flat schemas** (L1: 0.75 vs 0.94 with 300 training pages). It loses
  on choice and ordinal fields (0.52 and 0.42 vs 0.99 and 0.70); pooling the image to 14x14 patches likely loses the
  small print those fields depend on (not yet tested). It is close on the multi-section levels.
- **Zero-shot accuracy depends on wording.** The Reader reads your field names and descriptions; vague descriptions
  give vague answers. Locking a schema removes most of this.
- **A lock learns your data's regularities as well as the page.** A field that is constant in your training labels
  is answered with that constant, whatever the page shows. Check the no-model prior (`jeson.metrics.prior`) on your
  own labels before reading a number as accuracy.

## Types

- Only booleans, choices and ordinals are answered. Strings, numbers, boxes and multi-label fields are listed as
  unanswered (spans are experimental: the OCR side ships, the span head does not yet).
- Arrays of objects of variable length are not supported; repeated structures must be fixed sections.
- `$ref` / `$defs` are not supported.

## Evidence

- **The benchmark is synthetic.** Pages are rendered from three templates and aged like a scan, with truth known by
  construction, split 60/40 by document from one generator. That makes every number exact and reproducible, and it
  also means train and test pages share layouts. Transfer to unseen layouts and to real scans is not yet measured
  in this repository.
- Hosted models were measured on 20-50 pages a level; differences under about three points are noise.
- Some hosted runs lost requests to provider rate limits (HTTP 429); those requests are scored as wrong, and runs
  where more than half failed are not ranked.

## Runtime

- **The Reader needs a GPU.** On a CPU it takes 14-48 s a page (fp32). The FastReader runs on a CPU in under a
  second a page.
- Tested with Qwen3-VL (2B Instruct) on transformers 5; other VLMs need their position handling checked (the
  readout passes multimodal rotary positions explicitly).
- int8 quantisation of the SigLIP2 encoder for CPU serving is not ready: naive int8 changes the answers (in
  development measurements on the L2 pages: feature cosine 0.57-0.94 to fp32 depending on the method), and a head
  retrained on the quantised features recovers only part of the loss. Turning those measurements into a benchmark
  script is an open issue. See the roadmap.
- English page text only has been tested. The encoders are multilingual; nothing in jeson is English-specific, but
  no non-English number exists yet.
