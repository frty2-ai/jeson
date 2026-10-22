# Roadmap

In priority order. Each item says what would count as done. Issues for most of these are in
[good-first-issues.md](good-first-issues.md); bigger items welcome a design discussion first.

## 1. Repeated sections

The largest accuracy gap (see [limitations.md](limitations.md)). Plan:

- **Generator v2**: record every section's, cell's and printed value's DOM box through the scan transform (box
  truth for free), more templates with repeated structure (purchase orders, payslips, GST invoices, schedules, lab
  reports), 3-40 rows, multi-page continuation, depth 3, subtotal rows as distractors, empty tables, and two
  templates held out entirely.
- **Section queries**: the compiler already knows which leaves repeat; give each repeated section a query, train a
  box head on the DOM boxes, and attend inside the section's box when answering its fields. Counts come from the
  detected sections, not from a model that counts.
- **Readout on crops**: read each detected section as a small image with the VLM readout, as teacher and fallback.

Done when: on the benchmark's L2 and L3, row fields are within 5 points of header fields, and the locked models are
within 5 points of the best hosted models.

## 2. Strings as spans

Words, phrases and sentences printed on the page, returned verbatim without generation: OCR words with boxes
(ships, `jeson.spans.PageOCR`), a pointer head that picks the start and end word per field conditioned on its
description, and code that copies the words. A strong candidate warm start is GLiNER2 (Apache-2.0), a schema-driven
span extractor over text, with layout and image features added. Done when: exact match within 3 points of the OCR
ceiling on names, IDs, dates and amounts, ANLS >= 0.90 on sentences, null correct >= 95% on blank fields.

## 3. CPU serving

The FastReader is under a second a page on a CPU in fp32. Measured so far: a runtime change alone (ONNX Runtime,
OpenVINO) buys nothing at fp32; int8 is 2-3x faster but changes the answers; use P-cores only (more threads onto
efficiency cores was slower). Next: SmoothQuant and weight-only int8 with a head retrained on the quantised
features; a smaller encoder (SigLIP2 base, 86M) with its own head. Done when: a quantised FastReader within one
point of fp32 at under 400 ms a page on a desktop CPU.

## 4. Variable-length arrays

Arrays of objects without fixed sections: one record per detected anchor (a row, a line item), fields scored
against each record, with an explicit absent answer. Depends on 1.

## 5. Fewer labels per lock

Pretrain the FastReader on many schemas and templates so a new schema's lock needs tens of labelled pages instead
of hundreds, measured on templates the pretraining never saw.

## Not planned

Free text that is not on the page (summaries, descriptions) needs generation and stays out of scope: jeson lists
such fields as unanswered so a caller can route them to a generative model.
