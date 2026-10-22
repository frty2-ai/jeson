# How jeson works

jeson turns a JSON Schema and an image into JSON without generating text. Every typed field is a **choice among
the values the schema allows**, scored in parallel; the JSON is then assembled by code. This page walks through the
four parts: the schema compiler, the readout, the lock, and the fast reader.

```
                 ┌───────────────────────┐
  JSON Schema ──►│ schema compiler       │──► typed leaves: boolean / choice / ordinal (+ nullable)
                 └───────────────────────┘                │
                                                          ▼
                 ┌───────────────────────┐   ┌──────────────────────────────────────┐
  image ────────►│ VLM prefill, once     │──►│ packed rows over the KV cache:       │
                 │ (image + prompt +     │   │ {"field": value} for every value of  │
                 │  schema)              │   │ every field, one or a few forwards   │
                 └───────────────────────┘   └──────────────────────────────────────┘
                                                          │ per-field distributions
                                                          │ + hidden state at each field marker
                                                          ▼
                              ┌──────────────────────────────────────────┐
                              │ Lock (optional): per-schema CPU heads on │
                              │ the hidden states, fitted in seconds     │
                              └──────────────────────────────────────────┘
                                                          │
                                                          ▼
                              assembler: the argmax of each field, checked against its type ──► JSON
```

## 1. The schema compiler (`jeson/schema.py`)

`compile_schema(schema)` walks the schema once and turns every property into a **leaf** with a path, a kind, its
allowed values and whether it is nullable. The decision is made from the schema alone:

| JSON Schema | Kind | Answered by |
|---|---|---|
| `boolean` | `noul` | the readout or a head: true / false |
| `string` (or number) with `enum` | `choice` | one declared value |
| `enum` with `"x-ordered": true`, or `integer` with `minimum`..`maximum` (at most 21 levels) | `score` | one level |
| `"x-primitive": "span"` | `span` | experimental (OCR spans), see below |
| `number`, boxes, arrays of enums | `number`, `box`, `multi_label` | no head yet: listed as unanswered |
| free text, arrays of objects | needs generation | never answered: listed as unanswered |
| `object` | its properties, recursively | |

A nullable leaf (`"type": ["boolean", "null"]`, or `anyOf` with `{"type": "null"}`) gets `null` as one more
allowed value, so abstaining is an answer the model can choose, not an error.

Full reference: [schema.md](schema.md).

## 2. The readout (`jeson/readout.py`)

The readout uses a vision-language model (Qwen3-VL) **as a typed model**. It never calls `generate()`.

1. **One prefill.** The image, the task prompt and the schema go through the model once; the KV cache is kept. Only
   the last position's logits are computed (`logits_to_keep=1`).
2. **Rows.** For each typed leaf and each value it may take, a row is the text `{"<field>": <value>`, e.g.
   `{"currency": "INR"`. A 107-field schema with booleans and 7-way choices is about 280 rows.
3. **Packed scoring.** All rows go into one sequence (a few, if the schema is large) appended after the cached
   prefix. A 4D attention mask lets each row attend to the prefix and to its own earlier tokens only. Each row's
   positions restart where the prefix ends (Qwen3-VL uses multimodal rotary positions, so they are passed
   explicitly, shifted by the prefix's rope delta). Rows are therefore independent of each other, of the field
   order, and of how they are chunked: the same numbers as one full forward per row, at a fraction of the cost.
4. **Distributions.** A value's score is the sum of its tokens' log-probabilities after the part it shares with
   the field's other values (so `"INR"` and `"USD"` are compared only on what differs). The softmax over a
   field's values is its distribution. A value outside the schema is never scored, so it cannot be returned.
5. **Features.** For each field the readout also keeps the last-layer hidden state at the field marker (the last
   token of `{"<field>":`): a vector that summarises what the model reads for that field in that image. This is
   what a Lock trains on.

The language-model head runs only at the positions where a value token is scored. After each chunk the cache is
cropped back to the prefix. Peak memory stays within a 12 GB GPU for the 2B model with 107 fields.

**Why not constrained decoding?** Grammar-constrained generation (XGrammar, Outlines, provider "strict" modes)
also guarantees valid JSON, but it still decodes every token in order, so latency grows with the length of the
answer, and large schemas can exceed what a provider will compile (Anthropic's strict mode refused the benchmark's
107-field schema). The readout scores all fields in parallel; its cost grows with the number of rows, processed in
a few wide forwards, not with sequential decoding steps. A tuned constrained-decoding baseline (vLLM + XGrammar) in
`benchmarks/` is an open issue ([good-first-issues.md](good-first-issues.md)).

## 3. The lock (`jeson/lock.py`)

Zero-shot, the readout is as good as the VLM's reading of your schema's wording. For a schema you use in production,
**lock** it: label a few dozen to a few hundred images, read them once, and fit one small logistic regression per
field on the field-marker features. On one CPU thread this takes 0.5 s for the 9-field level (three templates,
three locks) and 3.5 s for 107 fields, from 300 images. The lock replaces the zero-shot distribution for the fields it covers; fields with too few labels keep the
zero-shot answer, and a field whose label never varies in training is answered with that constant.

- Features are L2-normalised and standardised with a variance floor (sd + 0.1 x median sd). Without the floor a
  near-constant feature dominates a small fit.
- A lock is bound to its schema: it stores a hash of the typed leaves (keys, kinds, values) and refuses a schema
  that changed them.
- A lock file is a plain `.npz` of arrays plus JSON metadata, loaded with `allow_pickle=False`: a lock from someone
  else cannot run code.

The idea behind the lock: in production a schema is a contract that changes rarely. Paying a few seconds of CPU
once per schema, instead of a large model per call, is the trade jeson is built around.

## 4. The fast reader (`jeson/fast.py`)

`FastReader` drops the VLM entirely:

- **Image:** SigLIP2 so400m (frozen) patch tokens, average-pooled from 27x27 to 14x14.
- **Text:** mmBERT-base (frozen) embeds each field's text (name, title, description) and each option as
  `"<field name>: <option>"`.
- **Head:** one 1.9M-parameter module shared by every field: a query per field from its text embedding, two blocks
  of cross-attention over the image patches, then a boolean logit, or a dot product with each option's embedding.

There are no per-field parameters; the schema's words tell the head which field it is answering, so the same head
serves a schema it was not trained on (with lower accuracy). Every field is a query in one cross-attention, so the
cost is flat in the number of fields: about 25 ms a page on a GPU and 0.8 s on 8 CPU threads, from 5 to 100
fields. It must be trained first (`FastReader.fit`): 5-21 s on a GPU for 300 pages.

## 5. Assembly (`jeson/assemble.py`)

The answer for each field is its most probable allowed value. `assemble()` places it at its path and checks it
against its kind (a boolean for `noul`, a declared value for `choice` and `score`, `null` only where allowed). A
failed check is raised as a bug instead of being emitted as malformed JSON. Fields without an answer are listed
under `"_unanswered"`, never invented.

## 6. Experimental: strings as spans (`jeson/spans/`)

Strings printed on the page (names, IDs, addresses, clauses) are planned as **spans**: OCR the page into words with
boxes, have a head point at a start word and an end word, and copy the words verbatim. A span can only be text that
is on the page. What ships today is the OCR side (`PageOCR`, with a second look at uncertain lines that tightens the
crop to the text band, which fixes text in ruled boxes) and the span scoring (`exact`, `anls`, `best_run`). The span
head itself is on the [roadmap](roadmap.md).
