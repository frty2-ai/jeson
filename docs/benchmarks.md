# Benchmarks: method

The numbers are in [benchmarks/RESULTS.md](../benchmarks/RESULTS.md), generated from the result files in
`benchmarks/results/`. This page says how they were made, so you can reproduce them or argue with them.

## The question

On the same pages, against the models people call in production, with each provider's strict structured output:
how accurate, how fast and how expensive is jeson, as the schema grows from 9 flat fields to 107 fields in repeated
sections, and what does locking a schema buy?

## The pages

`benchmarks/synth_docs.py` renders documents in headless Chromium and ages them like a scan (slight rotation, blur,
contrast, JPEG). Every label is known by construction, so every system is graded against the truth the page was
rendered from, not against another model. Names, numbers and IDs are invented.

| Level | What | Typed fields a page | Depth |
|---|---|---|---|
| L1 | a KYC form, an invoice or a bank statement; up to 10 header fields | 9 (7-10) | 1, flat |
| L2 | an invoice with 10 line-item sections (present, category, taxable) | 40 | 2 |
| L3 | a bank statement with 25 transaction sections (present, credit, channel, returned) | 107 | 2 |

500 pages a level, split by document: 300 training pages (60%) and 200 test pages (40%).
Hosted models answered the first 50 test pages of each level (20-40 for the most expensive models); every jeson
number in the headline tables is on the same first 50. With `--seed 7`, the generator reproduces the benchmark's
pages byte for byte.

## The systems

**jeson**, on one RTX 4070 SUPER (12 GB):

- *Reader, zero-shot*: Qwen3-VL-2B-Instruct readout, bf16, no training.
- *Reader + Lock*: a Lock fitted on the 300 training pages' readings. Fit time is wall clock on one CPU thread.
- *FastReader*: SigLIP2 so400m + mmBERT-base + the shared head, trained on the same 300 pages (30 epochs).

**Hosted**, through OpenRouter and the Anthropic API, each with strict JSON-schema output where offered
(OpenRouter `response_format` json_schema with `provider.require_parameters`; Anthropic `output_config.format`):
Gemini 2.5 Flash-Lite, Gemini 3.8 Flash, GPT-4.1 nano, GPT-5 nano, Qwen 3.7 Flash, Llama 4 Scout, Mistral Small,
Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5. Thinking off or at the lowest effort the model allows
(`benchmarks/run_hosted.py` lists the exact settings). Temperature 0 where accepted. Server-side model fallbacks off,
so every number is from the named model. Only typed fields are asked; ordinals are sent as string enums so every
provider sees the same types.

Claude's strict mode refused the 107-field L3 schema ("compiled grammar is too large"). Those requests failed before
generation, at no cost, and are reported as not ranked. The L3 Claude rows marked "schema in prompt" ran without
strict mode: the schema in the prompt, the outermost JSON object parsed from the text.

## The measures

- **Agreement**: share of typed fields equal to the truth (type included: `"9"` is not `9`, except for ordinals
  sent as strings, which are converted back). A missing field, an invalid value or a failed request counts as wrong.
- **No-model prior**: the most frequent training value per template and field, without looking at the page.
  Every agreement should be read as its distance above the prior. On L3 the prior is high (0.73): most transaction
  rows are absent or false.
- **Latency**: p50 wall clock a page, one request at a time (hosted: from the caller, network included).
- **Cost**: hosted rows are the providers' own reported cost at serial list price (Batch API runs are shown at what
  the same calls cost one at a time). jeson rows are serial GPU time at $0.34/h (an RTX 4090 at a community-cloud
  rate, a faster card than the one measured, so jeson is priced conservatively), full utilisation, idle excluded.

## Caveats

- 50 pages a level: differences under about three points are noise.
- One generator, three templates, i.i.d. split: train and test pages share layouts. This is a measure of a locked
  schema on its own document type, not of transfer to unseen layouts or real scans.
- Some hosted requests failed with provider rate limits (HTTP 429). They are scored as wrong; runs where more than
  half failed are listed as not ranked rather than as model results.
- Hosted latency includes the network from one location.

## Reproducing

```bash
pip install -e ".[all]" && playwright install chromium
python benchmarks/synth_docs.py --per-level 500 --seed 7 --out data/synth-docs
python benchmarks/run_jeson.py --data data/synth-docs          # GPU, about 20 minutes
python benchmarks/speed.py --data data/synth-docs
# paid, optional: see run_hosted.py --help; the ledger stops each provider at --max-usd
python benchmarks/report.py                                     # -> benchmarks/RESULTS.md
```
