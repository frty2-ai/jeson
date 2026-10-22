# Issues to get started with

Ready to file on GitHub (one heading per issue; labels in brackets). Each says why it matters, where to start, and
what counts as done. If you pick one up, comment on the issue so nobody duplicates the work.

---

## Support `$ref` / `$defs` in the schema compiler  [good first issue] [schema]

**Why.** Real schemas reuse definitions; today `compile_schema` raises `NotImplementedError` on `$ref`.
**Where.** `src/jeson/schema.py`: resolve local refs (`#/$defs/...`, `#/definitions/...`) before walking; refuse
cycles with a clear error. **Done when.** Tests in `tests/test_schema.py` cover a nested ref, a ref reused twice,
and a cycle.

## Abstain threshold for nullable fields  [good first issue] [api]

**Why.** Callers want "answer null unless confident" without retraining. **Where.** `Reading.json` /
`Reading.answers` in `src/jeson/reading.py`: an optional `min_confidence`; below it a nullable field returns null and
a non-nullable field is listed under `_unanswered`. **Done when.** Unit tests, and the README shows it.

## Lock your own schema: a tutorial notebook  [good first issue] [docs]

**Why.** The fastest way to see what jeson does. **Where.** `examples/`: generate 50 pages with
`benchmarks/synth_docs.py`, read them, fit a lock, compare zero-shot vs locked vs the no-model prior. **Done when.**
Runs top to bottom on a free GPU notebook in under 15 minutes.

## Apple Silicon (MPS) support  [help wanted] [runtime]

**Why.** Many contributors develop on Macs. **Where.** `Reader.from_pretrained` / `FastReader.from_pretrained`
device handling; the readout's 4D mask dtype on MPS. **Done when.** `examples/quickstart.py` runs on `mps` with the
same answers as CUDA on the example page.

## Batch `Reader.read` over many images  [help wanted] [performance]

**Why.** Throughput for back-office workloads. **Where.** `src/jeson/readout.py`: batch the prefill over images of
one schema (padding), then score rows per image. **Done when.** Same distributions as one-at-a-time (within bf16
noise) and a throughput number in `benchmarks/speed.py`.

## Cache the schema prefix for a locked schema  [help wanted] [performance]

**Why.** With a fixed schema, the schema text can sit before the image in the prompt and its KV cache can be reused
across calls, so a 100-field schema stops paying for its own prompt on every page. **Where.** `src/jeson/readout.py`
(prompt order, a reusable cache). **Done when.** Latency at 100 fields drops measurably in `benchmarks/speed.py`,
and the locked accuracy is re-measured with the new prompt order (a lock must be refitted after the change).

## A constrained-decoding baseline (vLLM + XGrammar)  [help wanted] [benchmarks]

**Why.** The fair comparison for "valid JSON from a local model" is tuned constrained decoding, not plain
`generate()`. **Where.** `benchmarks/run_constrained.py`: the same 2B model in vLLM with `guided_json`, on the same
pages; latency vs schema size and agreement. **Done when.** Its rows appear in `benchmarks/RESULTS.md`.

## A CPU quantisation benchmark for the FastReader  [help wanted] [benchmarks] [runtime]

**Why.** int8 is 2-3x faster on CPU but changes the answers with naive methods (docs/limitations.md). **Where.**
`benchmarks/cpu_int8.py`: export the SigLIP2 vision tower to ONNX; compare fp32, ONNX Runtime dynamic int8, OpenVINO
with NNCF SmoothQuant and weight-only int8; retrain the head on each variant's own features. **Done when.** A table
of latency vs agreement on L2 in RESULTS.md.

## New page templates for the generator  [good first issue] [benchmarks]

**Why.** Three templates are too few to measure transfer. **Where.** `benchmarks/synth_docs.py`: add a payslip or a
purchase order with typed fields and truth by construction, following the existing templates. **Done when.** The
new template renders, its truth matches the page on 10 spot-checked pages, and `run_jeson.py` runs on it.

## Non-English pages  [help wanted] [benchmarks]

**Why.** The encoders are multilingual, but there is no number for any other language. **Where.**
`benchmarks/synth_docs.py`: a locale option (Hindi in Devanagari is a good first target; fonts must render in
headless Chromium). **Done when.** A row per language in RESULTS.md.

## Other vision-language models in the Reader  [help wanted] [models]

**Why.** Users will want a 4B or 7B reader, or a different family. **Where.** `src/jeson/readout.py` assumes
Qwen3-VL's multimodal rotary positions (`rope_deltas`); generalise the position handling. **Done when.** A GPU test
shows packed scoring equals a full forward per row for the new model (port the exactness check, below).

## A GPU test: packed scoring equals a full forward  [good first issue] [tests]

**Why.** The readout's correctness rests on rows being independent of packing and order. **Where.**
`tests/test_readout_gpu.py` (marked `gpu`): for a small schema, compare every row's log-probabilities from
`Reader.read` against one plain forward of prefix + row. **Done when.** Passes within bf16 tolerance and documents
the tolerance.

## `jeson serve`: an HTTP endpoint with a lock registry  [help wanted] [api]

**Why.** Most users will call jeson over HTTP. **Where.** `src/jeson/serve.py` (optional `serve` extra): POST an
image and a schema id; locks loaded from a directory by schema hash. **Done when.** One request returns the JSON and
per-field confidence; a load test number in the docs.
