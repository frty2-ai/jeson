# jeson: notes for coding agents

jeson turns a JSON Schema and an image into JSON without generating text: every typed field is a choice among the
values the schema allows, scored in parallel, and the JSON is assembled by code. Read README.md and
docs/how-it-works.md before changing anything.

## Layout

- `src/jeson/schema.py`: JSON Schema -> typed leaves (`compile_schema`). `assemble.py`: leaves + answers -> JSON,
  with type checks. `reading.py`: the `Reading` every reader returns.
- `src/jeson/readout.py`: `Reader`, a VLM (Qwen3-VL) read as a typed model: one prefill, then every field's values
  scored in packed rows over the KV cache with a 4D mask and explicit M-RoPE positions. Never calls `generate()`.
- `src/jeson/lock.py`: `Lock`, per-schema logistic heads on the readout's field-marker features, fitted on a CPU;
  saved as `.npz` without pickle.
- `src/jeson/fast.py`: `FastReader`, frozen SigLIP2 + mmBERT + one shared cross-attention head.
- `src/jeson/spans/`: experimental OCR (`PageOCR`) and span scoring.
- `benchmarks/`: the synthetic page generator, our arms, the paid hosted comparison, speed, `report.py` ->
  `RESULTS.md`.

## Commands

```bash
pip install -e ".[dev]"            # core, no torch;  ".[models,dev]" for the readers; ".[all,dev]" for everything
pytest                             # CPU tests;  pytest -m gpu  needs CUDA and downloads weights
ruff check src tests benchmarks
python benchmarks/synth_docs.py --per-level 500 --out data/synth-docs
python benchmarks/run_jeson.py --data data/synth-docs && python benchmarks/report.py
```

## Rules

- Valid by construction: no parsing or repairing generated text anywhere. A head returns an allowed value or the
  assembler raises.
- Report every benchmark number next to the no-model prior on the same pages; split by document.
- Only reproducible results go into `benchmarks/RESULTS.md` (public weights, synthetic data, this code).
- Never commit data: `data/`, images and `.jsonl` are git-ignored. Never commit or print API keys.
- `benchmarks/run_hosted.py` costs money: do not run it, or raise `--max-usd`, without the maintainer's go-ahead.
- Keep lock and checkpoint formats free of pickle.
- Match the code style: dense, docstrings that say why, line length 160, `ruff` clean.
