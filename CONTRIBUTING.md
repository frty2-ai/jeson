# Contributing to jeson

Thanks for helping. jeson is small on purpose: a schema compiler, a readout, a lock, a fast reader and an
assembler. Most contributions fit in one of those files or in `benchmarks/`.

Good places to start: [docs/good-first-issues.md](docs/good-first-issues.md) (also filed as GitHub issues with the
`good first issue` label), then [docs/roadmap.md](docs/roadmap.md) for larger work. For anything that changes the
public API or adds a dependency, open an issue first so we can agree on the shape.

## Set up

```bash
git clone https://github.com/frty2-ai/jeson && cd jeson
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # the core: compiler, assembler, lock, metrics; no torch
pip install -e ".[models,dev]"          # + torch and transformers, for the Reader and the FastReader
pip install -e ".[all,dev]" && playwright install chromium   # + benchmarks and spans
```

The core tests run on a CPU without torch:

```bash
pytest                 # CPU tests
pytest -m gpu          # tests that need a CUDA GPU and download weights
ruff check src tests benchmarks
```

## How we work

- **Valid by construction.** Nothing in jeson parses or repairs generated text. A head returns an allowed value or
  the assembler raises. Keep it that way: if a new primitive cannot guarantee its type, it is not ready.
- **Every number against a baseline.** A benchmark result is reported next to the no-model prior
  (`jeson.metrics.prior`) on the same pages. A change that "improves accuracy" without the prior next to it will be
  asked for one.
- **Splits by document, never by field or by frame.** Training and test pages never share a document.
- **Reproducible results only.** Anything in `benchmarks/RESULTS.md` must be reproducible from this repository,
  public weights and the synthetic generator. Never add private, personal or customer data to the repository, the
  issues or a PR, including screenshots of it.
- **Paid APIs.** `benchmarks/run_hosted.py` charges a ledger and stops at `--max-usd`. Do not raise the defaults in
  a PR; say what you spent in the PR description.
- **Small, readable code.** Match the surrounding style: docstrings that say why, not what; no new abstraction
  until a second caller needs it.

## Pull requests

1. Branch from `main`; one change per PR.
2. Add or update tests. CPU tests for logic; a `gpu`-marked test for model behaviour.
3. If the change can move a benchmark number, run the benchmark and paste the before/after rows in the PR.
4. Update the docs the change touches (`docs/`, README examples).
5. `pytest` and `ruff check` pass.

By contributing you agree that your contribution is licensed under the Apache License 2.0 (see `LICENSE`).

## Adding a primitive

A primitive is a JSON Schema shape the compiler recognises and a head can answer.

1. `src/jeson/schema.py`: map the schema shape to a kind in `_kind`, and define its allowed values.
2. `src/jeson/assemble.py`: the type check in `check`.
3. A head: in the readout (score its allowed values as rows), in the lock, or in the fast head.
4. Tests for all three, and a benchmark field that exercises it.

## Reporting bugs

Use the bug template. Include the schema (or a minimal one that reproduces it), the image if you can share it (a
synthetic page is best), the jeson version, the model, and the full traceback. Do not attach documents with
personal data.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind, be specific, assume good faith.
