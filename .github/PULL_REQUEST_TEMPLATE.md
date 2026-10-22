## What and why

## How it was tested
- [ ] `pytest` and `ruff check src tests benchmarks` pass
- [ ] GPU tests run (`pytest -m gpu`), if the change touches a reader

## Benchmark impact
If this can move a number: before / after rows from `benchmarks/RESULTS.md`, next to the no-model prior. If you ran
`run_hosted.py`, what it cost.

## Checklist
- [ ] Docs updated (`docs/`, README examples)
- [ ] No data, images with personal information, or keys in the diff
