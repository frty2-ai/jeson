# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's **"Report a vulnerability"** button on the repository's
Security tab (private vulnerability reporting). Do not open a public issue. We aim to acknowledge within 3 working
days and to agree a disclosure date with you.

## What is in scope

- Loading a lock file must never execute code: locks are `.npz` arrays plus JSON metadata, loaded with
  `allow_pickle=False`. Anything that makes a crafted lock or FastReader checkpoint execute code is in scope.
- The prompt jeson sends tells the model to ignore instructions written in the image, and a typed field can only
  take an allowed value, so injected text cannot add fields or change the JSON's shape. It can still bias an
  answer. Reports of injected text that flips answers on realistic pages are welcome.
- `benchmarks/run_hosted.py` reads API keys from environment variables only and never writes them to disk.

## Data

jeson runs locally and sends nothing anywhere, except `benchmarks/run_hosted.py`, which sends benchmark pages to
the hosted providers you choose. Never run it on documents containing personal data.
