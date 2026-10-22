#!/usr/bin/env python3
"""Hosted models on the same pages, with strict structured output where the provider supports it. PAID: every call
is charged to a file-locked ledger, and the run stops when a provider reaches --max-usd.

Same input for everyone: the page image, the task prompt, and the schema in the text, plus the provider's strict
JSON-schema mode (OpenRouter `response_format` json_schema with `provider.require_parameters`; Anthropic
`output_config.format`). Ordinals are sent as string enums so every provider sees the same types. Thinking is off,
or at the lowest effort the model allows. Latency is wall clock from the caller, one request at a time; `--batch`
sends the untimed rest of an Anthropic run through the Batch API (half price). `--prompt-only` drops the strict mode
and parses the outermost JSON object from the text: Anthropic's strict mode refuses the 107-field L3 schema
("compiled grammar is too large").

  export OPENROUTER_API_KEY=...  ANTHROPIC_API_KEY=...
  python benchmarks/run_hosted.py --provider openrouter --model google/gemini-2.5-flash-lite --level L1 --n 50 --timed 20
  python benchmarks/run_hosted.py --provider anthropic --model claude-haiku-5-5 --level L3 --n 50 --timed 20 --batch --prompt-only
  python benchmarks/run_hosted.py --spent
Out: benchmarks/results/hosted/<model>-<level>[-prompt]-n<N>.json; raw answers under --work (not in the repo).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jeson import compile_schema, flatten

HERE = Path(__file__).resolve().parent
ANTHROPIC_PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}  # $/M tokens
INSTR = "\n\nAnswer with one JSON object only, that matches this JSON schema exactly (every property, no others):\n"
REASONING = {"openai/gpt-5-nano": {"effort": "low"}, "google/gemini-3.8-flash": {"effort": "minimal"}}
NO_STRUCTURED_FLAG = ("qwen/qwen3.7-flash",)  # served without OpenRouter's structured-outputs flag: no require_parameters
LOCK = threading.Lock()


def clean_error(e) -> str:
    """An error's text without request ids (they identify the caller's account)."""
    return re.sub(r"'(id|request_id)': '[^']*',?\s*", "", str(e))[:300]


class Ledger:
    def __init__(self, work: Path, caps: dict):
        self.path, self.caps = work / "ledger.json", caps
        work.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {"openrouter": 0.0, "anthropic": 0.0, "calls": 0}

    def charge(self, provider: str, cost: float) -> None:
        """Add a call's cost under an OS file lock (several runs may share one ledger); stop at the cap."""
        import fcntl
        with LOCK, open(self.path.with_suffix(".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            led = self.read()
            led[provider] = round(led[provider] + cost, 6)
            led["calls"] += 1
            self.path.write_text(json.dumps(led, indent=1))
            fcntl.flock(lk, fcntl.LOCK_UN)
        if led[provider] >= self.caps[provider]:
            raise SystemExit(f"{provider} reached its cap: ${led[provider]:.4f} of ${self.caps[provider]:.2f}")


def hosted_schema(schema: dict):
    """Typed fields only, x- keywords dropped, ordinals as string enums. Returns (schema, keys whose values are ints)."""
    ints = set()

    def walk(s, path):
        props = {}
        for k, v in s.get("properties", {}).items():
            p = f"{path}{k}"
            if v.get("type") == "object":
                sub = walk(v, p + ".")
                if sub["properties"]:
                    props[k] = sub
            elif v.get("type") == "boolean":
                props[k] = {"type": "boolean", "description": v.get("description", "")}
            elif "enum" in v and v.get("type") == "string":
                props[k] = {"type": "string", "enum": v["enum"], "description": v.get("description", "")}
            elif v.get("type") == "integer":
                vals = v.get("enum") or list(range(v["minimum"], v["maximum"] + 1))
                props[k] = {"type": "string", "enum": [str(x) for x in vals], "description": v.get("description", "")}
                ints.add(p)
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    return walk(schema, ""), ints


def b64(path: str) -> tuple[str, str]:
    mt = "image/png" if path.endswith(".png") else "image/jpeg"
    return mt, base64.standard_b64encode(Path(path).read_bytes()).decode()


def call_openrouter(model, page, hs, max_tokens, prompt_only):
    import httpx
    mt, data = b64(page["file"])
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}},
                                                      {"type": "text", "text": page["prompt"] + INSTR + json.dumps(hs, ensure_ascii=False)}]}]}
    if not prompt_only:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "observation", "strict": True, "schema": hs}}
        if not model.startswith(NO_STRUCTURED_FLAG):
            body["provider"] = {"require_parameters": True}
    if model.startswith("openai/gpt-5"):
        body.pop("temperature")  # reasoning models take no temperature; require_parameters would refuse to route it
    if model in REASONING:
        body["reasoning"] = REASONING[model]
    t0 = time.perf_counter()
    r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                   headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    dt = time.perf_counter() - t0
    j = r.json()
    if r.status_code >= 400 or "choices" not in j:
        raise RuntimeError(f"{r.status_code}: {str(j)[:300]}")
    u = j.get("usage") or {}
    return j["choices"][0]["message"].get("content") or "", dt, float(u.get("cost") or 0.0)


def anthropic_params(model, page, hs, max_tokens, prompt_only):
    mt, data = b64(page["file"])
    p = {"model": model, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}},
                                                   {"type": "text", "text": page["prompt"] + INSTR + json.dumps(hs, ensure_ascii=False)}]}],
         "output_config": {} if prompt_only else {"format": {"type": "json_schema", "schema": hs}}}
    if model == "claude-haiku-5-5":
        p["thinking"] = {"type": "disabled"}
    elif model == "claude-sonnet-5-5":
        p["thinking"] = {"type": "between_tools"}
        p["output_config"]["effort"] = "low"
    else:  # Opus: thinking cannot be disabled; lowest effort
        p["output_config"]["effort"] = "low"
    if not p["output_config"]:
        del p["output_config"]
    return p


def anthropic_cost(model, usage, batch: bool) -> float:
    pi, po = ANTHROPIC_PRICE[model]
    return (usage.input_tokens * pi + usage.output_tokens * po) / 1e6 * (0.5 if batch else 1.0)


def batch_anthropic(client, model, pages, schemas, max_tokens, prompt_only, ledger):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs = [Request(custom_id=f"p{i}", params=MessageCreateParamsNonStreaming(**anthropic_params(model, p, schemas[p["id"]][0], max_tokens, prompt_only)))
            for i, p in enumerate(pages)]
    b = client.messages.batches.create(requests=reqs)
    print(f"  batch {b.id}: {len(reqs)} requests", flush=True)
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(20)
    out = {}
    for res in client.messages.batches.results(b.id):
        page = pages[int(res.custom_id[1:])]
        if res.result.type == "succeeded":
            m = res.result.message
            cost = anthropic_cost(model, m.usage, True)
            ledger.charge("anthropic", cost)
            out[page["id"]] = {"text": next((x.text for x in m.content if x.type == "text"), ""), "latency_s": None, "cost": cost, "batch": True}
        else:
            out[page["id"]] = {"text": "", "latency_s": None, "cost": 0.0, "error": res.result.type}
    return out


def score(pages, answers, schemas):
    """Agreement over typed fields: a field missing or outside its declared values counts as wrong."""
    ok = n = valid = 0
    for p in pages:
        text = answers[p["id"]].get("text") or ""
        try:
            ans = json.loads(text[text.index("{"): text.rindex("}") + 1]) if "{" in text else None
            valid += ans is not None
        except ValueError:
            ans = None
        flat = flatten(ans) if isinstance(ans, dict) else {}
        ints = schemas[p["id"]][1]
        for leaf in compile_schema(p["schema"]).typed():
            v = flat.get(leaf.key, "__missing__")
            if leaf.key in ints and isinstance(v, str) and v.lstrip("-").isdigit():
                v = int(v)
            ok += v == p["truth"].get(leaf.key) and type(v) is type(p["truth"].get(leaf.key))
            n += 1
    return round(ok / max(1, n), 4), valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openrouter", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--level", choices=["L1", "L2", "L3"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--timed", type=int, default=None, help="how many of the n run one at a time for latency (default: all)")
    ap.add_argument("--batch", action="store_true", help="Anthropic: the untimed rest through the Batch API")
    ap.add_argument("--threads", type=int, default=4, help="OpenRouter: concurrency for the untimed rest")
    ap.add_argument("--prompt-only", action="store_true", help="no structured output: the schema only in the prompt")
    ap.add_argument("--data", type=Path, default=Path("data/synth-docs"))
    ap.add_argument("--work", type=Path, default=Path("data/hosted"))
    ap.add_argument("--max-usd", type=float, nargs=2, default=[2.0, 5.0], metavar=("OPENROUTER", "ANTHROPIC"))
    ap.add_argument("--spent", action="store_true")
    args = ap.parse_args()
    ledger = Ledger(args.work, {"openrouter": args.max_usd[0], "anthropic": args.max_usd[1]})
    if args.spent:
        print(json.dumps(ledger.read(), indent=1))
        return
    rows = [json.loads(line) for line in open(args.data / "cases.jsonl")]
    pages = [r for r in rows if r["level"] == args.level and r["split"] == "test"][: args.n]
    schemas = {p["id"]: hosted_schema(p["schema"]) for p in pages}
    max_tokens = {"L1": 800, "L2": 2500, "L3": 6000}[args.level] + (6000 if args.model == "claude-opus-5-5" else 0)
    timed = pages[: args.timed if args.timed is not None else len(pages)]
    rest = pages[len(timed):]
    client = None
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
    answers = {}

    def one(p):
        hs = schemas[p["id"]][0]
        try:
            if args.provider == "openrouter":
                text, dt, cost = call_openrouter(args.model, p, hs, max_tokens, args.prompt_only)
            else:
                t0 = time.perf_counter()
                m = client.messages.create(**anthropic_params(args.model, p, hs, max_tokens, args.prompt_only))
                dt, cost = time.perf_counter() - t0, anthropic_cost(args.model, m.usage, False)
                text = next((b.text for b in m.content if b.type == "text"), "")
        except SystemExit:
            raise
        except Exception as e:  # a failed call is a result: scored as wrong, not retried
            return p["id"], {"text": "", "latency_s": None, "cost": 0.0, "error": clean_error(e)}
        ledger.charge(args.provider, cost)
        return p["id"], {"text": text, "latency_s": dt, "cost": cost}

    for p in timed:
        k, v = one(p)
        answers[k] = v
    if rest:
        if args.provider == "anthropic" and args.batch:
            answers.update(batch_anthropic(client, args.model, rest, schemas, max_tokens, args.prompt_only, ledger))
        else:
            with ThreadPoolExecutor(args.threads) as ex:
                for k, v in ex.map(one, rest):
                    v["latency_s"] = None  # concurrent calls do not count for latency
                    answers[k] = v
    agree, valid = score(pages, answers, schemas)
    lat = [a["latency_s"] for a in answers.values() if a.get("latency_s")]
    serial = sum(a["cost"] * (2 if a.get("batch") else 1) for a in answers.values())  # what the same calls cost one at a time
    slug = args.model.replace("/", "_") + f"-{args.level}" + ("-prompt" if args.prompt_only else "") + f"-n{len(pages)}"
    (args.work / "answers").mkdir(parents=True, exist_ok=True)
    (args.work / "answers" / f"{slug}.json").write_text(json.dumps(answers, ensure_ascii=False))
    res = {"provider": args.provider, "model": args.model, "level": args.level, "mode": "prompt_only" if args.prompt_only else "strict",
           "n": len(pages), "agreement": agree, "valid_json": valid, "errors": sum(1 for a in answers.values() if a.get("error")),
           "first_error": next((a["error"] for a in answers.values() if a.get("error")), None),
           "latency_s_p50": round(float(np.percentile(lat, 50)), 3) if lat else None, "latency_n": len(lat),
           "cost_usd": round(sum(a["cost"] for a in answers.values()), 5), "usd_per_1k_serial": round(serial / len(pages) * 1000, 4)}
    out = HERE / "results" / "hosted"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
