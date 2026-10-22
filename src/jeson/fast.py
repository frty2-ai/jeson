"""The fast reader: a frozen image encoder and ONE small head shared by every field, about 20 ms a page on a GPU.

  image   SigLIP2 so400m patch tokens (frozen), average-pooled from 27x27 to 14x14
  field   mmBERT-base mean embedding of the leaf's text (name, title, description), frozen
  option  mmBERT-base mean embedding of "<field name>: <option>", frozen
  head    query = MLP(field); `depth` blocks of [cross-attention over the projected patches, MLP]; then
          boolean: one logit from the query; choice / score: query . MLP(option), softmax over the declared options

There are no per-field parameters and no JSON text in the model: the schema's words are the only thing that tells
the head which field it answers, so one head serves any schema, and its cost does not grow with the number of
fields in practice (every field is a query in the same cross-attention). It must be trained on labelled pages
first (FastReader.fit); about a minute on a GPU for a few hundred pages.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .reading import Reading
from .schema import compile_schema

VISION = "google/siglip2-so400m-patch14-384"
TEXT = "jhu-clsp/mmBERT-base"


def _head_cls():
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self, d_img: int, d_txt: int, d: int = 256, depth: int = 2, heads: int = 4, drop: float = 0.1):
            super().__init__()
            self.img = nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, d))
            self.q = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d), nn.GELU(), nn.Linear(d, d))
            self.opt = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d), nn.GELU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList()
            for _ in range(depth):
                self.blocks.append(nn.ModuleDict({
                    "ln": nn.LayerNorm(d), "att": nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True),
                    "ln2": nn.LayerNorm(d), "mlp": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))}))
            self.noul = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.out = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)

        def forward(self, patches, E, field_idx, opt_idx, is_noul, omask):
            """Logits (B, F, O): booleans in slots 0 (true) and 1 (false), choices over their declared options; padding -inf."""
            import torch
            kv = self.drop(self.img(patches))
            q = self.q(E[field_idx])
            for b in self.blocks:
                q = q + b["att"](b["ln"](q), kv, kv, need_weights=False)[0]
                q = q + b["mlp"](b["ln2"](q))
            choice = (self.out(q).unsqueeze(2) * self.opt(E[opt_idx])).sum(-1) / q.shape[-1] ** 0.5
            z = self.noul(q).squeeze(-1) / 2
            noul = torch.full_like(choice, float("-inf"))
            noul[..., 0], noul[..., 1] = z, -z
            logits = torch.where(is_noul.unsqueeze(-1), noul, choice)
            return logits.masked_fill(~omask, float("-inf"))

    return Head


def pool_patches(t, n: int = 14):
    """(g*g, d) patch tokens -> (n*n, d), averaging over the grid."""
    import torch.nn.functional as F
    g = int(round(t.shape[0] ** 0.5))
    x = t.float().T.reshape(1, -1, g, g)
    return F.adaptive_avg_pool2d(x, n).reshape(t.shape[1], n * n).T.half()


def _option_text(leaf, v) -> str:
    return f"{leaf.path[-1].replace('_', ' ')}: {json.dumps(v) if v is None or isinstance(v, bool) else str(v).replace('_', ' ')}"


def _is_bool(leaf) -> bool:
    return leaf.kind == "noul" and not leaf.nullable  # a nullable boolean is a 3-way choice (true / false / null)


class FastReader:
    def __init__(self, vision, processor, text_model, tokenizer, device: str = "cuda", pool: int = 14):
        self.vision, self.processor, self.text_model, self.tokenizer = vision, processor, text_model, tokenizer
        self.device, self.pool = device, pool
        self.head, self.config = None, {}
        self._emb: dict = {}

    @classmethod
    def from_pretrained(cls, vision_id: str = VISION, text_id: str = TEXT, device: str = "cuda") -> FastReader:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        dt = torch.float16 if device == "cuda" else torch.float32
        vision = AutoModel.from_pretrained(vision_id, dtype=dt).vision_model.to(device).eval()
        text = AutoModel.from_pretrained(text_id, dtype=dt).to(device).eval()
        r = cls(vision, AutoProcessor.from_pretrained(vision_id), text, AutoTokenizer.from_pretrained(text_id), device)
        r.config = {"vision": vision_id, "text": text_id}
        return r

    # ---------------------------------------------------------------- encoders

    def encode_images(self, images, batch: int = 16) -> list:
        import torch
        from PIL import Image
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                ims = [im if isinstance(im, Image.Image) else Image.open(im) for im in images[i:i + batch]]
                x = self.processor(images=[im.convert("RGB") for im in ims], return_tensors="pt").to(self.device)
                h = self.vision(pixel_values=x["pixel_values"].to(self.vision.dtype)).last_hidden_state
                out += [pool_patches(t, self.pool).cpu() for t in h]
        return out

    def _texts(self, texts) -> None:
        import torch
        todo = sorted({t for t in texts if t not in self._emb})
        with torch.no_grad():
            for i in range(0, len(todo), 32):
                b = todo[i:i + 32]
                x = self.tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                h = self.text_model(**x).last_hidden_state.float()
                m = x["attention_mask"].unsqueeze(-1).float()
                self._emb.update(zip(b, ((h * m).sum(1) / m.sum(1)).cpu()))

    def _batch(self, items, omax: int):
        """items: (compiled, patches, labels or None) -> tensors for the head, with a text table for this batch."""
        import torch
        texts = []
        for compiled, _, _ in items:
            for leaf in compiled.typed():
                texts.append(leaf.text)
                texts += [_option_text(leaf, v) for v in leaf.values]
        self._texts(texts)
        table = sorted(set(texts))
        row = {t: i for i, t in enumerate(table)}
        E = torch.stack([self._emb[t] for t in table]).float().to(self.device)
        B, Fm = len(items), max(len(c.typed()) for c, _, _ in items)
        fi = torch.zeros((B, Fm), dtype=torch.long)
        oi = torch.zeros((B, Fm, omax), dtype=torch.long)
        om = torch.zeros((B, Fm, omax), dtype=torch.bool)
        om[..., 0] = True  # padded fields keep one finite slot so softmax stays defined; their target is 0
        tg = torch.zeros((B, Fm, omax))
        isb = torch.zeros((B, Fm), dtype=torch.bool)
        for b, (compiled, _, lab) in enumerate(items):
            for j, leaf in enumerate(compiled.typed()):
                vals = leaf.values
                fi[b, j] = row[leaf.text]
                om[b, j, :len(vals)] = True
                isb[b, j] = _is_bool(leaf)
                oi[b, j, :len(vals)] = torch.tensor([row[_option_text(leaf, v)] for v in vals])
                if lab is not None and leaf.key in lab and lab[leaf.key] in vals:
                    tg[b, j, vals.index(lab[leaf.key])] = 1.0
        patches = torch.stack([p for _, p, _ in items]).to(self.device).float()
        d = self.device
        return patches, E, fi.to(d), oi.to(d), isb.to(d), om.to(d), tg.to(d)

    # ---------------------------------------------------------------- train / read

    def fit(self, images, schemas, labels, epochs: int = 30, batch: int = 16, lr: float = 3e-4, weight_decay: float = 0.05,
            seed: int = 0, patches=None) -> FastReader:
        """Train the shared head from scratch on labelled images. `schemas`: one schema for all, or one per image;
        `labels`: one dict per image, leaf key -> value. `patches`: precomputed encode_images output, to reuse."""
        import torch

        from .assemble import flatten
        schemas = schemas if isinstance(schemas, list) else [schemas] * len(images)
        compiled = [compile_schema(s) for s in schemas]
        patches = patches if patches is not None else self.encode_images(images)
        labs = [flatten(lb) if any(isinstance(v, dict) for v in lb.values()) else lb for lb in labels]
        items = list(zip(compiled, patches, labs))
        omax = max(len(leaf.values) for c in compiled for leaf in c.typed())
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = _head_cls()(patches[0].shape[-1], next(iter(self._emb.values())).shape[-1] if self._emb else self._text_dim()).to(self.device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        steps = epochs * ((len(items) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        t0 = time.perf_counter()
        head.train()
        for _ in range(epochs):
            order = rng.permutation(len(items))
            for s in range(0, len(order), batch):
                P, E, fi, oi, isb, om, tg = self._batch([items[j] for j in order[s:s + batch]], omax)
                lp = head(P, E, fi, oi, isb, om).log_softmax(-1).masked_fill(~om, 0.0)
                w = (tg.sum(-1) > 0).float()
                loss = (-(tg * lp).sum(-1) * w).sum() / w.sum().clamp_min(1e-6)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.head = head.eval()
        self.config.update({"d_img": int(patches[0].shape[-1]), "d_txt": int(E.shape[-1]), "omax": int(omax),
                            "train_seconds": round(time.perf_counter() - t0, 1), "pool": self.pool})
        return self

    def _text_dim(self) -> int:
        self._texts(["x"])
        return int(self._emb["x"].shape[-1])

    def read(self, image, schema: dict, patches=None) -> Reading:
        import torch
        if self.head is None:
            raise RuntimeError("the fast head is untrained: call fit() or load()")
        compiled = compile_schema(schema)
        if not compiled.typed():
            return Reading(compiled, {}, {}, 0.0)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = patches if patches is not None else self.encode_images([image])[0]
        omax = max(self.config.get("omax", 2), max(len(leaf.values) for leaf in compiled.typed()))
        with torch.no_grad():
            P, E, fi, oi, isb, om, _ = self._batch([(compiled, p, None)], omax)
            pr = self.head(P, E, fi, oi, isb, om).softmax(-1)[0].cpu()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        dists = {leaf.key: dict(zip(leaf.values, pr[j, :len(leaf.values)].tolist())) for j, leaf in enumerate(compiled.typed())}
        return Reading(compiled, dists, {}, ms)

    def extract(self, image, schema: dict) -> dict:
        return self.read(image, schema).json

    def save(self, path) -> Path:
        """The head's weights (safetensors) and its config (JSON) in a directory."""
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.head.state_dict().items()}, str(path / "head.safetensors"))
        (path / "config.json").write_text(json.dumps(self.config, indent=1))
        return path

    @classmethod
    def load(cls, path, device: str = "cuda") -> FastReader:
        from safetensors.torch import load_file
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        r = cls.from_pretrained(cfg["vision"], cfg["text"], device)
        r.pool = cfg.get("pool", 14)
        head = _head_cls()(cfg["d_img"], cfg["d_txt"]).to(device)
        head.load_state_dict(load_file(str(path / "head.safetensors"), device=device))
        r.head, r.config = head.eval(), cfg
        return r
