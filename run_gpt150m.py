"""Run the packed GPT-150M model (int5 g64 blocks + int6 embedding) on a Mac / CPU / CUDA.
Setup:   pip install torch numpy tiktoken
Usage:   python run_gpt150m.py gpt150m_int5g64.npz "The Roman Empire was founded when"
         python run_gpt150m.py gpt150m_int5g64.npz            (interactive: type a prompt, Enter)
         python run_gpt150m.py gpt150m_int5g64.npz --check    (verify the KV cache matches the full forward pass)
Generation uses a KV cache: each new token costs one small forward pass instead of re-running the whole text.
"""
import argparse, sys
# Model: ~152M-param GPT (RoPE, RMSNorm, tied embeddings, flash attention via SDPA)
import math, time, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50304      # GPT-2 vocab (50257) padded to a multiple of 64
    block_size: int = 1024
    n_layer: int = 16
    n_head: int = 12
    n_embd: int = 768

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight

def rope_cache(T, D, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):      # x: (B, H, T, D)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
    def forward(self, x): return self.proj(F.gelu(self.fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.n_embd), Attention(c), RMSNorm(c.n_embd), MLP(c)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.nf = RMSNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight           # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():            # scaled init for residual projections
            if n.endswith("proj.weight"): nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))
        cos, sin = rope_cache(c.block_size, c.n_embd // c.n_head, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for b in self.blocks: x = b(x, self.cos, self.sin)
        logits = self.lm_head(self.nf(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=50):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1, :50257] / temperature          # ignore padded vocab entries
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


import numpy as np

def unpack_bits(packed, n, bits):
    m = np.unpackbits(packed, bitorder="little", count=n * bits).reshape(n, bits)
    u = np.zeros(n, dtype=np.uint8)
    for j in range(bits): u |= (m[:, j] << j)
    return u

def dequantize(packed, scale, bits, rows, cols):
    g = cols // scale.shape[1]
    u = unpack_bits(packed, rows * cols, bits).reshape(rows, cols // g, g)
    q = u.astype(np.float32) - 2 ** (bits - 1)
    return (q * scale.astype(np.float32)[..., None]).reshape(rows, cols)

def load_weights(path):
    z = np.load(path)
    cfg, weights = [int(v) for v in z["config"]], {}
    for key in z.files:
        if key.endswith(".q"):
            name = key[:-2]
            bits, rows, cols = [int(v) for v in z[name + ".m"]]
            weights[name] = dequantize(z[key], z[name + ".s"], bits, rows, cols)
        elif key.endswith(".raw"):
            weights[key[:-4]] = z[key]
    return cfg, weights

import tiktoken, time

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

# ---------------- KV-cache inference (uses the trained modules; model code above is unchanged) ----------------
def new_cache(model, device):
    c = model.c; hd = c.n_embd // c.n_head
    dt = model.wte.weight.dtype
    return [(torch.zeros(1, c.n_head, c.block_size, hd, device=device, dtype=dt),
             torch.zeros(1, c.n_head, c.block_size, hd, device=device, dtype=dt)) for _ in range(c.n_layer)]

def attn_cached(attn, x, cos, sin, K, V, pos):
    B, T, C = x.shape
    q, k, v = attn.qkv(x).view(B, T, 3, attn.h, C // attn.h).permute(2, 0, 3, 1, 4)
    q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
    k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
    K[:, :, pos:pos + T] = k; V[:, :, pos:pos + T] = v          # store this chunk's keys/values
    kk, vv = K[:, :, :pos + T], V[:, :, :pos + T]
    # prefill (many tokens, starting at 0) needs the causal mask; a single new token may see everything
    y = F.scaled_dot_product_attention(q, kk, vv, is_causal=(T > 1))
    return attn.proj(y.transpose(1, 2).reshape(B, T, C))

@torch.no_grad()
def forward_cached(model, ids, cache, pos):
    """ids: (1, T) new tokens placed at positions pos..pos+T-1. Returns logits of the LAST position."""
    x = model.wte(ids)
    for blk, (K, V) in zip(model.blocks, cache):
        x = x + attn_cached(blk.attn, blk.n1(x), model.cos, model.sin, K, V, pos)
        x = x + blk.mlp(blk.n2(x))
    return model.lm_head(model.nf(x[:, -1:]))[:, 0, :50257].float()

def sample(logits, history, temperature, top_k, rep_penalty):
    seen = torch.unique(history[0, -128:])
    l = logits[0, seen]
    logits[0, seen] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)
    logits = logits / temperature
    v, _ = torch.topk(logits, top_k)
    logits[logits < v[:, [-1]]] = -float("inf")
    return torch.multinomial(F.softmax(logits, -1), 1)

@torch.no_grad()
def generate(model, enc, prompt, device, max_new=150, temperature=0.8, top_k=50, rep_penalty=1.1, stream=True):
    block = model.c.block_size
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=device)[:, -(block - 1):]
    cache, pos = new_cache(model, device), 0
    gen, printed, t0 = [], 0, time.time()
    if stream: print(prompt, end="", flush=True)
    logits = forward_cached(model, idx, cache, pos); pos = idx.shape[1]          # prefill the prompt
    for _ in range(max_new):
        nxt = sample(logits, idx, temperature, top_k, rep_penalty)
        idx = torch.cat([idx, nxt], dim=1); gen.append(nxt.item())
        if stream:
            text = enc.decode(gen)
            if not text.endswith("\ufffd"):                                        # wait for complete characters
                print(text[printed:], end="", flush=True); printed = len(text)
        if pos >= block:                                                           # context full: keep the newest half, rebuild cache
            keep = idx[:, -(block // 2):]
            cache, pos = new_cache(model, device), 0
            logits = forward_cached(model, keep, cache, pos); pos = keep.shape[1]
        else:
            logits = forward_cached(model, nxt, cache, pos); pos += 1
    if stream:
        print(enc.decode(gen)[printed:])
        print(f"[{len(gen)} tokens in {time.time()-t0:.1f}s = {len(gen)/(time.time()-t0):.1f} tok/s]")
    return prompt + enc.decode(gen)

@torch.no_grad()
def check(model, enc, device, prompt="The Roman Empire was founded when", steps=8):
    """Greedy-decode with the cache and compare logits against a full forward pass at every step."""
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    cache = new_cache(model, device)
    logits = forward_cached(model, idx, cache, 0); pos = idx.shape[1]; worst = 0.0
    for s in range(steps):
        full = model(idx)[0][:, -1, :50257].float()
        d = (full - logits).abs().max().item(); worst = max(worst, d)
        same = full.argmax(-1).item() == logits.argmax(-1).item()
        print(f"step {s}: max |logit diff| = {d:.2e}   same top token: {same}")
        nxt = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits = forward_cached(model, nxt, cache, pos); pos += 1
    print("PASS" if worst < 1e-2 else "FAIL (differences too large)", f"(worst diff {worst:.2e})")

def load_model(path, device):
    cfgv, weights = load_weights(path)
    model = GPT(Config(*cfgv))
    params = dict(model.named_parameters())
    assert set(params) == set(weights), "weight names do not match the model"
    with torch.no_grad():
        for name, w in weights.items(): params[name].copy_(torch.from_numpy(w))
    return model.to(device).eval()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("prompt", nargs="?")
    ap.add_argument("--tokens", type=int, default=150); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--check", action="store_true", help="verify the KV cache against the full forward pass and exit")
    a = ap.parse_args()

    device = pick_device(); print("device:", device)
    model = load_model(a.weights, device)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    enc = tiktoken.get_encoding("gpt2")      # first run downloads the tokenizer file, so needs internet once

    if a.check:
        check(model, enc, device); sys.exit()
    prompts = [a.prompt] if a.prompt else iter(lambda: input("\nprompt> "), None)
    for p in prompts:
        generate(model, enc, p, device, a.tokens, a.temp, a.top_k, a.rep)
        if a.prompt: break