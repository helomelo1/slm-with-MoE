import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model_config


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()

        inv_freq = 1.0 / base * (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)   # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1) # (seq_len, dim)

        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]
    

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        topk_wt = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        return topk_wt, topk_idx
    

def load_balance_loss(router_logits, n_experts):
    probs = F.softmax(router_logits, dim=-1)
    expert_load = probs.mean(dim=[0, 1])

    target = torch.ones_like(expert_load) / n_experts

    return F.mse_loss(expert_load, target)


class Expert(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_experts, top_k):
        super().__init__()

        self.experts = nn.ModuleList([Expert(d_model, d_ffn) for _ in range(n_experts)])
        self.router = TopKRouter(d_model, n_experts, top_k)

    def forward(self, x):
        B, T, D = x.shape
        wt, idx = self.router(x)            # (B, T, K) | (B, T, K)
        out = torch.zeros_like(x)

        for k in range(self.router.top_k):
            expert_idx = idx[:, :, k]       # (B, T)
            w = wt[:, :, k].unsqueeze(-1)   # (B, T, -1)

            for e_id, expert in enumerate(self.experts):
                mask = (expert_idx == e_id)
                if mask.any():
                    out[mask] += w[mask] * expert(x[mask])

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]

        self.q_proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)
        self.v_proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)
        self.k_proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)
        self.o_proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)

        self.moe = MoELayer(
            cfg["d_model"],
            cfg["d_ffn"],
            cfg["n_experts"],
            cfg["top_k"]
        )

        self.norm1 = nn.RMSNorm(cfg["d_model"])
        self.norm2 = nn.RMSNorm(cfg["d_model"])

    def forward(self, x, rope, kv_cache=None, use_cache=None):
        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        # Project and reshape to [B, H, T, HD]
        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        offset = kv_cache[0].shape[2] if (use_cache and kv_cache is not None) else 0
        cos, sin = rope(offset + T)
        cos, sin = cos[offset:offset + T], sin[offset:offset + T]
        q, k = apply_rope(q, k, cos, sin)

        if use_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)

            new_cache = (k, v)
        else:
            new_cache = None

        if use_cache and kv_cache is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)

        x = self.norm1(x + attn_out)
        x = self.norm1(x + self.moe(x))

        return x, new_cache
    

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.rope = RoPE(cfg["d_model"] // cfg["n_heads"], max_seq_len=cfg["max_seq_len"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])

        self.norm = nn.RMSNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

        self.head.weight = self.embed.weight

    def forward(self, input_ids, kv_caches=None, use_cache=False):
        x = self.embed(input_ids)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(
                x, self.rope,
                kv_cache=cache,
                use_cache=use_cache
            )

            new_caches.append(new_cache)
        
        x = self.norm(x)
        x = self.head(x)

        if use_cache:
            return x, new_caches
        return x