"""
SLM-MoE — Experiments: OOD Evaluation, and POS Affinity Analysis.

  python experiments.py --checkpoint checkpoints/final.pt        # all
  python experiments.py --checkpoint checkpoints/final.pt --ood
  python experiments.py --checkpoint checkpoints/final.pt --pos
"""

import argparse
import math
import json
import os
from collections import defaultdict

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_dataset

from config import model_config
from utils import LanguageModel


def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt.get("config", model_config)
    model = LanguageModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded ({n / 1e6:.1f}M params) on {device}\n")
    return model, cfg


@torch.no_grad()
def sentence_log_prob(model, sp, text, device, max_len=256):
    ids = ([sp.bos_id()] + sp.encode(text))[:max_len]
    x   = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y   = torch.tensor([ids[1:]],  dtype=torch.long, device=device)
    logits   = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp  = log_probs[0, torch.arange(y.size(1)), y[0]]
    return token_lp.sum().item(), y.size(1)

# Experiment 1 — OOD Perplexity

def run_ood(model, sp, device, max_samples=2000):
    print("=" * 60)
    print("  Experiment 1 — OOD Perplexity (WikiText-2)")
    print("=" * 60)

    results = {}

    print("  Loading TinyStories/validation...")
    ts_ds = load_dataset("roneneldan/TinyStories", split="validation")
    ts_loss = ts_tokens = 0
    for ex in tqdm(ts_ds.select(range(min(max_samples, len(ts_ds)))), desc="  TinyStories val"):
        lp, n = sentence_log_prob(model, sp, ex["text"], device)
        ts_loss -= lp; ts_tokens += n

    ts_ppl = math.exp(ts_loss / ts_tokens)
    results["tinystories_val_ppl"] = round(ts_ppl, 4)
    print(f"\n  TinyStories val  →  perplexity = {ts_ppl:.2f}")

    print("\n  Loading WikiText-2/test...")
    wt_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_loss = wt_tokens = 0
    for ex in tqdm(wt_ds, desc="  WikiText-2 test"):
        text = ex["text"].strip()
        if len(text) < 10:
            continue
        lp, n = sentence_log_prob(model, sp, text, device)
        wt_loss -= lp; wt_tokens += n

    wt_ppl = math.exp(wt_loss / wt_tokens)
    results["wikitext2_test_ppl"] = round(wt_ppl, 4)
    results["ood_ratio"] = round(wt_ppl / ts_ppl, 4)

    print(f"\n  ┌─ OOD Summary ──────────────────────────────────────┐")
    print(f"  │  TinyStories val ppl  : {ts_ppl:>10.2f}  (in-distribution) │")
    print(f"  │  WikiText-2 test ppl  : {wt_ppl:>10.2f}  (out-of-domain)   │")
    print(f"  │  OOD / ID ratio       : {wt_ppl / ts_ppl:>10.2f}x                  │")
    print(f"  └────────────────────────────────────────────────────┘\n")
    return results

# Experiment 2 — POS Affinity Analysis

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PUNCT",
            "DET",  "ADP",  "PRON", "PART", "NUM",  "OTHER"]


def hook_token_routing(model, n_layers, n_experts):
    routing = {i: [] for i in range(n_layers)}
    handles = []
    for lid, block in enumerate(model.blocks):
        def _h(l):
            def fn(_, __, out):
                _, idx = out
                routing[l].extend(idx[:, :, 0].view(-1).tolist())
            return fn
        handles.append(block.moe.router.register_forward_hook(_h(lid)))
    return routing, handles


def unhook(handles):
    for h in handles:
        h.remove()


def run_pos_affinity(model, sp, device, max_samples=1000, out_dir="plots"):
    print("=" * 60)
    print("  Experiment 3 — POS Affinity Analysis")
    print("=" * 60)

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        print("  [ERROR] pip install spacy && python -m spacy download en_core_web_sm"); return

    n_layers  = model_config["n_layers"]
    n_experts = model_config["n_experts"]
    affinity  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    baseline  = defaultdict(int)

    print(f"  Loading TinyStories/validation ({max_samples} stories)...")
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    ds = ds.select(range(min(max_samples, len(ds))))

    for ex in tqdm(ds, desc="  POS affinity"):
        text = ex["text"].strip()
        if not text:
            continue

        sp_ids  = sp.encode(text)
        sp_toks = [sp.id_to_piece(i) for i in sp_ids]
        if not sp_ids:
            continue

        doc = nlp(text)
        char_to_pos = {}
        for tok in doc:
            for c in range(tok.idx, tok.idx + len(tok.text)):
                char_to_pos[c] = tok.pos_

        sp_pos_tags = []
        cursor = 0
        for piece in sp_toks:
            clean = piece.lstrip("▁")
            pos   = char_to_pos.get(cursor, "OTHER")
            sp_pos_tags.append(pos if pos in POS_TAGS else "OTHER")
            cursor += len(clean)

        input_ids = torch.tensor([[sp.bos_id()] + sp_ids], dtype=torch.long, device=device)
        routing, handles = hook_token_routing(model, n_layers, n_experts)
        with torch.no_grad():
            model(input_ids)
        unhook(handles)

        for layer_id in range(n_layers):
            token_experts = routing[layer_id][1: 1 + len(sp_pos_tags)]
            for expert_id, pos in zip(token_experts, sp_pos_tags):
                affinity[layer_id][expert_id][pos] += 1
                if layer_id == 0:
                    baseline[pos] += 1

    total_baseline = sum(baseline.values()) or 1
    print(f"\n  Baseline POS distribution:")
    for pos in POS_TAGS:
        print(f"    {pos:<8s} {baseline[pos] / total_baseline * 100:5.1f}%")

    results = {"layers": {}}
    print()

    for lid in range(n_layers):
        print(f"  Layer {lid}:")
        results["layers"][lid] = {}
        for eid in range(n_experts):
            counts = affinity[lid][eid]
            total  = sum(counts.values()) or 1
            dist   = {pos: round(counts[pos] / total * 100, 2) for pos in POS_TAGS}
            top3   = sorted(dist.items(), key=lambda x: -x[1])[:3]
            ratios = {pos: dist[pos] / (baseline[pos] / total_baseline * 100)
                      if baseline[pos] > 0 else 0 for pos in POS_TAGS}
            top_pos, top_ratio = max(ratios.items(), key=lambda x: x[1])

            results["layers"][lid][eid] = {
                "distribution": dist, "top3": top3,
                "peak_affinity_pos": top_pos,
                "peak_affinity_ratio": round(top_ratio, 2),
            }
            top3_str = "  ".join(f"{p} {v:.1f}%" for p, v in top3)
            print(f"    E{eid}  top3: {top3_str}   [peak: {top_pos} ×{top_ratio:.1f}]")
        print()

    _save_pos_heatmap(affinity, baseline, n_layers, n_experts, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "pos_affinity.json")
    serialisable = {str(lid): {str(eid): results["layers"][lid][eid]
                    for eid in range(n_experts)} for lid in range(n_layers)}
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Results saved → {json_path}")
    return results


def _save_pos_heatmap(affinity, baseline, n_layers, n_experts, out_dir):
    total_baseline = sum(baseline.values()) or 1
    n_pos  = len(POS_TAGS)
    n_rows = n_layers * n_experts
    mat    = np.ones((n_rows, n_pos))

    for lid in range(n_layers):
        for eid in range(n_experts):
            row    = lid * n_experts + eid
            counts = affinity[lid][eid]
            total  = sum(counts.values()) or 1
            for j, pos in enumerate(POS_TAGS):
                base = baseline[pos] / total_baseline * 100
                mat[row, j] = (counts[pos] / total * 100) / base if base > 0 else 1.0

    row_labels = [f"L{lid} E{eid}" for lid in range(n_layers) for eid in range(n_experts)]
    fig, ax = plt.subplots(figsize=(14, n_rows * 0.55 + 1.5))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(n_pos)); ax.set_xticklabels(POS_TAGS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(n_rows):
        for j in range(n_pos):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}×", ha="center", va="center",
                    fontsize=7, color="black" if 0.5 < v < 2.5 else "white")
    plt.colorbar(im, ax=ax, label="Routing ratio vs. baseline (1.0 = neutral)")
    ax.set_title("POS Affinity per Expert per Layer\n(expert_routing% / corpus_baseline%)", fontsize=11)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pos_affinity_heatmap.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Heatmap saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint",  required=True)
    pa.add_argument("--sp_model",    default="tokenizer/tinystories_sp.model")
    pa.add_argument("--device",      default=None)
    pa.add_argument("--out_dir",     default="plots")
    pa.add_argument("--ood",         action="store_true")
    pa.add_argument("--pos",         action="store_true")
    pa.add_argument("--max_samples", type=int, default=1000)
    pa.add_argument("--max_pairs",   type=int, default=500)
    args = pa.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, _ = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    run_all     = not (args.ood or args.pos)
    all_results = {}

    if run_all or args.ood:
        r = run_ood(model, sp, device, max_samples=args.max_samples)
        if r: all_results["ood"] = r

    if run_all or args.pos:
        r = run_pos_affinity(model, sp, device,
                             max_samples=args.max_samples, out_dir=args.out_dir)
        if r: all_results["pos"] = r

    if all_results:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "experiment_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  All results saved → {out_path}")


if __name__ == "__main__":
    main()