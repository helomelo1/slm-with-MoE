# SLM-MoE
### A 12M Parameter Decoder-only Transformer with Sparse Mixture of Experts

> Built from scratch in PyTorch · Trained on TinyStories

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | ~12M |
| Layers | 6 |
| Hidden dim (`d_model`) | 256 |
| Attention heads | 4 (head_dim = 64, RoPE) |
| Experts per layer | 4 (Top-2 routing) |
| Expert FFN dim | 512 |
| Vocabulary | 16,000 (SentencePiece BPE) |
| Context length | 256 tokens |

Each FFN block is replaced by a sparse MoE layer. A learned router selects the top-2 experts per token; their outputs are weighted and summed.

![Architecture](/plots-and-images/architecture.png)

---

## Training

- **Dataset:** `roneneldan/TinyStories`
- **Optimizer:** AdamW (lr=3e-4, cosine decay, 500-step warmup)
- **Effective batch size:** 64 (32 × 2 grad accum)
- **Steps:** 20,000

**Load balancing:** An auxiliary MSE loss between the empirical routing distribution and a uniform target prevents expert collapse. Implemented via non-invasive PyTorch forward hooks — core model logic stays clean.

---

## Results

### Generation
```
Prompt │ there was once a boy named
Output │ John. John wanted to play a game with his friends...
```

### KV Cache Benchmark *(Apple M-series, 200 tokens, 5 runs)*

| Method | Tokens/sec | Latency | Speedup |
|---|---|---|---|
| No cache | 17.8 | 11.2s | — |
| KV cache | 40.8 | 4.9s | **2.28×** |

### OOD Perplexity

| Dataset | Perplexity | Domain |
|---|---|---|
| TinyStories val | **8.36** | In-distribution |
| WikiText-2 test | 15,238 | Out-of-domain |

The large OOD gap reflects tokenizer-level domain mismatch (vocabulary trained on TinyStories) rather than pure memorization.

---

## Mechanistic Interpretability

### Expert Routing Heatmap
![Expert Routing](/plots-and-images/experts_step_20000.png)

Early layers (L0–L3) show uniform routing (~25% per expert). Deeper layers (L4–L5) develop specialization — L5 E2 routes only **3.2%** of tokens, indicating concentrated computation.

### POS Affinity Analysis
![POS Affinity](/plots-and-images/pos_affinity_heatmap.png)

Routing distributions were mapped against corpus-level POS baselines. Key findings:
- **Expert 2** develops the clearest signature — avoids NUM (0.6×) while preferring ADV/ADJ in deeper layers (L3 E2: ADV **1.9×**, L5 E2: ADJ **1.7×**)
- **L5 E2** shows the most distinct profile: avoids NOUN (0.8×) and PRON (0.7×), prefers ADJ (1.7×) and PART (1.6×)
- Early layers remain near-neutral (~1.0×), confirming specialization is a deep-layer phenomenon

---

## Inference

```bash
# Standard — with expert routing display
python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"

# Fast — KV cache
python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time" --use_cache

# Interactive
python generate.py --checkpoint checkpoints/final.pt --interactive

# Benchmark
python generate.py --checkpoint checkpoints/final.pt --benchmark
```

---

## Experiments

```bash
python experiments.py --checkpoint checkpoints/final.pt          # all
python experiments.py --checkpoint checkpoints/final.pt --ood
python experiments.py --checkpoint checkpoints/final.pt --blimp
python experiments.py --checkpoint checkpoints/final.pt --pos
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python main.py                                          # train
python main.py --resume checkpoints/step_10000.pt      # resume
python main.py --eval_only --resume checkpoints/final.pt
```

**Pre-trained weights (Step 20,000):** [Download from Google Drive](https://drive.google.com/file/d/1sBxGOBGw7pLLTniH9Vni3Eb5OPzgZZWM/view?usp=sharing) → place at `checkpoints/final.pt`