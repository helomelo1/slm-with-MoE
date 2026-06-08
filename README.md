# SLM-MoE: Small Language Model with Sparse Mixture of Experts

## Overview
This repository contains a custom implementation of a Sparse Mixture of Experts (MoE) Language Model, built entirely from scratch in PyTorch. The primary objective of this project is to explore the training dynamics, routing behaviors, and inference optimizations of MoE architectures in a constrained parameter regime (~12M parameters).

The model is trained on the `roneneldan/TinyStories` dataset, demonstrating that sparse conditional computation can be effectively leveraged to learn syntax, grammar, and basic semantic grouping without requiring billion-parameter scales.

## Architecture

The core architecture is a decoder-only Transformer with sparse MoE layers substituting the standard feed-forward networks (FFNs).

![SLM-MoE Architecture Diagram](/plots-and-images/architecture.png)

### Hyperparameters
- **Parameter Count:** ~12M
- **Layers:** 6
- **Hidden Dimension (`d_model`):** 256
- **Attention:** 4 heads (`head_dim` = 64) with Rotary Positional Embeddings (RoPE)
- **MoE Configuration:** 4 experts per layer, Top-2 routing strategy
- **Expert Dimension (`d_ffn`):** 512
- **Vocabulary Size:** 16,000 (Custom trained SentencePiece BPE)
- **Context Length:** 256 tokens

## Training Dynamics and Load Balancing
A known failure mode of MoE architectures is expert collapse, where the router converges to utilizing only a small subset of experts, effectively reducing the active capacity of the network.

To mitigate this, the training loop implements an auxiliary load-balancing loss. The loss minimizes the Mean Squared Error (MSE) between the empirical routing distribution and a uniform target distribution across the batch. Non-invasive PyTorch forward hooks are utilized to capture the raw router gate logits during the forward pass, ensuring the core model logic remains decoupled from the training objective.

## Mechanistic Interpretability
A core focus of this project is understanding the internal representations formed by the experts.

### Layer-wise Routing Heatmaps
By visualizing the routing distribution per layer, we can observe the efficacy of the load-balancing auxiliary loss and the onset of expert specialization. The heatmap below demonstrates how tokens are routed across 4 experts in a 6-layer architecture.

![Expert Routing Heatmap at Step 20,000](/plots-and-images/experts_step_20000.png)

A key observation is that early layers (L0–L3) exhibit relatively uniform routing (~20–30% per expert), while deeper layers (L4–L5) show pronounced specialization — Layer 5 Expert 2 routes only 3.2% of tokens, indicating the network has learned to concentrate specific computations in specific experts at higher layers. This aligns with findings in large-scale MoE literature (Switch Transformer, Mixtral) and demonstrates that specialization emerges even at 12M parameter scale.

Current and ongoing experiments also include:
- **Polysemanticity Analysis:** Passing validation sets through the network and mapping the highest-activating tokens for each expert to determine if experts are learning monosemantic linguistic features (e.g., punctuation, verbs, specific semantic clusters).

## Inference and Generation
The inference pipeline is designed for both programmatic evaluation and interactive testing. The generation script supports standard stochastic sampling techniques including Temperature, Top-K, and Nucleus (Top-p) sampling.

### KV Cache
The attention mechanism implements KV caching to avoid recomputing Key and Value states for past tokens during autoregressive generation. This reduces the per-step complexity from O(N²) to O(N).

Benchmark results on Apple M-series (MPS), averaged over 5 runs:

| Method     | Tokens/sec | Latency (200 tok) | Speedup |
|------------|------------|-------------------|---------|
| No Cache   | 17.8       | 11.2s             | —       |
| KV Cache   | 40.8       | 4.9s              | **2.28x** |

The speedup scales with sequence length — longer contexts benefit more from caching as the proportion of recomputed states grows.

Both modes are available via the generation script:
```bash
# Standard generation with expert routing stats
python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"

# Fast cached generation
python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time" --use_cache

# Benchmark both modes
python generate.py --checkpoint checkpoints/final.pt --benchmark
```

## Planned Research
The following evaluative experiments are actively being implemented:

1. **Out-of-Distribution (OOD) Evaluation:** Computing zero-shot perplexity on out-of-domain datasets (e.g., WikiText-2) to measure generalization vs. dataset memorization.
2. **Syntax Benchmarking (BLiMP):** Evaluating zero-shot grammatical competence using the Benchmark of Linguistic Minimal Pairs to measure structural understanding independent of cross-entropy loss.
3. **POS Affinity Analysis:** Mapping the POS-tag distribution of tokens routed to each expert per layer to characterize whether deep-layer specialization (observed in L4–L5) corresponds to syntactic categories.
4. **Beam Search:** Implementing beam search decoding to improve global coherence in generated narratives.

## Usage

### Pre-trained Weights
If you do not want to train the model from scratch, you can download the final trained 12M parameter weights (Step 20,000) here:
[Download `final.pt` (Google Drive)](https://drive.google.com/file/d/1sBxGOBGw7pLLTniH9Vni3Eb5OPzgZZWM/view?usp=sharing)

Place the downloaded file in the `checkpoints/` directory as `checkpoints/final.pt` to use it with the evaluation and generation scripts.

### Dependencies
```bash
pip install -r requirements.txt
```

### Training
The training script automatically detects the optimal device (CUDA, MPS, or CPU). On the first run, it will automatically download the TinyStories dataset and train the SentencePiece tokenizer.
```bash
python main.py
```

### Resume Training
```bash
python main.py --resume checkpoints/step_1000.pt
```

### Evaluation
Evaluate perplexity, token accuracy, and generate a routing heatmap for a specific checkpoint:
```bash
python main.py --eval_only --resume checkpoints/final.pt
```

### Generation
```bash
# Interactive mode (standard, with expert routing display)
python generate.py --checkpoint checkpoints/final.pt --interactive

# Interactive mode (fast, with KV cache)
python generate.py --checkpoint checkpoints/final.pt --interactive --use_cache

# Single prompt
python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"

# Benchmark KV cache speedup
python generate.py --checkpoint checkpoints/final.pt --benchmark
```