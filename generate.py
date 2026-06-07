"""
Text generation with SLM-MoE + expert usage tracking.

  python generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
  python generate.py --checkpoint checkpoints/final.pt --interactive
"""

import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm

from config import model_config
from utils import LanguageModel


# ━━━  Expert tracking hooks  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hook_expert_counts(model, n_experts):
    counts = {i: torch.zeros(n_experts) for i in range(len(model.blocks))}
    handles = []
    for lid, block in enumerate(model.blocks):
        def _h(l, c):
            def fn(_, __, out):
                _, idx = out
                for e in range(n_experts):
                    c[l][e] += (idx == e).sum().item()
            return fn
        handles.append(block.moe.router.register_forward_hook(_h(lid, counts)))
    return counts, handles


def unhook(handles):
    for h in handles:
        h.remove()


def print_experts(counts, n_experts):
    print("\n  ┌─ Expert Routing ─────────────────────────────────────────┐")
    for lid in sorted(counts):
        c = counts[lid]
        total = c.sum().item()
        if total == 0:
            continue
        bars = "  ".join(
            f"E{e} {'█' * int(c[e] / total * 15):<15s}{c[e] / total * 100:4.1f}%"
            for e in range(n_experts)
        )
        print(f"  │ L{lid}  {bars} │")
    overall = sum(counts.values(), torch.zeros(n_experts))
    t = overall.sum().item()
    print(f"  ├─────────────────────────────────────────────────────────┤")
    o = "  ".join(f"E{e} {overall[e] / t * 100:4.1f}%" for e in range(n_experts))
    print(f"  │ ALL  {o}  │")
    print(f"  └─────────────────────────────────────────────────────────┘\n")


# ━━━  Generation  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate(model, sp, prompt, max_tokens=200,
             temperature=0.8, top_k=50, top_p=0.9, device="cpu"):
    model.eval()
    n_experts = model_config["n_experts"]
    counts, hooks = hook_expert_counts(model, n_experts)

    ids = [sp.bos_id()] + sp.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_tokens):
        if input_ids.size(1) > model_config["max_seq_len"]:
            input_ids = input_ids[:, -model_config["max_seq_len"]:]

        logits = model(input_ids)[:, -1, :]

        # temperature
        if temperature > 0:
            logits = logits / temperature

        # top-k
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits[logits < kth] = float("-inf")

        # top-p  (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        tok = (torch.multinomial(probs, 1) if temperature > 0
               else probs.argmax(-1, keepdim=True))

        if tok.item() == sp.eos_id():
            break
        generated.append(tok.item())
        input_ids = torch.cat([input_ids, tok], dim=1)

    unhook(hooks)
    return sp.decode(generated), counts


# ━━━  Main  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint",  required=True)
    pa.add_argument("--sp_model",    default="tokenizer/tinystories_sp.model")
    pa.add_argument("--prompt",      default=None)
    pa.add_argument("--interactive", action="store_true")
    pa.add_argument("--max_tokens",  type=int,   default=200)
    pa.add_argument("--temperature", type=float,  default=0.8)
    pa.add_argument("--top_k",       type=int,    default=50)
    pa.add_argument("--top_p",       type=float,  default=0.9)
    pa.add_argument("--device",      default=None)
    args = pa.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", model_config)

    model = LanguageModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded ({n / 1e6:.1f}M params) on {device}")
    print(f"    temp={args.temperature}  top_k={args.top_k}  "
          f"top_p={args.top_p}\n")

    if args.interactive:
        print("Interactive mode  (type 'quit' to exit)\n")
        while True:
            prompt = input("You > ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            text, counts = generate(
                model, sp, prompt, args.max_tokens,
                args.temperature, args.top_k, args.top_p, device,
            )
            print(f"\nModel > {text}")
            print_experts(counts, cfg["n_experts"])

    elif args.prompt:
        text, counts = generate(
            model, sp, args.prompt, args.max_tokens,
            args.temperature, args.top_k, args.top_p, device,
        )
        print(f"Prompt │ {args.prompt}")
        print(f"Output │ {text}")
        print_experts(counts, cfg["n_experts"])

    else:
        print("Pass --prompt \"...\" or --interactive")


if __name__ == "__main__":
    main()