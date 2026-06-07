"""
SLM-MoE — Train & Eval (all-in-one).

  python main.py                                            # train
  python main.py --resume checkpoints/step_1000.pt          # resume
  python main.py --eval_only --resume checkpoints/final.pt  # eval only
"""

import os, time, math, argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from config import model_config, train_config
from utils import LanguageModel, load_balance_loss


# ━━━  Tokenizer  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SP_MODEL = "tokenizer/tinystories_sp.model"

def train_tokenizer():
    raw = "tinystories_raw.txt"
    print("📥  Loading TinyStories for tokenizer training...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    print(f"📝  Dumping text → {raw}")
    with open(raw, "w") as f:
        for ex in tqdm(ds, desc="  Writing stories", unit=" stories"):
            f.write(ex["text"].strip() + "\n")

    os.makedirs("tokenizer", exist_ok=True)
    print("🔧  Training SentencePiece BPE (vocab=16000)...")
    spm.SentencePieceTrainer.train(
        input=raw,
        model_prefix="tokenizer/tinystories_sp",
        vocab_size=model_config["vocab_size"],
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        shuffle_input_sentence=True,
        max_sentence_length=4096,
    )
    os.remove(raw)

    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    test = "Once upon a time there was a little cat."
    print(f"✅  Tokenizer ready  (vocab={sp.get_piece_size()})")
    print(f"    \"{test}\"  →  {sp.encode(test)}")


# ━━━  Dataset  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StoriesDataset(Dataset):
    def __init__(self, split, max_seq_len, sp):
        self.max_seq_len = max_seq_len
        BOS, EOS = sp.bos_id(), sp.eos_id()

        print(f"📥  Loading TinyStories/{split}...")
        ds = load_dataset("roneneldan/TinyStories", split=split)

        self.examples = []
        buf = []
        for ex in tqdm(ds, desc=f"  Tokenizing {split}", unit=" stories"):
            toks = [BOS] + sp.encode(ex["text"]) + [EOS]
            buf.extend(toks)
            while len(buf) >= max_seq_len + 1:
                self.examples.append(buf[: max_seq_len + 1])
                buf = buf[max_seq_len:]

        print(f"✅  {split}: {len(self.examples):,} examples\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        t = self.examples[idx]
        return (torch.tensor(t[:-1], dtype=torch.long),
                torch.tensor(t[1:],  dtype=torch.long))


# ━━━  Hooks (no model changes)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hook_router_logits(model):
    storage, handles = [], []
    for block in model.blocks:
        def _h(s):
            def fn(_, __, out): s.append(out)
            return fn
        handles.append(block.moe.router.gate.register_forward_hook(_h(storage)))
    return storage, handles


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


# ━━━  Display  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_experts(counts, n_experts):
    print()
    for lid in sorted(counts):
        c = counts[lid]
        total = c.sum().item()
        if total == 0:
            continue
        parts = " │ ".join(
            f"E{e} {'█' * int(c[e] / total * 20):<20s} {c[e] / total * 100:5.1f}%"
            for e in range(n_experts)
        )
        print(f"  L{lid}  {parts}")
    overall = sum(counts.values(), torch.zeros(n_experts))
    t = overall.sum().item()
    parts = " │ ".join(f"E{e} {overall[e] / t * 100:5.1f}%" for e in range(n_experts))
    print(f"  ALL  {parts}\n")


def save_expert_heatmap(counts, n_experts, step, out_dir="plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n_layers = len(counts)
    mat = np.zeros((n_layers, n_experts))
    for lid in range(n_layers):
        t = counts[lid].sum().item()
        if t > 0:
            for e in range(n_experts):
                mat[lid, e] = counts[lid][e].item() / t * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{e}" for e in range(n_experts)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    for i in range(n_layers):
        for j in range(n_experts):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if mat[i, j] > 40 else "black")
    plt.colorbar(im, label="Routing %")
    ax.set_title(f"Expert Usage — step {step}")
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/experts_step_{step}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊  Heatmap → {path}")


# ━━━  LR schedule  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cosine_lr(step, warmup, total, peak, floor=1e-6):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


# ━━━  Eval  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def run_eval(model, loader, device, n_experts, max_batches=200):
    model.eval()
    counts, hooks = hook_expert_counts(model, n_experts)
    tot_loss = tot_corr = tot_tok = 0

    for i, (x, y) in enumerate(tqdm(loader, total=min(max_batches, len(loader)),
                                     desc="  Evaluating", leave=False)):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        tot_loss += F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        ).item() * y.numel()
        tot_corr += (logits.argmax(-1) == y).sum().item()
        tot_tok += y.numel()

    unhook(hooks)
    model.train()

    loss = tot_loss / tot_tok
    return loss, math.exp(min(loss, 30)), tot_corr / tot_tok, counts


# ━━━  Main  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--device",    default=None)
    pa.add_argument("--max_steps", type=int, default=None)
    pa.add_argument("--resume",    default=None, help="checkpoint path")
    pa.add_argument("--eval_only", action="store_true")
    pa.add_argument("--lb_coeff",  type=float, default=0.01)
    args = pa.parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    max_steps = args.max_steps or train_config["max_steps"]
    n_experts = model_config["n_experts"]

    # tokenizer — train if missing
    if not os.path.exists(SP_MODEL):
        print("⚠️   Tokenizer not found — training one first...\n")
        train_tokenizer()
        print()
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)

    # data
    train_ds = StoriesDataset("train",      model_config["max_seq_len"], sp)
    val_ds   = StoriesDataset("validation", model_config["max_seq_len"], sp)
    train_loader = DataLoader(train_ds, train_config["batch_size"],
                              shuffle=True,  pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   train_config["batch_size"],
                              shuffle=False, pin_memory=True, num_workers=4)

    # model
    model = LanguageModel(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"📂  Resumed from {args.resume}  (step {start_step})")

    # ── eval-only mode ──
    if args.eval_only:
        loss, ppl, acc, counts = run_eval(
            model, val_loader, device, n_experts, max_batches=500
        )
        print(f"\n  loss {loss:.4f}  │  ppl {ppl:.2f}  │  acc {acc * 100:.2f}%")
        print_experts(counts, n_experts)
        save_expert_heatmap(counts, n_experts, start_step)
        return

    # ── training ──
    print("=" * 60)
    print("  SLM-MoE Training")
    print("=" * 60)
    eff = train_config["batch_size"] * train_config["grad_accum"]
    print(f"  Device       {device}")
    print(f"  Params       {n_params:,}  ({n_params / 1e6:.1f}M)")
    print(f"  Batch        {train_config['batch_size']} × "
          f"{train_config['grad_accum']} = {eff}")
    print(f"  Steps        {max_steps:,}")
    print(f"  LR           {train_config['lr']}  "
          f"(cosine, {train_config['warmup_steps']} warmup)")
    print(f"  LB coeff     {args.lb_coeff}")
    print("=" * 60, "\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config["lr"],
        betas=(0.9, 0.95), weight_decay=0.1,
    )
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    os.makedirs("checkpoints", exist_ok=True)
    model.train()
    it = iter(train_loader)
    step = start_step
    run_loss = run_lb = 0.0

    pbar = tqdm(total=max_steps, initial=start_step, desc="  Training",
                unit="step", dynamic_ncols=True)

    while step < max_steps:
        t0 = time.time()
        optimizer.zero_grad()
        acc_ce = acc_lb = 0.0

        for _ in range(train_config["grad_accum"]):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)

            rl_store, rl_hooks = hook_router_logits(model)

            logits = model(x)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            lb = sum(
                load_balance_loss(rl, n_experts) for rl in rl_store
            ) / len(rl_store)
            loss = (ce + args.lb_coeff * lb) / train_config["grad_accum"]
            loss.backward()

            acc_ce += ce.item() / train_config["grad_accum"]
            acc_lb += lb.item() / train_config["grad_accum"]

            unhook(rl_hooks)
            rl_store.clear()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, train_config["warmup_steps"],
                       max_steps, train_config["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        dt = time.time() - t0
        run_loss += acc_ce
        run_lb += acc_lb
        step += 1
        pbar.update(1)

        if step % 10 == 0:
            al = run_loss / 10
            alb = run_lb / 10
            pbar.set_postfix(
                loss=f"{al:.4f}",
                ppl=f"{math.exp(min(al, 20)):.1f}",
                lb=f"{alb:.5f}",
                lr=f"{lr:.2e}",
            )
            run_loss = run_lb = 0.0

        if step % train_config["eval_every"] == 0:
            pbar.write(f"\n{'─' * 60}")
            vloss, vppl, vacc, ecounts = run_eval(
                model, val_loader, device, n_experts
            )
            pbar.write(f"  📊 step {step}  val_loss {vloss:.4f}  "
                       f"val_ppl {vppl:.2f}  acc {vacc * 100:.2f}%")
            print_experts(ecounts, n_experts)
            save_expert_heatmap(ecounts, n_experts, step)

            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_config,
                "val_loss": vloss,
                "val_ppl": vppl,
            }, f"checkpoints/step_{step}.pt")
            pbar.write(f"  💾 checkpoints/step_{step}.pt")
            pbar.write(f"{'─' * 60}\n")
            model.train()

    pbar.close()

    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config,
    }, "checkpoints/final.pt")
    print("\n✅  Done — checkpoints/final.pt")


if __name__ == "__main__":
    main()