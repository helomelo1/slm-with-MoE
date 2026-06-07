model_config = {
    "vocab_size": 16000,
    "d_model": 256,
    "n_heads": 4,
    "head_dim": 256 // 4,
    "n_layers": 6,
    "d_ffn": 512,   # hidden_size per expert
    "n_experts": 4, # no of MoE experts
    "top_k": 2,
    "max_seq_len": 256,
    "dropout": 0.1
}

train_config = {
    "batch_size": 32,
    "grad_accum": 2, # doesn't dissipate the grad for 32 batches | effective batch_size = 64
    "lr": 3e-4,
    "warmup_steps": 500,
    "max_steps": 20000,
    "eval_every": 500
}