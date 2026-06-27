import torch
from cs336_basics import TransformerLM, try_gpu

# "GPT-2 XL-sized"
vocab_size = 50257
context_length = 1024  # S = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288
theta = 10000.0
S = context_length  # "Assume that our input sequence has context_length tokens."
d_k = d_model // num_heads


def rough_count():
    # 1. Token Embedding
    emb_params = vocab_size * d_model

    # 2. Single Transformer Layer
    # Attention QKV+O
    per_layer_attn = 4 * d_model**2
    # SwiGLU FFN: W1, W2, W3 三个矩阵
    per_layer_ffn = 3 * d_model * d_ff
    # Two RMSNorm (only g weight, each d_model)
    per_layer_lnorm = 2 * d_model
    per_layer = per_layer_attn + per_layer_ffn + per_layer_lnorm

    all_layers = num_layers * per_layer

    # 3. Final RMSNorm
    final_lnorm = d_model

    # 4. Independent LM Head (no weight tying)
    lm_head_params = vocab_size * d_model

    total_params = emb_params + all_layers + final_lnorm + lm_head_params
    mem_GB = total_params * 4 / (1024**3)
    params_B = total_params / 1_000_000_000

    print(f"手工计算总参数: {total_params:,} ≈ {params_B:.3f} B")
    print(f"FP32 内存: {mem_GB:.2f} GB")


def torch_count():
    dv = try_gpu()
    lm = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dv,
        torch.float32,
        theta,
    )
    # total_params = 0
    # # 逐层打印每个模块参数量
    # for name, p in lm.named_parameters():
    #     cnt = p.numel()
    #     total_params += cnt if p.requires_grad else 0
    #     print(f"{name:60} | {cnt:,}")

    total_trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    params_B = total_trainable / 1_000_000_000
    print(f"\n总可训练参数量: {total_trainable:,} ≈ {params_B:.3f} B")
    mem_bytes = total_trainable * 4
    mem_gb = mem_bytes / (1024**3)
    print(f"单精度加载所需内存: {mem_gb:.2f} GB")


def compute_forward_flops(S, vocab_size, d_model, num_heads, d_ff, num_layers):
    d_k = d_model // num_heads
    # ----------------------
    # 1. Per-layer Attention FLOPs
    # Q, K, V projection: 3 * (2 * S * d_model * d_model)
    qkv_proj_flop = 3 * 2 * S * d_model * d_model
    # Output projection W_O: 2 * S * d_model * d_model
    out_proj_flop = 2 * S * d_model * d_model

    # QK^T attention score: per head, 2*S*d_k*S, total num_heads
    qk_flop = num_heads * 2 * S * d_k * S
    # softmax @ V: per head, 2*S*S*d_k
    attn_v_flop = num_heads * 2 * S * S * d_k
    per_layer_attn_flop = qkv_proj_flop + out_proj_flop + qk_flop + attn_v_flop

    # ----------------------
    # 2. Per-layer SwiGLU FFN FLOPs
    w1_flop = 2 * S * d_model * d_ff
    w3_flop = 2 * S * d_model * d_ff
    w2_flop = 2 * S * d_ff * d_model
    per_layer_ffn_flop = w1_flop + w3_flop + w2_flop

    # Total per transformer block
    per_layer_total = per_layer_attn_flop + per_layer_ffn_flop
    all_layers_flop = num_layers * per_layer_total

    # ----------------------
    # 3. Final LM head projection
    lm_head_flop = 2 * S * d_model * vocab_size

    total_flops = all_layers_flop + lm_head_flop

    # 分项汇总字典，方便查看占比
    breakdown = {
        "single_layer_attention": per_layer_attn_flop,
        "single_layer_ffn": per_layer_ffn_flop,
        "all_layers_attention": num_layers * per_layer_attn_flop,
        "all_layers_ffn": num_layers * per_layer_ffn_flop,
        "lm_head": lm_head_flop,
        "total_FLOPs": total_flops,
    }

    return breakdown


def print_flop_ratio(info):
    total = info["total_FLOPs"]
    attn_ratio = info["all_layers_attention"] / total * 100
    ffn_ratio = info["all_layers_ffn"] / total * 100
    head_ratio = info["lm_head"] / total * 100
    print("===== (c) FLOPs 占比 =====")
    print(f"全部层注意力总占比: {attn_ratio:.2f}%")
    print(f"全部层FFN总占比:    {ffn_ratio:.2f}%")
    print(f"输出头LM Head占比:  {head_ratio:.2f}%")
    return attn_ratio, ffn_ratio, head_ratio


def analyze_gpt2_family_flops(S=1024, vocab_size=50257):
    configs = {
        "GPT2-Small": {"d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072},
        "GPT2-Medium": {
            "d_model": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "d_ff": 4096,
        },
        "GPT2-Large": {
            "d_model": 1280,
            "num_heads": 20,
            "num_layers": 36,
            "d_ff": 5120,
        },
    }
    res = {}
    print("\n===== (d) GPT2 系列 FLOPs 拆解 S=1024 =====")
    for name, cfg in configs.items():
        fb = compute_forward_flops(
            S,
            vocab_size,
            cfg["d_model"],
            cfg["num_heads"],
            cfg["d_ff"],
            cfg["num_layers"],
        )
        total = fb["total_FLOPs"]
        ra = fb["all_layers_attention"] / total * 100
        rf = fb["all_layers_ffn"] / total * 100
        rh = fb["lm_head"] / total * 100
        res[name] = {"attn%": ra, "ffn%": rf, "head%": rh, "total_flop": total}
        print(
            f"{name:12} | Attn:{ra:5.2f}% | FFN:{rf:5.2f}% | LMHead:{rh:5.2f}% | Total:{total:.3e}"
        )
    return res


def compare_context_length_flops(old_S=1024, new_S=16384):
    old = compute_forward_flops(old_S, vocab_size, d_model, num_heads, d_ff, num_layers)
    new = compute_forward_flops(new_S, vocab_size, d_model, num_heads, d_ff, num_layers)
    ratio_total = new["total_FLOPs"] / old["total_FLOPs"]
    print(f"\n===== (e) 上下文长度 1024 → 16384 对比 =====")
    print(f"原总FLOPs(S=1024): {old['total_FLOPs']:.3e}")
    print(f"新总FLOPs(S=16384): {new['total_FLOPs']:.3e}")
    print(f"总FLOPs放大倍数: {ratio_total:.2f}x")
    return old, new


def main():
    rough_count()
    print("=" * 40)
    torch_count()

    # 调用计算GPT-2 XL，S=1024
    flop_info = compute_forward_flops(
        S, vocab_size, d_model, num_heads, d_ff, num_layers
    )
    # 格式化打印结果
    print("===== (b) 前向传播 FLOPs 分项统计 (S=1024) =====")
    for name, val in flop_info.items():
        gflop = val / 1e9
        print(f"{name:25} | {val:.3e} FLOPs | {gflop:.2f} GFLOPs")
    total_gflop = flop_info["total_FLOPs"] / 1e9
    print(f"\n总前向FLOPs: {flop_info['total_FLOPs']:.3e} ≈ {total_gflop:.2f} GFLOPs")

    # 统计各层FLOPs占比
    print_flop_ratio(flop_info)

    # 分析GPT-2系列各个规模模型，分析FLOPs占比
    gpt2_family_data = analyze_gpt2_family_flops()

    # "increase the context length to 16,384"
    old, new = compare_context_length_flops(1024, 16384)
    print("\n旧比例：")
    print_flop_ratio(old)
    print("\n新比例：")
    print_flop_ratio(new)


if __name__ == "__main__":
    main()
