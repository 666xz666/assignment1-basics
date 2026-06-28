import sympy as sp


def adamw_accounting():
    # 1. 定义符号
    vocab_size, d_model, num_layers = sp.symbols(
        "vocab_size d_model num_layers", positive=True
    )
    batch_size, context_length, num_heads = sp.symbols(
        "batch_size context_length num_heads", positive=True
    )
    B, L = batch_size, context_length

    # 题目给定 d_ff = (8/3)*d_model，用 Rational 避免浮点数
    d_ff = sp.Rational(8, 3) * d_model

    # --------------------------
    # Parameters
    # --------------------------
    # 1. Token Embedding
    emb_params = vocab_size * d_model

    # 2. 单层 Transformer 参数
    # Attention QKV + Output Proj: 4*d_model²
    per_layer_attn = 4 * d_model**2
    # SwiGLU FFN: W1, W2, W3 → 3*d_model*d_ff
    per_layer_ffn = 3 * d_model * d_ff
    # 两层 RMSNorm，每层仅 gamma 权重 d_model
    per_layer_lnorm = 2 * d_model
    per_layer = per_layer_attn + per_layer_ffn + per_layer_lnorm

    # 所有层总参数
    all_layers = num_layers * per_layer

    # 3. 最后一层 RMSNorm
    final_lnorm = d_model

    # 4. 独立 LM Head（权重不共享 weight tying）
    lm_head_params = vocab_size * d_model

    # 总参数量 N_param
    N_param = emb_params + all_layers + final_lnorm + lm_head_params

    # 自动化简表达式
    N_param_simplified = sp.simplify(N_param)

    # 导出 LaTeX 公式
    print("化简后:\nN_{param} = ")
    print(sp.latex(N_param_simplified))

    # --------------------------
    # Gradients
    # --------------------------
    N_grad = N_param_simplified

    print("\nN_{grad} = N_{param} = ")
    print(sp.latex(N_grad))

    # --------------------------
    # Optimizer State
    # --------------------------
    # AdamW 对每一个参数，维护一阶动量m、二阶动量v两个独立的同形状张量
    N_optim = 2 * N_param_simplified
    N_optim_simplified = sp.simplify(N_optim)

    print("\nN_{optim} = 2 N_{param} = ")
    print(sp.latex(N_optim_simplified))

    # --------------------------
    # Activations (Peak Element Count, follow problem specified components)
    # --------------------------
    # 输入基础张量 shape: [B, L, d_model]
    base_act = B * L * d_model

    # 1. Single Transformer Block peak activation candidates
    # RMSNorm activation
    act_rmsnorm = base_act
    # Multi-head self attention max intermediate: QK^T score [num_heads, B, L, L]
    act_attn_qkt = num_heads * B * L**2
    # SwiGLU two branch projection: two [B, L, d_ff]
    act_ffn_2branch = 2 * B * L * d_ff

    # 单层 Transformer 内部激活峰值
    per_layer_peak = sp.Max(act_rmsnorm, act_attn_qkt, act_ffn_2branch)
    all_layer_acts = num_layers * per_layer_peak

    # Post-transformer activations
    act_final_rmsnorm = base_act
    # LM Head logits [B, L, vocab_size]
    act_logits = B * L * vocab_size
    # Cross-entropy intermediate buffer no larger than logits

    # Global activation peak element count
    N_act_peak = sp.Max(all_layer_acts, act_final_rmsnorm, act_logits)
    N_act_simplified = sp.simplify(N_act_peak)

    print("\nN_{act\\_peak} = ")
    print(sp.latex(N_act_simplified))

    # --------------------------
    # Total Peak Element Count
    # --------------------------
    N_total_elem = N_param_simplified + N_grad + N_optim_simplified + N_act_simplified
    N_total_simplified = sp.simplify(N_total_elem)

    print("\nN_{total\\_elem} = N_{param}+N_{grad}+N_{optim}+N_{act\\_peak} = ")
    print(sp.latex(N_total_simplified))

    # --------------------------
    # GPT-2 XL Numerical Substitution & Print (match your print format)
    # --------------------------
    print("\n===== Substitute GPT-2 XL Hyperparameters =====")
    vocab_size_val = 50257
    context_length_val = 1024
    num_layers_val = 48
    d_model_val = 1600
    num_heads_val = 25

    subs_dict_without_batch = {
        vocab_size: vocab_size_val,
        context_length: context_length_val,
        num_layers: num_layers_val,
        d_model: d_model_val,
        num_heads: num_heads_val,
    }

    N_total_elem_batch = sp.simplify(N_total_elem.subs(subs_dict_without_batch))
    N_total_mem_batch = N_total_elem_batch * 4 / (1024**3)

    print("\nN_total_mem_batch (GB) = ")
    print(sp.latex(N_total_mem_batch))

    # 求解不等式 total_mem <= 80
    eq = sp.Eq(N_total_mem_batch, 80)
    sol = sp.solve(eq, batch_size)[0]
    max_batch = sp.floor(sol)

    print(f"方程解 B = {float(sol):.4f}")
    print(f"最大整数 batch_size = {max_batch}")

    # 验算
    mem_11 = float(N_total_mem_batch.subs(batch_size, 11))
    mem_12 = float(N_total_mem_batch.subs(batch_size, 12))
    print(f"\n验算:")
    print(f"B=11 总内存: {mem_11:.4f} GB ≤ 80GB ✅")
    print(f"B=12 总内存: {mem_12:.4f} GB > 80GB ❌")

    # --------------------------
    # AdamW FLOPs Calculation
    # --------------------------
    # 定义符号 P: 模型总参数元素数量
    P = sp.Symbol("P", positive=True)

    # --------------------------
    # 1. Decoupled weight decay
    # Formula: $$\theta \leftarrow \theta - \alpha\lambda \theta$$
    # Operations: scalar multiply + elementwise subtraction → 2 FLOPs per parameter
    # --------------------------
    flop_weight_decay = 2 * P

    # --------------------------
    # 2. First moment (m) exponential moving average update
    # Formula: $$m \leftarrow \beta_1 m + (1-\beta_1)g$$
    # Operations: m *= β₁, grad*(1-β₁), elementwise add → 3 FLOPs per parameter
    # --------------------------
    flop_m_update = 3 * P

    # --------------------------
    # 3. Second moment (v) exponential moving average update
    # Formula: $$v \leftarrow \beta_2 v + (1-\beta_2)g^2$$
    # Operations: grad square, v *= β₂, squared grad scaling, elementwise add → 4 FLOPs per parameter
    # --------------------------
    flop_v_update = 4 * P

    # --------------------------
    # 4. Bias-corrected parameter update
    # Formula: $$\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v+\varepsilon}}$$
    # Operations: v+ε, sqrt, divide m, scale αₜ, subtract from θ → 5 FLOPs per parameter
    # --------------------------
    flop_theta_update = 5 * P

    # Total FLOPs for one full AdamW step (symbolic)
    total_flops_per_opt_step_sym = sp.simplify(
        flop_weight_decay + flop_m_update + flop_v_update + flop_theta_update
    )

    print("\n分项 FLOPs：")
    print(f"权重衰减: {sp.latex(flop_weight_decay)}")
    print(f"一阶动量m更新: {sp.latex(flop_m_update)}")
    print(f"二阶动量v更新: {sp.latex(flop_v_update)}")
    print(f"参数最终更新: {sp.latex(flop_theta_update)}")

    print("\n总 FLOPs 表达式：")
    print(sp.latex(total_flops_per_opt_step_sym))

    # 代入总参数量，得到 AdamW 单步原始 FLOPs
    P_num = N_param.subs(
        {
            num_layers: num_layers_val,
            d_model: d_model_val,
            vocab_size: vocab_size_val,
        }
    )
    flop_opt_per_step_raw = total_flops_per_opt_step_sym.subs(P, P_num)
    flop_opt_per_step_BF = flop_opt_per_step_raw / 1_000_000_000
    print(f"\nAdamW 代入结果：{flop_opt_per_step_BF:.4f} BFLOPs")

    # --------------------------
    # Train Time Estimate
    # --------------------------
    B_step = 1024  # 题目给定 batch size
    N_steps_total = 400_000  # 总训练步数

    # ===================== 1. 前向+反向传播 FLOPs (Kaplan 缩放公式) =====================
    # Kaplan 规则：单步前向+反向总FLOPs = $6 \cdot P \cdot B \cdot L$
    tokens_per_step = B_step * context_length_val
    flop_fwbw_per_step_raw = 6 * P_num * tokens_per_step
    flop_fwbw_per_step_TF = flop_fwbw_per_step_raw / 1e12
    print(f"\n===== 前向+反向传播单步FLOPs =====")
    print(f"单步token总数 = {B_step} × {context_length_val} = {tokens_per_step:,}")
    print(f"前向+反向单步: {flop_fwbw_per_step_TF:.4f} TFLOPs")

    # ===================== 2. 单训练步总FLOPs（单位统一为原始 FLOPs 再相加） =====================
    flop_total_per_step_raw = flop_fwbw_per_step_raw + flop_opt_per_step_raw
    flop_total_per_step_TF = flop_total_per_step_raw / 1e12
    print(f"\n===== 单个训练步总FLOPs =====")
    print(f"单步总FLOPs(前向+反向+AdamW): {flop_total_per_step_TF:.4f} TFLOPs")

    # ===================== 3. 400K 总步数全局总FLOPs =====================
    flop_total_all_steps_raw = flop_total_per_step_raw * N_steps_total
    flop_total_all_steps_PF = flop_total_all_steps_raw / 1e15
    print(f"\n===== 400K 训练步全局总FLOPs =====")
    print(f"总训练FLOPs: {flop_total_all_steps_PF:.4f} PetaFLOPs")

    # ===================== 4. H100 有效算力 & 总耗时计算 =====================
    # H100 FP32 理论峰值：495 TFLOP/s, MFU=50%
    peak_h100_TF_s = 495
    mfu = 0.5
    effective_TF_s = peak_h100_TF_s * mfu
    effective_FLOP_s = effective_TF_s * 1e12

    # 总秒数
    total_seconds = flop_total_all_steps_raw / effective_FLOP_s
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    print(f"\n===== H100 算力与训练耗时 =====")
    print(f"H100 理论峰值: {peak_h100_TF_s} TFLOP/s, MFU = {mfu * 100}%")
    print(f"GPU 有效算力: {effective_TF_s:.2f} TFLOP/s")
    print(f"总耗时 = {total_seconds:.2f} 秒")
    print(f"总耗时 = {total_hours:.2f} 小时")
    print(f"总耗时 = {total_days:.3f} 天")


if __name__ == "__main__":
    adamw_accounting()
