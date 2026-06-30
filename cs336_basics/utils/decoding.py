import torch

from ..nn import TransformerLM
from .function import softmax


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
) -> torch.Tensor:
    r"""
    自回归解码函数，实现带温度缩放的Top-p（核采样）生成，满足本次解码作业全部要求

    ## 数学公式说明
    ### 1. 温度缩放 Temperature Scaling
    设预测下一个token原始logits为 $\{v_i\}$：
    $$
    z_i = \frac{v_i}{T},\quad
    P(x_{t+1}=i\mid x_{1\dots t})
    = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
    $$
    式中 $T=\texttt{temperature}$。
    $T>1$ 分布更平缓，生成随机性更强；$0<T<1$ 分布更尖锐，偏向高概率词，生成更确定。

    ### 2. Top-p 核采样 Nucleus Sampling (Holtzman et al., 2020)
    将概率从大到小排序 $p_1\ge p_2\ge\dots\ge p_V$，找到最小下标 $k$ 满足累积概率阈值：
    $$
    \sum_{i=1}^k p_i \ge \texttt{top\_p}
    $$
    将下标 $k$ 之后所有token概率置为 $-\infty$，重新归一化概率分布，仅在候选集合内采样下一个token。

    ### 生成终止逻辑
    循环迭代采样 $x_{t+1}$ 拼接到序列末尾，满足任一条件停止：
    1. 采样出的token等于终止符 eos_token_id (<|endoftext|>)
    2. 已生成 max_new_tokens 个新词
    每次前向前自动截断输入长度不超过模型 context_length，防止RoPE位置索引越界报错

    Args:
        model: TransformerLM
            仅解码器结构语言模型，内部固定上下文窗口长度，使用RoPE旋转位置编码
        prompt_ids: torch.Tensor, shape [B, seq_len]
            初始提示词token编号，支持批量维度输入
        max_new_tokens: int
            最大允许生成的新词数量上限
        temperature: float
            温度缩放系数，必须大于0
        top_p: float
            核采样累积概率阈值，取值范围 $0<\text{top\_p}\le 1$
        eos_token_id: int
            文本结束符<|endoftext|>对应的token编号，触发提前终止生成

    Returns:
        torch.Tensor, shape [B, total_seq_len]
            原始提示词序列拼接生成内容后的完整token序列
    """
    # 输入合法性校验
    assert temperature > 0.0, "温度系数必须大于0"
    assert 0.0 < top_p <= 1.0, "top-p阈值需要在(0, 1]区间内"
    # 获取模型预设最大上下文窗口长度
    ctx_len = model.context_length
    # 拷贝初始prompt，避免修改外部传入张量
    tokens = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # 如果当前序列总长度超出上下文上限，截断最前面多余token
        if tokens.size(1) > ctx_len:
            tokens = tokens[:, -ctx_len:]

        # 模型前向，输出完整序列每个位置logits [batch, seq_len, vocab_size]
        logits = model(tokens)
        # 仅取出最后一个位置的logits，用于预测下一个token
        last_logits = logits[:, -1, :]

        # 第一步：温度缩放处理
        scaled_logits = last_logits / temperature

        # 第二步：Top-p核采样逻辑
        probs = softmax(scaled_logits, dim=-1)
        # 按概率从高到低排序，同时记录原词表下标
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # 计算排序后概率的累积和
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 标记累积概率超出top-p阈值的位置
        mask = cumulative_probs > top_p
        # 掩码右移一位：保证至少保留概率最大的一个token不会被屏蔽
        mask[:, 1:] = mask[:, 1:] & mask[:, :-1]

        # 把超出阈值的候选token logits填充为负无穷
        sorted_logits_masked = scaled_logits.scatter(
            dim=-1, index=sorted_indices, src=scaled_logits
        )
        sorted_logits_masked[mask] = -float("inf")

        # 重新归一化得到截断后的概率分布
        final_probs = softmax(sorted_logits_masked, dim=-1)
        # 多项式采样得到下一个token
        next_token = torch.multinomial(final_probs, num_samples=1)

        # 将新token拼接到序列末尾
        tokens = torch.cat([tokens, next_token], dim=1)

        # 碰到结束符，提前跳出生成循环
        if next_token.item() == eos_token_id:
            break

    return tokens
