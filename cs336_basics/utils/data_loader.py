import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    对单批输入目标子序列进行采样，用于自回归下一标记语言建模。
    给定一个长的扁平token-ID数组，随机采样起始位置，构造输入序列
    及其对应的右移目标序列。

    Rule:
    For a starting index $i$:
    Input sequence: $\boldsymbol{x}_{\text{in}} = x[i:i+L]$
    Target sequence: $\boldsymbol{x}_{\text{target}} = x[i+1:i+1+L]$
    where $L=\text{context\_length}$.

    Args:
        dataset: 1D numpy integer array storing global token IDs of entire corpus
        batch_size: number of independent sequences sampled in one batch
        context_length: sequence length $L$ for each input/target sample
        device: target device string for output tensors, e.g. "cpu", "cuda:0"

    Returns:
        tuple:
            - input_tokens: torch.Tensor shape (batch_size, context_length), input token sequences
            - target_tokens: torch.Tensor shape (batch_size, context_length), next-token prediction targets
            Both tensors are moved to specified device, dtype torch.long.
    """
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(low=0, high=max_start_idx, size=batch_size)

    input_list = []
    target_list = []
    for i in start_indices:
        seq_in = dataset[i : i + context_length]
        seq_tgt = dataset[i + 1 : i + 1 + context_length]
        input_list.append(seq_in)
        target_list.append(seq_tgt)

    input_tokens = torch.tensor(np.stack(input_list), dtype=torch.long, device=device)
    target_tokens = torch.tensor(np.stack(target_list), dtype=torch.long, device=device)

    return input_tokens, target_tokens
