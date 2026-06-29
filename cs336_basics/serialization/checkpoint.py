import os
import typing
import torch
from torch import nn, optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    r"""
    保存包含模型权重、优化器状态和当前迭代步骤的训练检查点。
    将所有状态打包到一个字典容器中，然后通过torch序列化。保存到指定路径。

    Stored contents overview:
    $$
    \text{checkpoint} =
    \left\{
    \begin{aligned}
    &\texttt{model\_state\_dict}: \text{model.state\_dict()}, \\
    &\texttt{optimizer\_state\_dict}: \text{optimizer.state\_dict()}, \\
    &\texttt{iteration}: \text{current training iteration number}
    \end{aligned}
    \right.
    $$

    Args:
        model: PyTorch model instance whose parameter state will be saved
        optimizer: Optimizer instance (e.g. AdamW) to preserve momentum/learning rate scheduling state
        iteration: Integer training step counter to record progress
        out: Target output, can be file path string, path-like object, or binary file stream object

    Returns:
        None
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    r"""
    Load checkpoint from source, restore model parameters and optimizer internal state in-place.
    Extract and return the iteration number stored inside the checkpoint.

    Workflow:
    1. Deserialize checkpoint dictionary via torch.load
    2. model.load_state_dict(...) restore network weights
    3. optimizer.load_state_dict(...) restore optimizer running statistics
    4. return saved iteration value

    Args:
        src: Checkpoint source, can be file path string, path-like object, or binary file stream
        model: Empty / initialized model instance to be overwritten with saved weights
        optimizer: Initialized optimizer instance to receive saved optimizer state

    Returns:
        int: The training iteration value originally saved inside this checkpoint
    """
    checkpoint = torch.load(src, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
