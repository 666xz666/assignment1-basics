import torch
import torch.nn as nn
import einops

class Softmax(nn.Module):
    """
    自定义 Softmax 网络层，对齐 PyTorch nn.Softmax 设计
    Softmax 计算公式 LaTeX：
    $$
    \text{softmax}(\boldsymbol{x})_j = \frac{\exp\left(x_j-\max(\boldsymbol{x})\right)}{\sum_{k}\exp\left(x_k-\max(\boldsymbol{x})\right)}
    $$
    初始化时固定归一化维度，前向传播仅传入特征张量；内置最大值平移防止数值溢出。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 初始化阶段固定归一化维度，与原生 nn.Softmax 保持一致
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 任意形状输入特征张量

        Returns:
            归一化后同形状张量

        NOTE: exp（x）在较大值时可以变为inf（则inf / inf = NaN）
        softmax 操作对对所有输入添加任意常数 c 保持不变。
        通常，我们会从v的所有元素中减去v中最大的元素，使得新的最大元素为0。
        """
        # shape: (..., dim, ...) 形状不变
        # keepdim=False 时: (..., 1, ...)
        # .values返回张量结果
        # .indices返回下标
        max_val = x.max(dim=self.dim, keepdim=True).values
        y = x - max_val
        exp = torch.exp(y)
        exp_sum = exp.sum(dim=self.dim, keepdim=True)
        return exp / exp_sum
        