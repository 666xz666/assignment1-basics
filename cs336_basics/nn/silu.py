import torch
from torch import nn


class SiLU(nn.Module):
    def __init__(self):
        """
        Sigmoid Linear Unit (SiLU) activation function.
        Formula:
        $$
        \operatorname{SiLU}(x) = x \cdot \sigma(x),\quad \sigma(x)=\frac{1}{1+e^{-x}}
        $$
        Uses torch.sigmoid internally for numerical stability.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Element-wise SiLU activation.
        :param x: arbitrary-shape input tensor
        :return: same-shape activated tensor
        """
        """
        这里直接使用torch的实现保证数值稳定性
        PyTorch 内置算子优势
        自定义反向公式，减少运算步数
        PyTorch 为 sigmoid 手写了反向核，直接复用前向结果 \(\sigma(x)\) 计算梯度，不用重新算一遍 \(\exp\)，计算量更少、浮点误差更小。
        梯度裁剪保护，杜绝极端梯度爆炸 / NaN
        当 \(|x|\) 极大时，\(\sigma'(x)\approx0\)（梯度消失区域），原生算子做边界数值钳位，避免极小梯度被浮点压缩成 0 或意外 NaN；手写版本遇到极端输入极易出现梯度断崖。
        CUDA 向量化原子级优化
        GPU 上 torch.sigmoid 是单个融合算子（Kernel fusion），一个 GPU 线程束批量计算；
        手写 exp + add + div 是三次独立算子调用，中间张量要读写显存，不仅慢，中间显存读写会额外引入浮点舍入噪声，梯度累积偏差更大。
        """
        sigma = torch.sigmoid(x)
        return x * sigma
