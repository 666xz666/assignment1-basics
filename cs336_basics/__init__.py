import importlib.metadata

# 从子包统一导入，不再写深层文件路径
from .utils import try_gpu
from .bpe_tokenizer import BPETokenizer, BPETrainer, load_bpe_tokenizer
from .nn import (
    Linear,
    Embedding,
    RMSNorm,
    SiLU,
    SwiGLUFeedForward,
    RoPE,
    Softmax,
    ScaledDotProductAttention,
    MultiheadSelfAttention,
    TransformerLM,
    TransformerBlock,
)
from .optim import SGD
from .utils import cross_entropy

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "try_gpu",
    "BPETokenizer",
    "BPETrainer",
    "load_bpe_tokenizer",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SiLU",
    "SwiGLUFeedForward",
    "RoPE",
    "Softmax",
    "ScaledDotProductAttention",
    "MultiheadSelfAttention",
    "TransformerLM",
    "TransformerBlock",
    "cross_entropy",
    "SGD",
]
