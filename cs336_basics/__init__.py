import importlib.metadata
from .bpe_tokenizer.tokenizer import BPETokenizer, load_bpe_tokenizer
from .bpe_tokenizer.trainer import BPETrainer

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "BPETokenizer",
    "BPETrainer",
    "load_bpe_tokenizer"
]
