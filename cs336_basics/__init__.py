import importlib.metadata
from .tokenizer import train_bpe

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "train_bpe"
]
