import regex as re
# GPT-2 官方预分词正则，全局只编译一次
GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)