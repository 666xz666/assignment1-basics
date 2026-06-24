from __future__ import annotations
from typing import Iterator, Iterable

from cs336_basics.bpe_tokenizer.utils import pre_tokenize

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        GPT2 风格 BPE 分词器，对齐 tiktoken 行为
        :param vocab: id -> bytes 词表
        :param merges: 有序合并规则列表 [(a,b), ...]
        :param special_tokens: 特殊字符串列表
        """
        # 正向词表 id → bytes
        self.vocab: dict[int, bytes] = vocab
        # 反向词表 bytes → id
        self.bytes_to_id: dict[bytes, int] = {b: idx for idx, b in vocab.items()}

        # 合并优先级：pair -> 优先级序号（越小越先合并）
        self.merge_rank: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            self.merge_rank[pair] = rank

        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []

    def _bpe_merge(self, raw_bytes: bytes) -> list[bytes]:
        """对单个字节串执行完整BPE合并，返回子词字节列表"""
        if not raw_bytes:
            return []
        tokens: list[bytes] = [bytes([b]) for b in raw_bytes]

        while True:
            min_rank = None
            min_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merge_rank:
                    r = self.merge_rank[pair]
                    if (min_rank is None) or r < min_rank:
                        min_rank = r
                        min_pair = (i, pair)
            if min_pair is None:
                break

            idx, target_pair = min_pair
            new_token = target_pair[0] + target_pair[1]
            tokens = tokens[:idx] + [new_token] + tokens[idx+2:]
        return tokens

    def encode(self, text: str) -> list[int]:
        """完整字符串编码 → token id 列表，严格对齐 tiktoken gpt2"""
        segments = pre_tokenize(text, self.special_tokens)
        ids: list[int] = []
        for seg in segments:
            if seg in self.special_tokens:
                seg_b = seg.encode("utf-8")
                ids.append(self.bytes_to_id[seg_b])
            else:
                seg_bytes = seg.encode("utf-8")
                sub_tokens = self._bpe_merge(seg_bytes)
                for st in sub_tokens:
                    ids.append(self.bytes_to_id[st])
        return ids

    def decode(self, ids: list[int]) -> str:
        """id列表还原原始字符串，encode往返可逆"""
        total_bytes = b""
        for token_id in ids:
            total_bytes += self.vocab[token_id]
        return total_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式迭代编码，逐段生成id，极低内存占用"""
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            for tid in chunk_ids:
                yield tid