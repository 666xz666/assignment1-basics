import regex as re


from cs336_basics.bpe_tokenizer.constants import GPT2_PRETOKEN_PATTERN

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []

        # 反向映射：bytes -> token id
        self.byte_to_id: dict[bytes, int] = {b: idx for idx, b in vocab.items()}
        # BPE 合并优先级字典：pair -> merge 序号（序号越小优先级越高）
        self.bpe_rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        # 特殊 token 字节集合，加速判断
        self.special_bytes_set: set[bytes] = {s.encode("utf-8") for s in self.special_tokens}

    def _pre_tokenize(self, text: str) -> list[str]:
        """GPT2 预分词 + 特殊 token 隔离拆分，和训练阶段分词逻辑严格对齐"""
        if not self.special_tokens:
            return [m.group() for m in GPT2_PRETOKEN_PATTERN.finditer(text)]

        # 长特殊token优先匹配分割
        sorted_spec = sorted(self.special_tokens, key=len, reverse=True)
        pat = "(" + "|".join(re.escape(t) for t in sorted_spec) + ")"
        parts = re.split(pat, text)
        res = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                res.append(part)
            else:
                res.extend([m.group() for m in GPT2_PRETOKEN_PATTERN.finditer(part)])
        return res

    def _bpe_merge_word(self, token_bytes: bytes) -> list[bytes]:
        """对单个字节单词执行迭代BPE合并，返回子词字节列表"""
        # 拆成初始单字节列表
        subwords = [bytes([b]) for b in token_bytes]
        while len(subwords) > 1:
            # 找出当前所有相邻pair里优先级最高（rank最小）的一对
            min_rank = float("inf")
            best_pair = None
            for i in range(len(subwords) - 1):
                pair = (subwords[i], subwords[i+1])
                r = self.bpe_rank.get(pair, float("inf"))
                if r < min_rank:
                    min_rank = r
                    best_pair = pair
            if best_pair is None:
                break  # 无可合并pair
            # 原地合并
            new_sub = []
            i = 0
            a, b = best_pair
            while i < len(subwords):
                if i < len(subwords)-1 and subwords[i] == a and subwords[i+1] == b:
                    new_sub.append(a + b)
                    i += 2
                else:
                    new_sub.append(subwords[i])
                    i += 1
            subwords = new_sub
        return subwords

    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表（对外主编码接口）"""
        pre_tokens = self._pre_tokenize(text)
        ids = []
        for tok_str in pre_tokens:
            tok_b = tok_str.encode("utf-8")
            # 特殊token直接查表
            if tok_b in self.byte_to_id:
                ids.append(self.byte_to_id[tok_b])
                continue
            # 普通单词BPE合并
            subs = self._bpe_merge_word(tok_b)
            for sub_b in subs:
                ids.append(self.byte_to_id[sub_b])
        return ids

    def decode(self, ids: list[int]) -> str:
        """token id 列表 -> 原始文本（对外主解码接口）"""
        total_bytes = b"".join([self.vocab[idx] for idx in ids])
        return total_bytes.decode("utf-8", errors="replace")