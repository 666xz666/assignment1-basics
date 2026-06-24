import os
import pathlib
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO, Any
import multiprocessing
import ahocorasick

from cs336_basics.bpe_tokenizer.constants import GPT2_PRETOKEN_PATTERN


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    使用AC自动机多模式匹配，对齐特殊token切分文件，避免特殊token跨块截断
    """
    # 参数校验
    for tok in split_special_tokens:
        assert isinstance(tok, bytes), "All special tokens must be bytestring"
    if not split_special_tokens:
        raise ValueError("split_special_tokens cannot be empty")

    def _build_token_automaton(tokens: list[bytes]) -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for idx, tok in enumerate(tokens):
            tok_str = tok.decode("utf-8")
            automaton.add_word(tok_str, (idx, tok))
        automaton.make_automaton()
        return automaton

    automaton = _build_token_automaton(split_special_tokens)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            min_offset = None
            chunk_str = mini_chunk.decode("utf-8", errors="ignore")
            for end_idx, (_, tok_bytes) in automaton.iter(chunk_str):
                start_idx = end_idx - len(tok_bytes) + 1
                if (min_offset is None) or (start_idx < min_offset):
                    min_offset = start_idx

            if min_offset is not None:
                chunk_boundaries[bi] = initial_position + min_offset
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(args) -> Counter[tuple[bytes, ...]]:
    """
    单进程处理文件分片，GPT2预分词 + 特殊token隔离，返回词频
    NOTE: 实际内部生成 tuple[int,...]，上层 _count_corpus 做类型适配
    args: (input_path: Path, start: int, end: int, special_tokens: list[str])
    """
    input_path, start, end, special_tokens = args
    word_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_data = f.read(end - start)

    chunk = raw_data.decode("utf-8", errors="ignore")

    if not special_tokens:
        for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
            word_str = match.group(0)
            token_tuple = tuple(word_str.encode("utf-8", errors="ignore"))
            word_counter[token_tuple] += 1
        return word_counter

    sorted_special = sorted(special_tokens, key=len, reverse=True)
    special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
    for part in re.split(special_pat, chunk):
        if part in special_tokens:
            word_counter[tuple(part.encode('utf-8', errors="ignore"))] += 1
        else:
            for match in re.finditer(GPT2_PRETOKEN_PATTERN, part):
                word_str = match.group(0)
                token_tuple = tuple(word_str.encode("utf-8", errors="ignore"))
                word_counter[token_tuple] += 1
    return word_counter


class BPETrainer:
    def __init__(
        self,
        input_path: pathlib.Path,
        vocab_size: int,
        special_tokens: list[str],
        num_processers: int | None = None
    ):
        # 基础配置
        self.input_path: pathlib.Path = input_path
        self.target_vocab_size: int = vocab_size
        self.special_tokens: list[str] = special_tokens
        # 自动CPU核心数，支持外部手动指定进程数
        self.num_processers: int = num_processers if num_processers is not None else multiprocessing.cpu_count()

        # 特殊token字节缓存
        self.special_token_bytes: list[bytes] = [s.encode("utf-8") for s in special_tokens]
        self.special_byte_set: set[bytes] = set(self.special_token_bytes)

        # 全局词表映射
        self.vocab: dict[int, bytes] = {}
        self.byte_to_id: dict[bytes, int] = {}
        self.id_to_byte: dict[int, bytes] = {}
        self.next_token_id: int = 0

        # 全局词频: 单词ID元组 -> 出现频次
        self.global_word_freq: Counter[tuple[int, ...]] = Counter()

        # Pair统计结构（增量BPE核心结构）
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.pair_to_words: defaultdict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

        # 合并规则列表（输出结果）
        self.merges: list[tuple[bytes, bytes]] = []

    def _init_vocab_mappings(self) -> None:
        """初始化基础词表：0~255单字节 + 自定义特殊Token"""
        self.next_token_id = 0
        # 0-255 原始字节
        for byte_val in range(256):
            b = bytes([byte_val])
            self.vocab[self.next_token_id] = b
            self.byte_to_id[b] = self.next_token_id
            self.id_to_byte[self.next_token_id] = b
            self.next_token_id += 1
        # 插入特殊token，避免重复
        for sp_byte in self.special_token_bytes:
            if sp_byte not in self.byte_to_id:
                self.vocab[self.next_token_id] = sp_byte
                self.byte_to_id[sp_byte] = self.next_token_id
                self.id_to_byte[self.next_token_id] = sp_byte
                self.next_token_id += 1

    def _count_corpus(self) -> None:
        """多进程分片统计全局预分词频次，适配process_chunk输出int元组"""
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processers, self.special_token_bytes)

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.special_tokens))

        with multiprocessing.Pool(processes=self.num_processers) as pool:
            chunk_counters = pool.map(process_chunk, tasks)

        # 汇总所有分片词频
        for cnt in chunk_counters:
            self.global_word_freq.update(cnt)

        # 转换：tuple[int(字节值)] → tuple[token_id]
        converted_freq = Counter()
        for int_byte_tuple, count in self.global_word_freq.items():
            word_id_tuple = tuple(
                self.byte_to_id[bytes([byte_val])] for byte_val in int_byte_tuple
            )
            converted_freq[word_id_tuple] += count
        self.global_word_freq = converted_freq

    def _init_pair_counts_full(self) -> None:
        """仅训练初始化时执行一次：遍历全部单词，一次性统计所有相邻Pair频次
        修复：完整判断多字节特殊Token，整体跳过内部Pair统计，防止被拆分
        """
        self.pair_counts.clear()
        self.pair_to_words.clear()

        for word_id_tuple, freq in self.global_word_freq.items():
            word_len = len(word_id_tuple)
            if word_len < 2:
                continue

            # 拼接完整单词字节，判断是否整体为特殊Token
            full_word_bytes = b"".join(self.id_to_byte[i] for i in word_id_tuple)
            if full_word_bytes in self.special_byte_set:
                # 特殊Token禁止内部Pair参与合并，直接跳过该单词
                continue

            # 遍历相邻Pair统计频次
            for i in range(word_len - 1):
                pair = (word_id_tuple[i], word_id_tuple[i+1])
                self.pair_counts[pair] += freq
                self.pair_to_words[pair].add(word_id_tuple)

    def _remove_word_pairs(self, word_tup: tuple[int, ...], freq: int):
        """删除一个单词全部相邻pair，扣减全局计数与反向映射"""
        if len(word_tup) < 2:
            return
        for i in range(len(word_tup) - 1):
            p = (word_tup[i], word_tup[i+1])
            self.pair_counts[p] -= freq
            self.pair_to_words[p].discard(word_tup)
            # 频次清零则删除key，避免max读取无效0频pair
            if self.pair_counts[p] <= 0:
                del self.pair_counts[p]

    def _add_word_pairs(self, word_tup: tuple[int, ...], freq: int):
        """新增一个单词全部相邻pair，增加全局计数与反向映射；自动拦截特殊token"""
        if len(word_tup) < 2:
            return

        for i in range(len(word_tup) - 1):
            p = (word_tup[i], word_tup[i+1])
            self.pair_counts[p] += freq
            self.pair_to_words[p].add(word_tup)

    def _merge_word_tuple(
        self,
        word_id_tuple: tuple[int, ...],
        target_pair: tuple[int, int],
        new_token_id: int
    ) -> tuple[int, ...]:
        """将单个单词内所有指定相邻Pair合并为新ID"""
        new_word = []
        i = 0
        a, b = target_pair
        while i < len(word_id_tuple):
            if i + 1 < len(word_id_tuple) and word_id_tuple[i] == a and word_id_tuple[i+1] == b:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word_id_tuple[i])
                i += 1
        return tuple(new_word)

    def _merge_one_step(self) -> bool:
        """【增量更新版】一轮BPE合并：仅修改受影响单词与Pair，不再全局重统计"""
        if not self.pair_counts:
            return False

        # 1. 择优逻辑完全不变：频次优先，同分字节字典序更大优先
        max_freq = max(self.pair_counts.values())
        candidate_pairs = [p for p, cnt in self.pair_counts.items() if cnt == max_freq]
        best_pair = max(
            candidate_pairs,
            key=lambda p: (self.id_to_byte[p[0]], self.id_to_byte[p[1]])
        )
        a, b = best_pair
        new_id = self.next_token_id

        # 2. 取出所有包含当前待合并pair的单词
        affected_words = list(self.pair_to_words.get(best_pair, set()))

        # 3. 逐个增量更新单词、pair计数
        for old_word in affected_words:
            cnt = self.global_word_freq.pop(old_word)
            # 移除旧单词所有pair计数
            self._remove_word_pairs(old_word, cnt)
            # 执行合并生成新单词
            new_word = self._merge_word_tuple(old_word, best_pair, new_id)
            # 更新全局词频
            self.global_word_freq[new_word] = self.global_word_freq.get(new_word, 0) + cnt
            # 添加新单词所有pair计数
            self._add_word_pairs(new_word, cnt)

        # 4. 注册新合并子词到词表（逻辑和朴素版完全一致）
        byte_a = self.id_to_byte[a]
        byte_b = self.id_to_byte[b]
        merged_byte = byte_a + byte_b
        self.merges.append((byte_a, byte_b))

        self.vocab[new_id] = merged_byte
        self.byte_to_id[merged_byte] = new_id
        self.id_to_byte[new_id] = merged_byte
        self.next_token_id += 1

        # 增量核心改动：删除全局重统计pair的调用
        # self._init_pair_counts_full()
        return True

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """主训练入口，迭代合并直到词表达到目标大小"""
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()

        # 初始化流程（仅此处全局统计一次pair）
        self._init_vocab_mappings()
        self._count_corpus()
        self._init_pair_counts_full()

        # 迭代增量合并
        while len(self.vocab) < self.target_vocab_size:
            success = self._merge_one_step()
            if not success:
                break

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(15)

        return self.vocab, self.merges