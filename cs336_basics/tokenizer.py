import os
from typing import BinaryIO
import ahocorasick
import pathlib
from collections import Counter
import multiprocessing
import regex as re

from cs336_basics.config import NUM_PROCESSERS

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    使用AC自动机多模式匹配，多特殊token对齐文件分块边界
    """
    # 参数校验
    for tok in split_special_tokens:
        assert isinstance(tok, bytes), "All special tokens must be bytestring"
    if not split_special_tokens:
        raise ValueError("split_special_tokens cannot be empty")

    # 全局构建一次AC自动机
    def _build_token_automaton(tokens: list[bytes]) -> ahocorasick.Automaton:
        """构建AC自动机，存放所有分隔用特殊token"""
        automaton = ahocorasick.Automaton()
        for idx, tok in enumerate(tokens):
            # bytes -> str 再添加
            tok_str = tok.decode("utf-8")
            automaton.add_word(tok_str, (idx, tok))
        automaton.make_automaton()
        return automaton
    automaton = _build_token_automaton(split_special_tokens)

    # 获取文件总字节大小
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
            # AC自动机单次扫描，找出所有匹配
            chunk_str = mini_chunk.decode("utf-8", errors="ignore")
            for end_idx, (_, tok_bytes) in automaton.iter(chunk_str):
                start_idx = end_idx - len(tok_bytes) + 1
                if (min_offset is None) or (start_idx < min_offset):
                    min_offset = start_idx

            if min_offset is not None:
                chunk_boundaries[bi] = initial_position + min_offset
                break
            initial_position += mini_chunk_size

    # 去重并升序返回
    return sorted(set(chunk_boundaries))

def process_chunk(args) -> Counter[tuple[bytes, ...]]:
        """
        单个进程处理一个文件块，返回该块内预分词词频
        args: (input_path: Path, start: int, end: int, special_tokens: list[str])
        return: Counter{单词字节元组: 频次}
        """
        input_path, start, end, special_tokens = args
        word_counter = Counter()

        with open(input_path, "rb") as f:
            f.seek(start)
            raw_data = f.read(end - start)

        # 简单解码容错
        chunk = raw_data.decode("utf-8", errors="ignore")

        # 采用GPT-2的规则预分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 没有特殊token
        if not special_tokens:
            for match in re.finditer(PAT, chunk):
                word_str = match.group(0)
                token_tuple = tuple(word_str.encode("utf-8", errors="ignore"))
                word_counter[token_tuple] += 1

            return word_counter
        
        # 构建基于特殊token的正则规则把文本切片
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
        for part in re.split(special_pat, chunk):
            if part in special_tokens:
                # 特殊token统计
                word_counter[tuple(part.encode('utf-8', errors="ignore"))] += 1
            else:
                # 不含特殊token的片段，按照正常方法处理
                for match in re.finditer(PAT, part):
                    word_str = match.group(0)
                    token_tuple = tuple(word_str.encode("utf-8", errors="ignore"))
                    word_counter[token_tuple] += 1
        
        return word_counter

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable() # 开始分析

    # 1. pretokenization
    # 1.1 打开文件
    # 统一转为 Path 对象方便操作
    input_path = pathlib.Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {input_path}")

    # 1.2 分块
    special_token_bytes = [tok.encode("utf-8") for tok in special_tokens]
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, NUM_PROCESSERS, special_token_bytes)

    # 1.3 组装每个任务参数
    tasks = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))

    # 1.4 多进程并行统计词频
    global_word_freq = Counter()
    with multiprocessing.Pool(processes=NUM_PROCESSERS) as pool:
        # 每个任务返回一个块的Counter
        chunk_counters = pool.map(process_chunk, tasks)

    # 1.5 合并所有块频次得到全局词频
    for cnt in chunk_counters:
        global_word_freq.update(cnt)
    # ========== 至此：得到 BPE 训练需要的全局 word_freq ==========
    
    # 2. 合并操作
    # 2.1 初始化基础词表（256字节 + special tokens）
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 2.1 填充初始 0~255 字节到 vocab
    token_id = 0
    byte_to_id = dict()  # bytes -> id
    id_to_byte = dict()  # id -> bytes
    for byte_val in range(256):
        b = bytes([byte_val])
        vocab[token_id] = b
        byte_to_id[b] = token_id
        id_to_byte[token_id] = b
        token_id += 1

    # 2.2 插入 special tokens 到 vocab（保证不可拆分）
    for sp_byte in special_token_bytes:
        if sp_byte not in byte_to_id:
            vocab[token_id] = sp_byte
            byte_to_id[sp_byte] = token_id
            id_to_byte[token_id] = sp_byte
            token_id += 1
    special_byte_set = set(special_token_bytes)

    # 2.3 while len(vocab) < vocab_size:
    #       统计所有相邻pair频次
    #       取最高频pair合并，加入merges
    #       更新 global_word_freq
    #       新增子词进vocab
    def _get_pair_counts(word_freq: Counter[tuple[int, ...]], special_byte_set: set[bytes], id_to_byte: dict[int, bytes]) -> Counter[tuple[int, int]]:
        pair_counts = Counter()
        for word_tuple, freq in word_freq.items():
            # 整数列表还原完整字节
            full_word = b"".join(id_to_byte[i] for i in word_tuple)
            if full_word in special_byte_set:
                continue
            if len(word_tuple) < 2:
                continue
            for i in range(len(word_tuple)-1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_word_tuple(word_tuple: tuple[int,...], target_pair: tuple[int,int], new_id: int) -> tuple[int,...]:
        new_word = []
        i = 0
        a, b = target_pair
        while i < len(word_tuple):
            if i < len(word_tuple)-1 and word_tuple[i]==a and word_tuple[i+1]==b:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word)
    
    # 主循环
    while len(vocab) < vocab_size:
        pair_counts = _get_pair_counts(global_word_freq, special_byte_set, id_to_byte)
        if not pair_counts:
            break

        max_freq = max(pair_counts.values())
        # 筛选所有频次最高的候选pair
        candidates = [p for p, cnt in pair_counts.items() if cnt == max_freq]
        # 按两个token的字节内容字典序取最大
        best_pair = max(candidates, key=lambda p: (id_to_byte[p[0]], id_to_byte[p[1]]))

        # ========== 关键：整数pair转回(bytes, bytes)存入merges，匹配返回格式 ==========
        p1_byte = id_to_byte[best_pair[0]]
        p2_byte = id_to_byte[best_pair[1]]
        merges.append((p1_byte, p2_byte))

        new_id = token_id
        new_word_freq = Counter()
        for word_tup, cnt in global_word_freq.items():
            merged_tup = _merge_word_tuple(word_tup, best_pair, new_id)
            new_word_freq[merged_tup] += cnt
        global_word_freq = new_word_freq

        # 新词注册进三张表
        merged_byte = p1_byte + p2_byte
        vocab[new_id] = merged_byte
        byte_to_id[merged_byte] = new_id
        id_to_byte[new_id] = merged_byte
        token_id += 1

    profiler.disable() # 结束分析
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(15) # 打印耗时最长的 15 个函数

    return vocab, merges

       
