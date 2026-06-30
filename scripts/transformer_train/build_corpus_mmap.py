import json
import os
import numpy as np
from typing import Tuple
from cs336_basics import (
    BPETokenizer,
    load_bpe_tokenizer,
    create_memmap_corpus,
)


def build_corpus_mmap(
    txt_path: str,
    tokenizer_cfg_json: str,
    dtype: np.dtype = np.int64,
) -> Tuple[np.memmap, BPETokenizer]:
    """
    仅执行：原始文本编码 → 生成完整全局语料memmap，不划分训练/验证集
    mmap 自动与 txt 同名、同目录，后缀替换为 .mmap
    :param txt_path: 原始文本文件路径
    :param tokenizer_cfg_json: 分词器独立配置json路径
    :param dtype: mmap存储整型
    :return: 完整语料memmap只读对象, 分词器实例
    """
    # 自动生成同路径同名 mmap 路径
    base, _ = os.path.splitext(txt_path)
    mmap_out_path = f"{base}.mmap"

    # 读取独立分词配置（与训练配置完全分离）
    with open(tokenizer_cfg_json, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    config_dir = tok_cfg["tokenizer_root_dir"]
    special_tokens = tok_cfg["special_tokens"]

    # 加载自定义BPE分词器
    tokenizer: BPETokenizer = load_bpe_tokenizer(config_dir, special_tokens)

    # 第一轮遍历统计总token数，不占用海量内存
    total_tokens = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        stream = tokenizer.encode_iterable(f)
        for _ in stream:
            total_tokens += 1

    # 分配memmap空间
    full_mmap = create_memmap_corpus(mmap_out_path, total_tokens, dtype=dtype)

    # 第二轮流式写入token id
    offset = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        stream = tokenizer.encode_iterable(f)
        for tid in stream:
            full_mmap[offset] = tid
            offset += 1
    full_mmap.flush()
    # 释放映射避免文件占用
    del full_mmap

    # 只读重新加载返回
    full_corpus = np.memmap(mmap_out_path, dtype=dtype, mode="r")
    return full_corpus, tokenizer


def main():
    # 在这里填写你的路径，也可以后续改成命令行传参
    txt_file = "./data/tinystories_example.txt"
    tok_cfg_path = "./data/config/tiny_example/tokenizer_config.json"
    build_corpus_mmap(txt_file, tok_cfg_path)


if __name__ == "__main__":
    main()
