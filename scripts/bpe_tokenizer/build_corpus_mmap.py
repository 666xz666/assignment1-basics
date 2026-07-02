import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import multiprocessing
from typing import Tuple, List
from cs336_basics import (
    BPETokenizer,
    load_bpe_tokenizer,
    create_memmap_corpus,
)


def encode_chunk_worker(
    chunk_lines: List[str], tokenizer_cfg_json: str, out_npy_path: str
):
    """子进程：编码一段行文本，保存为npy数组，分片内部顺序不变"""
    with open(tokenizer_cfg_json, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    tokenizer = load_bpe_tokenizer(
        tok_cfg["tokenizer_root_dir"], tok_cfg["special_tokens"]
    )

    all_ids = []
    for line in chunk_lines:
        ids = tokenizer.encode(line)
        all_ids.extend(ids)
    arr = np.array(all_ids, dtype=np.int64)
    np.save(out_npy_path, arr)


def build_corpus_mmap_multi_proc(
    txt_path: str,
    tokenizer_cfg_json: str,
    dtype: np.dtype = np.int64,
    num_proc: int = None,
) -> Tuple[np.memmap, BPETokenizer]:
    # 自动设置进程数：CPU物理核心一半，避免满载卡顿
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() // 2)
    base, _ = os.path.splitext(txt_path)
    mmap_out_path = f"{base}.mmap"
    tmp_dir = f"{base}_encode_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. 读取全部行、均匀连续分片（全局顺序固定，不乱序）
    print("=== 步骤1：读取文本并划分连续分片 ===")
    with open(txt_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    total_lines = len(all_lines)
    chunks = []
    lines_per_chunk = (total_lines + num_proc - 1) // num_proc
    for i in range(num_proc):
        start = i * lines_per_chunk
        end = min(start + lines_per_chunk, total_lines)
        chunk = all_lines[start:end]
        chunks.append((chunk, os.path.join(tmp_dir, f"chunk_{i:04d}.npy")))
    print(f"总行数: {total_lines}, 划分 {num_proc} 个并行分片")

    # 2. 多进程并行编码 + 分片完成进度条
    print(f"\n=== 步骤2：{num_proc} 进程并行编码文本 ===")
    tasks = []
    pool = multiprocessing.Pool(processes=num_proc)
    for chunk, npy_path in chunks:
        tasks.append(
            pool.apply_async(
                encode_chunk_worker, args=(chunk, tokenizer_cfg_json, npy_path)
            )
        )
    pool.close()
    for task in tqdm(tasks, desc="分片编码完成进度", total=len(tasks), unit="分片"):
        task.wait()

    # 3. 遍历所有分片统计全局总Token，带进度条
    print("\n=== 步骤3：统计全局总 Token 数量 ===")
    total_tokens = 0
    chunk_npy_paths = [p for (_, p) in chunks]
    for p in tqdm(chunk_npy_paths, desc="遍历分片统计Token", unit="分片"):
        arr = np.load(p, mmap_mode="r")
        total_tokens += len(arr)
    print(f"全局总 Token：{total_tokens:,}")

    # 4. 按原始分片顺序合并写入总mmap，总粒度进度条（最直观）
    print("\n=== 步骤4：有序合并写入最终 Memmap ===")
    full_mmap = create_memmap_corpus(mmap_out_path, total_tokens, dtype=dtype)
    offset = 0
    pbar_total = tqdm(total=total_tokens, unit="token", desc="合并写入进度")
    for p in chunk_npy_paths:
        arr = np.load(p)
        L = len(arr)
        full_mmap[offset : offset + L] = arr
        offset += L
        pbar_total.update(L)
    pbar_total.close()
    full_mmap.flush()
    del full_mmap

    # 清理临时分片文件
    shutil.rmtree(tmp_dir)

    # 只读重载返回，兼容原有接口
    full_corpus = np.memmap(mmap_out_path, dtype=dtype, mode="r")
    with open(tokenizer_cfg_json, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    tokenizer = load_bpe_tokenizer(
        tok_cfg["tokenizer_root_dir"], tok_cfg["special_tokens"]
    )
    return full_corpus, tokenizer


# 保留你原来单进程原版函数，可随时切换对比
def build_corpus_mmap(
    txt_path: str,
    tokenizer_cfg_json: str,
    dtype: np.dtype = np.int64,
) -> Tuple[np.memmap, BPETokenizer]:
    """
    原始单进程版本，结果与多进程版本完全等价
    """
    base, _ = os.path.splitext(txt_path)
    mmap_out_path = f"{base}.mmap"

    with open(tokenizer_cfg_json, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    config_dir = tok_cfg["tokenizer_root_dir"]
    special_tokens = tok_cfg["special_tokens"]
    tokenizer: BPETokenizer = load_bpe_tokenizer(config_dir, special_tokens)

    total_tokens = 0
    print("第一轮：统计全部 Token 总数")
    with open(txt_path, "r", encoding="utf-8") as f:
        stream = tokenizer.encode_iterable(f)
        for _ in tqdm(stream, desc="统计Token", unit="token"):
            total_tokens += 1
    print(f"统计完成，总 Token：{total_tokens:,}")

    full_mmap = create_memmap_corpus(mmap_out_path, total_tokens, dtype=dtype)
    offset = 0
    print("\n第二轮：编码并写入 Memmap")
    with open(txt_path, "r", encoding="utf-8") as f:
        stream = tokenizer.encode_iterable(f)
        for tid in tqdm(stream, total=total_tokens, desc="写入Memmap", unit="token"):
            full_mmap[offset] = tid
            offset += 1
    full_mmap.flush()
    del full_mmap

    full_corpus = np.memmap(mmap_out_path, dtype=dtype, mode="r")
    return full_corpus, tokenizer


def main():
    txt_file = "./data/owt_valid.txt"
    tok_cfg_path = "./data/config/owt_example/tokenizer_config.json"

    # Ubuntu 直接启用多进程，可手动指定进程数，如 num_proc=6
    build_corpus_mmap_multi_proc(txt_file, tok_cfg_path, num_proc=4)

    # 需要对比原单进程结果时打开下面注释
    # build_corpus_mmap(txt_file, tok_cfg_path)


if __name__ == "__main__":
    # Ubuntu fork 效率更高，注释掉spawn强制声明，可选保留兼容
    # multiprocessing.set_start_method("spawn", force=True)
    main()
