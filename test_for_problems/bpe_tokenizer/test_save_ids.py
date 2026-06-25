import os
import shutil
import numpy as np
from pathlib import Path
import multiprocessing
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.tokenizer import load_bpe_tokenizer

# ===================== 全局路径配置 =====================
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

TS_TRAIN_TXT = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
OWT_TRAIN_TXT = DATA_DIR / "owt_train.txt"

TS_TOKENIZER_DIR = OUTPUT_DIR / "run_bpe_train_on_tinystories_output"
OWT_TOKENIZER_DIR = OUTPUT_DIR / "run_bpe_train_on_owt_output"

# TinyStories 合并单文件路径
TS_BIN_OUT_DIR = TS_TOKENIZER_DIR / "bin"
TS_TOKENS_BIN = TS_BIN_OUT_DIR / "ts_train_tokens.bin"
TS_OFFSETS_NPY = TS_BIN_OUT_DIR / "ts_train_offsets.npy"

# OWT 原生分片目录
OWT_SHARD_DIR = OWT_TOKENIZER_DIR / "bin_shards"

# 统一分片目录（TS包装为单个分片目录，与OWT结构对齐）
TS_UNIFIED_SHARD_DIR = TS_TOKENIZER_DIR / "unified_shards"
OWT_UNIFIED_SHARD_DIR = OWT_SHARD_DIR

SPECIAL_TOKENS = ["<|endoftext|>"]
EOF = "<|endoftext|>"
CHUNK_READ_SIZE = 131072
WORKER_NUM = max(2, multiprocessing.cpu_count() - 1)
SHARD_TOKEN_LIMIT = 50_000_000
# ==============================================================================

def scan_all_docs_with_est_length(file_path, tokenizer=None):
    """不再真实encode预估，用字符快速估算，避免两轮编码卡死"""
    docs_with_len = []
    buffer = []
    total_est_tokens = 0
    AVG_CHAR_PER_TOKEN = 3.2  # 英文文本经验值
    with open(file_path, "r", encoding="utf-8") as f:
        while chunk := f.read(CHUNK_READ_SIZE):
            buffer.append(chunk)
            full = "".join(buffer)
            if EOF in full:
                parts = full.split(EOF)
                for seg in parts[:-1]:
                    seg = seg.strip()
                    if seg:
                        est_len = int(len(seg) / AVG_CHAR_PER_TOKEN)
                        docs_with_len.append((seg, est_len))
                        total_est_tokens += est_len
                buffer = [parts[-1]]
    return docs_with_len, total_est_tokens

def build_shards_by_token_threshold(docs_with_len, token_limit):
    shards = []
    current_shard = []
    current_sum = 0
    for doc_text, doc_len in docs_with_len:
        if current_sum + doc_len > token_limit and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_sum = 0
        current_shard.append(doc_text)
        current_sum += doc_len
    if current_shard:
        shards.append(current_shard)
    return shards

def worker_encode_shard_task(q, out_dir, tokenizer_path, special_tokens):
    tokenizer = load_bpe_tokenizer(tokenizer_path, special_tokens)
    while True:
        try:
            shard_idx, doc_batch = q.get(timeout=1)
        except:
            break
        shard_bin = out_dir / f"shard_{shard_idx:06d}.bin"
        shard_off = out_dir / f"shard_{shard_idx:06d}_offset.npy"
        offsets = [0]
        all_ids = []
        for doc in doc_batch:
            ids = tokenizer.encode(doc)
            ids += tokenizer.encode(EOF)
            all_ids.extend(ids)
            offsets.append(len(all_ids))
        arr = np.array(all_ids, dtype=np.uint16)
        with open(shard_bin, "wb") as f:
            arr.tofile(f)
        np.save(shard_off, np.array(offsets, dtype=np.uint64))

def encode_to_multi_shards(txt_path, out_shard_dir, tokenizer_dir, special_tokens):
    Path(out_shard_dir).mkdir(exist_ok=True, parents=True)
    tokenizer = load_bpe_tokenizer(str(tokenizer_dir), special_tokens)

    print("\n[1/3] 扫描文档并预估长度...")
    docs_with_len, total_est_tok = scan_all_docs_with_est_length(str(txt_path), tokenizer)
    total_docs = len(docs_with_len)
    print(f"总文档: {total_docs}, 预估总Token: {total_est_tok:,}")

    print("\n[2/3] 按Token阈值划分子分片...")
    shard_batches = build_shards_by_token_threshold(docs_with_len, SHARD_TOKEN_LIMIT)
    total_shards = len(shard_batches)
    print(f"生成分片总数: {total_shards}")

    print("\n[3/3] 多进程队列并行编码写入分片...")
    task_q = multiprocessing.Queue()
    for idx, batch in enumerate(shard_batches):
        task_q.put((idx, batch))

    procs = []
    for _ in range(WORKER_NUM):
        p = multiprocessing.Process(
            target=worker_encode_shard_task,
            args=(task_q, out_shard_dir, str(tokenizer_dir), special_tokens)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print(f"\n编码完成！分片目录: {out_shard_dir}")
    return out_shard_dir

def merge_shards_to_single_bin(out_bin, out_offset_npy, shard_dir):
    shard_list = sorted(list(shard_dir.glob("shard_*.bin")))
    if not shard_list:
        return
    Path(out_bin).parent.mkdir(exist_ok=True, parents=True)
    f_out = open(out_bin, "wb")
    total_offsets = [0]
    base = 0
    for bin_path in shard_list:
        off_path = bin_path.with_name(bin_path.stem + "_offset.npy")
        shard_arr = np.fromfile(bin_path, dtype=np.uint16)
        shard_off = np.load(off_path)
        f_out.write(shard_arr.tobytes())
        shifted = (shard_off + base).tolist()[1:]
        total_offsets.extend(shifted)
        base += len(shard_arr)
    f_out.close()
    np.save(out_offset_npy, np.array(total_offsets, dtype=np.uint64))
    print(f"合并完成 -> {out_bin}, {out_offset_npy}")

def wrap_single_bin_to_shard_dir(bin_path: Path, offset_path: Path, out_shard_dir: Path):
    """将单文件bin+offset封装成标准单分片目录，统一格式"""
    out_shard_dir.mkdir(exist_ok=True, parents=True)
    target_bin = out_shard_dir / "shard_000000.bin"
    target_off = out_shard_dir / "shard_000000_offset.npy"
    shutil.copy2(bin_path, target_bin)
    shutil.copy2(offset_path, target_off)
    print(f"已封装单文件为统一分片目录: {out_shard_dir}")

def get_eos_id(tokenizer_dir):
    tok = load_bpe_tokenizer(str(tokenizer_dir), SPECIAL_TOKENS)
    return tok.encode(EOF)[0]

def main():
    # ========== TinyStories 流程：分片编码 → 合并单文件 → 封装为统一分片目录 ==========
    # shard_dir_ts = encode_to_multi_shards(
    #     txt_path=str(TS_TRAIN_TXT),
    #     out_shard_dir=TS_TOKENIZER_DIR / "tmp_shards",
    #     tokenizer_dir=str(TS_TOKENIZER_DIR),
    #     special_tokens=SPECIAL_TOKENS
    # )

    # 处理之前生成好的单文件
    # merge_shards_to_single_bin(TS_TOKENS_BIN, TS_OFFSETS_NPY, shard_dir_ts)
    # wrap_single_bin_to_shard_dir(TS_TOKENS_BIN, TS_OFFSETS_NPY, TS_UNIFIED_SHARD_DIR)

    # ========== OWT 流程：直接生成分片，无需合并 ==========
    encode_to_multi_shards(
        txt_path=str(OWT_TRAIN_TXT),
        out_shard_dir=OWT_SHARD_DIR,
        tokenizer_dir=str(OWT_TOKENIZER_DIR),
        special_tokens=SPECIAL_TOKENS
    )

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()