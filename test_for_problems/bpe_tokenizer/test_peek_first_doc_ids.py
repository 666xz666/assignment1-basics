import numpy as np
from pathlib import Path
from cs336_basics.bpe_tokenizer.tokenizer import load_bpe_tokenizer

# 路径复用你现有配置
ROOT_DIR = Path(".")
OUTPUT_DIR = ROOT_DIR / "output"
TS_TOKENIZER_DIR = OUTPUT_DIR / "run_bpe_train_on_tinystories_output"
TS_BIN_OUT_DIR = TS_TOKENIZER_DIR / "bin"
TS_TOKENS_BIN = TS_BIN_OUT_DIR / "ts_train_tokens.bin"
TS_OFFSETS_NPY = TS_BIN_OUT_DIR / "ts_train_offsets.npy"
SPECIAL_TOKENS = ["<|endoftext|>"]

def peek_first_document():
    # 1. 加载偏移数组
    offsets = np.load(TS_OFFSETS_NPY)
    start = offsets[0]
    end = offsets[1]

    # 2. mmap 映射二进制，取出第一条 token id
    mm = np.memmap(TS_TOKENS_BIN, dtype=np.uint16, mode="r")
    first_ids = mm[start:end].tolist()

    # 3. 加载对应分词器
    tokenizer = load_bpe_tokenizer(str(TS_TOKENIZER_DIR), SPECIAL_TOKENS)
    text = tokenizer.decode(first_ids)

    print("===== 第一条文档 Token IDs =====")
    print(first_ids[:30], "...(省略后半部分)")
    print(f"总Token数量: {len(first_ids)}")
    print("\n===== 解码还原原文 =====")
    print(text)

if __name__ == "__main__":
    peek_first_document()