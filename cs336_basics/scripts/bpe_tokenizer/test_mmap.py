import numpy as np
from cs336_basics import BPETokenizer, load_bpe_tokenizer
import json

# 1. 加载验证集 mmap
mmap_path = "./data/owt_train.mmap"
corpus = np.memmap(mmap_path, dtype=np.int64, mode="r")
print(f"总 token 数量: {len(corpus):,}")

# 2. 加载你生成该文件所用的**同一个BPE分词器**（关键：必须同一套词表/合并规则）
tok_cfg_path = "./data/config/owt_example/tokenizer_config.json"
with open(tok_cfg_path, "r", encoding="utf-8") as f:
    tok_cfg = json.load(f)
tokenizer = load_bpe_tokenizer(
    config_dir=tok_cfg["tokenizer_root_dir"],
    special_tokens=tok_cfg["special_tokens"]
)

# 随机起始位置
start = np.random.randint(0, len(corpus)-300)
slice_tokens = corpus[start:start+300]

# token id 转回字符串
decoded_text = tokenizer.decode(slice_tokens.tolist())
print("="*60)
print(f"抽样位置 [{start}:{start+300}] 解码结果：")
print(decoded_text)
print("="*60)