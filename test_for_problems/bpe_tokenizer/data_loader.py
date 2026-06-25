"""
滑动窗口版数据集 Dataset
设计说明：
1. 整体策略：将每个分片内所有文档拼接为连续一维 token 流，使用固定长度滑动窗口采样，允许窗口跨文档（自然包含原生 <|endoftext|> 分隔符）
2. 不再强制样本末尾必须为 EOS，EOS 仅作为原始文档边界标记，由语料本身自带，不额外插入、不填充虚假 EOS
3. 窗口步长 = 上下文长度 ctx_len，相邻窗口无重叠，最大化数据利用率
4. offset 依然保留：仅编码阶段用来定位单篇文档边界做分片切分；本滑动窗口读取逻辑直接操作整条 token 流，不再依赖 offset 做采样
5. 自回归格式：x = 窗口[:-1], y = 窗口[1:]
6. 兼容两套数据集目录：TinyStories(单分片目录) / OWT(多分片目录)
7. 修复多进程 DataLoader mmap 句柄冲突问题，不在 __init__ 常驻内存映射对象
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# 与编码脚本路径常量对齐
ROOT_DIR = Path(".")
OUTPUT_DIR = ROOT_DIR / "output"
TS_TOKENIZER_DIR = OUTPUT_DIR / "run_bpe_train_on_tinystories_output"
OWT_TOKENIZER_DIR = OUTPUT_DIR / "run_bpe_train_on_owt_output"
TS_UNIFIED_SHARD_DIR = TS_TOKENIZER_DIR / "unified_shards"
OWT_UNIFIED_SHARD_DIR = OWT_TOKENIZER_DIR / "bin_shards"
SPECIAL_TOKENS = ["<|endoftext|>"]
EOF = "<|endoftext|>"

from cs336_basics.bpe_tokenizer.tokenizer import load_bpe_tokenizer

def get_eos_id(tokenizer_dir):
    tok = load_bpe_tokenizer(str(tokenizer_dir), SPECIAL_TOKENS)
    return tok.encode(EOF)[0]

class SlidingWindowShardDataset(Dataset):
    def __init__(
        self,
        shard_dir: Path,
        ctx_len: int = 2048
    ):
        self.ctx_len = ctx_len
        self.window_size = ctx_len
        self.samples_info = []

        shard_list = sorted(list(shard_dir.glob("shard_*.bin")))
        for bin_p in shard_list:
            # 仅记录分片路径，不常驻mmap
            bin_path = str(bin_p)
            # 打开映射获取总token数量
            mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
            total_tokens = len(mm)
            del mm

            # 计算该分片可生成多少个完整无重叠滑动窗口
            num_windows = (total_tokens - self.window_size) // self.window_size + 1
            for win_idx in range(num_windows):
                start = win_idx * self.window_size
                end = start + self.window_size
                self.samples_info.append((bin_path, start, end))

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        bin_path, start, end = self.samples_info[idx]
        mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
        window = mm[start:end].astype(np.int64)
        del mm

        # 自回归输入输出移位
        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:])
        return x, y

def get_dataloader(
    dataset_type: str,
    batch_size=8,
    ctx_len=2048,
    shuffle=True,
    num_workers=2,
    pin_memory=True
):
    """
    统一获取滑动窗口DataLoader
    :param dataset_type: "ts" / "owt"
    """
    if dataset_type == "ts":
        ds = SlidingWindowShardDataset(
            shard_dir=TS_UNIFIED_SHARD_DIR,
            ctx_len=ctx_len
        )
    elif dataset_type == "owt":
        ds = SlidingWindowShardDataset(
            shard_dir=OWT_UNIFIED_SHARD_DIR,
            ctx_len=ctx_len
        )
    else:
        raise ValueError('dataset_type must be "ts" or "owt"')

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader

if __name__ == "__main__":
    # 测试 TinyStories 滑动窗口加载
    loader_ts = get_dataloader("ts", batch_size=2, ctx_len=128)
    for x, y in loader_ts:
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("single window last token id:", y[0, -1].item())
        break

    # 测试 OWT 加载
    # loader_owt = get_dataloader("owt", batch_size=2, ctx_len=128)
    # for x, y in loader_owt:
    #     print("x shape:", x.shape)
    #     break