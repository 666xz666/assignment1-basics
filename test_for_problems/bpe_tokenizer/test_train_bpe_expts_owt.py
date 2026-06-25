import os
import json
from cs336_basics import BPETrainer

def main():
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable() # 开始分析

    # 内存追踪启动
    import tracemalloc
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    trainer = BPETrainer(
        "/home/projects/CS336/assignment1-basics/data/owt_train.txt", 
        32_000, 
        ["<|endoftext|>"],
        5000
    )
    vocab, merges = trainer.train()

    profiler.disable() # 结束分析
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(5) # 打印最耗时的5个函数

    # 内存对比计算
    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_current, total_peak = tracemalloc.get_traced_memory()
    print(f"\n=== 内存占用统计 ===")
    print(f"当前内存占用: {total_current / 1024 / 1024:.2f} MB")
    print(f"训练峰值内存: {total_peak / 1024 / 1024:.2f} MB")

    # 前5行内存占用最大代码位置
    print("\n=== 内存占用最大代码位置 Top5 ===")
    for stat in top_stats[:5]:
        print(stat)
    tracemalloc.stop()

    # 遍历所有token，按 bytes 长度排序，取最长
    longest_token_id = max(vocab.items(), key=lambda item: len(item[1]))
    token_id, token_bytes = longest_token_id

    print(f"最长token ID: {token_id}")
    print(f"字节内容: {token_bytes!r}")
    print(f"字节长度: {len(token_bytes)}")
    # 尝试转字符串
    try:
        s = token_bytes.decode("utf-8")
        print(f"对应文本: {s}")
    except UnicodeDecodeError:
        print("无法完整解码为UTF-8字符")

    # 保存产生结果
    output_root = "output"
    output_dir = os.path.join(output_root, "run_bpe_train_on_owt_output")
    os.makedirs(output_dir, exist_ok=True)
    readable_vocab = {int(k): list(v) for k, v in vocab.items()}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, indent=4)

    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 将 bytes 转换为逗号分隔的数字字符串
            s1 = ",".join(map(str, list(p1)))
            s2 = ",".join(map(str, list(p2)))
            f.write(f"{s1} {s2}\n")

if __name__ == "__main__":
    main()