import json
import random
import time
from pathlib import Path
from typing import List, Tuple
from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer, load_bpe_tokenizer

EOF_TOKEN = "<|endoftext|>"
EOF_BYTES = EOF_TOKEN.encode("utf-8")

def split_docs_by_eof_stream(file_path: str) -> List[str]:
    """
    流式读取文件，以 <|endoftext|> 切分完整文档，不一次性载入全文
    返回：所有去除末尾<|endoftext|>的纯净文档列表
    """
    docs = []
    buffer = []
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                # 文件收尾
                remaining = "".join(buffer).strip()
                if remaining:
                    docs.append(remaining)
                break
            buffer.append(chunk)
            combined = "".join(buffer)
            if EOF_TOKEN in combined:
                parts = combined.split(EOF_TOKEN)
                # 前面完整分段
                for p in parts[:-1]:
                    doc = p.strip()
                    if doc:
                        docs.append(doc)
                # 剩余部分放回buffer等待后续补齐
                buffer = [parts[-1]]
    return docs


def sample_random_10_full_docs(file_path: str) -> List[str]:
    """获取全部完整文档，随机采样最多10篇"""
    all_docs = split_docs_by_eof_stream(file_path)
    total = len(all_docs)
    if total == 0:
        return []
    sample_cnt = min(10, total)
    selected = random.sample(all_docs, sample_cnt)
    return selected


def calc_single(text: str, tokenizer: BPETokenizer) -> Tuple[int, int, float]:
    raw_bytes = len(text.encode("utf-8"))
    token_ids = tokenizer.encode(text)
    token_cnt = len(token_ids)
    ratio = raw_bytes / token_cnt if token_cnt > 0 else 0.0
    return raw_bytes, token_cnt, ratio


def run_eval(
    corpus_name: str,
    valid_file: str,
    tokenizer_dir: str,
    out_log_path: str
) -> Tuple[float, float]:
    """
    返回 (平均压缩比, 吞吐量 bytes/s)
    """
    tokenizer = load_bpe_tokenizer(tokenizer_dir, ["<|endoftext|>"])
    docs = sample_random_10_full_docs(valid_file)
    if not docs:
        print(f"{corpus_name}: 文件未解析出有效文档")
        return 0.0, 0.0

    log_lines = [
        f"====== Evaluation: {corpus_name} ======\n",
        f"Source file: {valid_file}\n",
        f"Tokenizer model dir: {tokenizer_dir}\n",
        "ID | Original Full Document (split by <|endoftext|>) | Raw Bytes | Token Count | Compression Ratio (bytes/token)\n"
    ]
    total_b = 0
    total_t = 0
    encode_start = time.perf_counter()

    for idx, txt in enumerate(docs, start=1):
        b, t, r = calc_single(txt, tokenizer)
        total_b += b
        total_t += t
        line = f"{idx:2d} | {txt} | {b:4d} | {t:3d} | {r:.4f}"
        log_lines.append(line)
        print(line)

    encode_elapsed = time.perf_counter() - encode_start
    avg_ratio = total_b / total_t if total_t > 0 else 0.0
    throughput_bytes_per_sec = total_b / encode_elapsed if encode_elapsed > 0 else 0.0

    log_lines.append(f"\nTotal encoded bytes: {total_b}")
    log_lines.append(f"Total encode time: {encode_elapsed:.4f} s")
    log_lines.append(f"Throughput: {throughput_bytes_per_sec:.2f} bytes/sec")
    log_lines.append(f"Average compression ratio over 10 full documents: {avg_ratio:.4f}")

    Path(out_log_path).parent.mkdir(exist_ok=True)
    with open(out_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Result saved to: {out_log_path}")
    print(f"Throughput for this run: {throughput_bytes_per_sec:.2f} bytes/s\n")
    return avg_ratio, throughput_bytes_per_sec


if __name__ == "__main__":
    TINY_STORIES_VALID = "/home/projects/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    OWT_VALID = "/home/projects/CS336/assignment1-basics/data/owt_valid.txt"

    TOK_TS = "output/run_bpe_train_on_tinystories_output"
    TOK_OWT = "output/run_bpe_train_on_owt_output"
    LOG_DIR = "output/logs"

    print("=== 1. TinyStories trained BPE ===")
    ts_self_ratio, ts_self_throughput = run_eval(
        corpus_name="TinyStories Tokenizer on TinyStories Valid",
        valid_file=TINY_STORIES_VALID,
        tokenizer_dir=TOK_TS,
        out_log_path=f"{LOG_DIR}/ts_tokenizer_vs_ts_valid.txt"
    )
    ts_cross_ratio, ts_cross_throughput = run_eval(
        corpus_name="TinyStories Tokenizer on OWT Valid",
        valid_file=OWT_VALID,
        tokenizer_dir=TOK_TS,
        out_log_path=f"{LOG_DIR}/ts_tokenizer_vs_owt_valid.txt"
    )

    print("=== 2. OpenWebText trained BPE ===")
    owt_cross_ratio, owt_cross_throughput = run_eval(
        corpus_name="OWT Tokenizer on TinyStories Valid",
        valid_file=TINY_STORIES_VALID,
        tokenizer_dir=TOK_OWT,
        out_log_path=f"{LOG_DIR}/owt_tokenizer_vs_ts_valid.txt"
    )
    owt_self_ratio, owt_self_throughput = run_eval(
        corpus_name="OWT Tokenizer on OWT Valid",
        valid_file=OWT_VALID,
        tokenizer_dir=TOK_OWT,
        out_log_path=f"{LOG_DIR}/owt_tokenizer_vs_owt_valid.txt"
    )

    print("========== Summary ==========")
    print(f"TS tokenizer @ TS valid avg ratio: {ts_self_ratio:.4f}")
    print(f"TS tokenizer @ OWT valid avg ratio: {ts_cross_ratio:.4f}")
    print(f"OWT tokenizer @ TS valid avg ratio: {owt_cross_ratio:.4f}")
    print(f"OWT tokenizer @ OWT valid avg ratio: {owt_self_ratio:.4f}")

    print("\n========== Throughput Summary (bytes/second) ==========")
    print(f"TinyStories BPE on TS set: {ts_self_throughput:.2f} B/s")
    print(f"TinyStories BPE on OWT set: {ts_cross_throughput:.2f} B/s")
    print(f"OpenWebText BPE on TS set: {owt_cross_throughput:.2f} B/s")
    print(f"OpenWebText BPE on OWT set: {owt_self_throughput:.2f} B/s")

    # 两个分词器各自平均吞吐量
    avg_ts_throughput = (ts_self_throughput + ts_cross_throughput) / 2
    avg_owt_throughput = (owt_cross_throughput + owt_self_throughput) / 2
    print(f"\nAverage throughput TinyStories Tokenizer: {avg_ts_throughput:.2f} bytes/s")
    print(f"Average throughput OpenWebText Tokenizer: {avg_owt_throughput:.2f} bytes/s")