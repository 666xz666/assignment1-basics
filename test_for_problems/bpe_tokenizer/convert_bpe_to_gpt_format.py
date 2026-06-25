import os
import json

def byte_list_to_str(byte_nums: list[int]) -> str:
    """字节数值列表 → GPT2 可视字符串，空格替换为 Ġ"""
    raw_bytes = bytes(byte_nums)
    try:
        s = raw_bytes.decode("utf-8")
        return s.replace(" ", "Ġ")
    except UnicodeDecodeError:
        # 无法解码就保留原始字节字面量
        return repr(raw_bytes)

def convert_vocab_and_merges(
    vocab_in_path: str,
    merges_num_in_path: str,
    vocab_out_path: str,
    merges_out_path: str
):
    # ========== 1、读取原始词表 id -> 字节列表 ==========
    with open(vocab_in_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    id_to_bytes = {int(k): v for k, v in raw_vocab.items()}

    # 转换：id→字节列表  =>  token字符串→id（GPT2/HF标准 vocab）
    str_token_to_id = {}
    for tid, blist in id_to_bytes.items():
        token_str = byte_list_to_str(blist)
        str_token_to_id[token_str] = tid

    # 保存新 vocab_gpt2.json
    with open(vocab_out_path, "w", encoding="utf-8") as f:
        json.dump(str_token_to_id, f, ensure_ascii=False, indent=4)
    print(f"[Done] GPT2格式词表已写出：{vocab_out_path}")

    # ========== 2、转换 merges 数字格式 → GPT2 文本格式 ==========
    merge_pairs = []
    with open(merges_num_in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part1, part2 = line.split()
            nums1 = list(map(int, part1.split(",")))
            nums2 = list(map(int, part2.split(",")))
            merge_pairs.append((nums1, nums2))

    with open(merges_out_path, "w", encoding="utf-8") as f:
        for b_list1, b_list2 in merge_pairs:
            s1 = byte_list_to_str(b_list1)
            s2 = byte_list_to_str(b_list2)
            f.write(f"{s1} {s2}\n")
    print(f"[Done] GPT2格式merges已写出：{merges_out_path}")

if __name__ == "__main__":
    # 路径配置（和你训练输出目录对齐）
    work_dir = os.path.join("output", "run_bpe_train_on_owt_output")

    vocab_input = os.path.join(work_dir, "vocab.json")
    merges_num_input = os.path.join(work_dir, "merges.txt")

    vocab_gpt2_output = os.path.join(work_dir, "vocab_gpt2.json")
    merges_gpt2_output = os.path.join(work_dir, "merges_gpt2.txt")

    # 校验输入文件存在
    if not os.path.exists(vocab_input):
        raise FileNotFoundError(f"找不到输入词表：{vocab_input}")
    if not os.path.exists(merges_num_input):
        raise FileNotFoundError(f"找不到数字版merges：{merges_num_input}")

    convert_vocab_and_merges(
        vocab_in_path=vocab_input,
        merges_num_in_path=merges_num_input,
        vocab_out_path=vocab_gpt2_output,
        merges_out_path=merges_gpt2_output
    )
    print("\n全部转换完成！")