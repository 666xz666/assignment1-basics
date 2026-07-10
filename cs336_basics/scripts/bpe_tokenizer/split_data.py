import math
import os


def split_file(input_path: str, n_parts: int, output_dir: str = "split_out"):
    # 校验参数
    if n_parts <= 0:
        raise ValueError("份数必须是正整数")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    # 读取所有行（自动忽略末尾空行，如需保留删除strip判断）
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    total_lines = len(lines)
    print(f"总行数: {total_lines}, 要分成 {n_parts} 份")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 计算每份行数
    lines_per_part = math.ceil(total_lines / n_parts)

    # 切片拆分写入
    for idx in range(n_parts):
        start = idx * lines_per_part
        end = start + lines_per_part
        chunk = lines[start:end]
        out_file = os.path.join(output_dir, f"part_{idx + 1}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            for line in chunk:
                f.write(line + "\n")
        print(f"生成 {out_file}，行数: {len(chunk)}")


if __name__ == "__main__":
    # ========== 修改这里配置 ==========
    INPUT_FILE = "./data/owt_train.txt"  # 你的原始数据集路径
    SPLIT_NUM = 6  # 要分成几等份 n
    OUT_FOLDER = "./data/owt_train_split"  # 输出文件夹名
    # ==================================
    split_file(INPUT_FILE, SPLIT_NUM, OUT_FOLDER)
