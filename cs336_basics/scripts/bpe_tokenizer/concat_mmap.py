import numpy as np
import os


def concat_mmap_files(mmap_dir: str, output_path: str, dtype=np.int32):
    # 1. 收集并排序 mmap 文件，保证顺序 part_1, part_2 ...
    mmap_files = []
    for fname in sorted(os.listdir(mmap_dir)):
        if fname.endswith(".mmap"):
            mmap_files.append(os.path.join(mmap_dir, fname))

    if not mmap_files:
        raise FileNotFoundError(f"目录 {mmap_dir} 下无 .mmap 文件")

    # 2. 先遍历统计总长度
    total_len = 0
    for fpath in mmap_files:
        arr = np.memmap(fpath, dtype=dtype, mode="r")
        total_len += arr.shape[0]
        del arr

    print(f"总 token 数量: {total_len}")

    # 3. 创建输出大 mmap
    out_mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=total_len)

    # 4. 逐段按顺序拷贝拼接
    offset = 0
    for fpath in mmap_files:
        arr = np.memmap(fpath, dtype=dtype, mode="r")
        n = arr.shape[0]
        out_mmap[offset : offset + n] = arr
        offset += n
        print(
            f"写入 {os.path.basename(fpath)} → 偏移 {offset - n} ~ {offset - 1}, 长度 {n}"
        )
        del arr

    out_mmap.flush()
    print(f"\n合并完成，输出: {output_path}")


if __name__ == "__main__":
    # ========== 改这里配置 ==========
    MMAP_FOLDER = "./data/owt_train_split"  # 存放 part_1.mmap part_2.mmap ... 的目录
    OUT_MMAP = "./data/owt_train.mmap"
    DTYPE = np.int64
    # =================================
    concat_mmap_files(MMAP_FOLDER, OUT_MMAP, dtype=DTYPE)
