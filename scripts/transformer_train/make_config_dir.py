import json
import os

# 配置字典
run_name = "tiny_lr_1e-4_batch_64_step_7000"
config = {
    # 名称与路径
    "run_name": f"train_{run_name}",
    "run_id": None,
    "out_dir": f"./output/train_{run_name}",
    "train_data_path": "./data/TinyStoriesV2-GPT4-train.mmap",
    "valid_data_path": "./data/TinyStoriesV2-GPT4-valid.mmap",
    # 模型超参
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "num_layers": 4,
    "param_dtype": "float32",
    "theta": 10000.0,
    # 训练和步数
    "batch_size": 64,
    "max_iters": 7000,
    "warmup_steps": 700,
    "total_anneal_steps": 7000,
    # 学习率
    "alpha_max": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1e-08,
    "weight_decay": 0.01,
    "max_l2_norm": 1.0,
    "eps": 1e-08,
    # 检查点
    "resume": False,
    "ckpt_path": "",
}


def save_config(config_dict):
    # 获取输出目录
    out_dir = config_dict["out_dir"]
    # 不存在则创建文件夹
    os.makedirs(out_dir, exist_ok=True)
    # 拼接配置文件路径
    config_path = os.path.join(out_dir, "config.json")
    # 写入JSON，带缩进方便查看
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"配置已保存至: {config_path}")


if __name__ == "__main__":
    save_config(config)
