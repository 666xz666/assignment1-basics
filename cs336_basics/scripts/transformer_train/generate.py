import argparse
import torch
import json
import os
import warnings

# 屏蔽einops无关警告
warnings.filterwarnings(
    "ignore",
    message="It is discouraged to use axes names that are keywords: in",
    module="einops.parsing",
)

from cs336_basics.nn import TransformerLM
from cs336_basics.serialization import load_checkpoint
from cs336_basics.utils import try_gpu, decode
from cs336_basics.bpe_tokenizer import load_bpe_tokenizer, load_bpe_tokenizer_gpt2


def main():
    parser = argparse.ArgumentParser("GPT风格自回归文本生成脚本")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="模型权重路径，如 ./output/train_1/ckpt_it3000.pt",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="训练所用config.json路径"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./data/bpe_tokenizer",
        help="BPE分词器配置json路径",
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time,", help="续写起始提示词"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="生成续写token数量"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度 0~1.5")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p核采样阈值")
    args = parser.parse_args()

    device = try_gpu()

    # 1. 读取训练超参，复原模型结构
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab_size = cfg["vocab_size"]
    context_len = cfg["context_length"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    d_ff = cfg["d_ff"]
    num_layers = cfg["num_layers"]
    theta = cfg["theta"]
    dtype = getattr(torch, cfg["param_dtype"])

    # 2. 初始化模型 + 加载权重
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=theta,
        device=device,
        dtype=dtype,
    )
    # 加载断点（只加载模型参数，丢弃优化器、迭代步数）
    load_checkpoint(args.ckpt_path, model, optimizer=None)
    model.to(device)
    model.eval()
    print(f"✅ 模型权重加载完成: {args.ckpt_path}")

    # 3. 初始化分词器
    with open(args.tokenizer_dir, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    tokenizer = load_bpe_tokenizer_gpt2(
        tok_cfg["tokenizer_root_dir"], tok_cfg["special_tokens"]
    )
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    # 4. 编码prompt
    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    # print(prompt_ids)

    # 5. 调用你项目内置decode函数生成
    output_ids = decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    # print(output_ids)

    # 6. 解码输出文本
    gen_text = tokenizer.decode(output_ids[0].cpu().tolist())
    print(f"\n📝 输入Prompt: {args.prompt}")
    print("-" * 60)
    print(gen_text)


if __name__ == "__main__":
    main()

"""
python scripts/transformer_train/generate.py \
--ckpt_path ./output/train_1/ckpt_final.pt \
--config_path ./output/train_1/config.json \
--tokenizer_dir ./data/config/tiny_example/tokenizer_config.json \
--prompt "Once upon a time," \
--max_new_tokens 300 \
--temperature 0.7 \
--top_p 0.9
"""
