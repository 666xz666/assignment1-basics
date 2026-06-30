import json
import argparse
import numpy as np
import numpy.typing as npt
import torch
from types import SimpleNamespace
import wandb
import os
import time

from cs336_basics.nn import TransformerLM
from cs336_basics.optim import AdamW, CosineAnnealingWarmupLR
from cs336_basics.serialization import save_checkpoint, load_checkpoint
from cs336_basics.utils import (
    cross_entropy,
    get_batch,
    try_gpu,
    load_mmap_corpus,
    gradient_clipping,
    setup_logger,
)


def main(args):
    """
    Transformer 语言模型训练主函数
    :param args: 配置对象(SimpleNamespace), 由JSON配置文件解析而来，支持 . 属性访问
    """
    # 初始化日志
    logger = setup_logger(args.out_dir)
    # 全局起始时钟，用于统计 wall-clock time
    train_start_time = time.time()

    logger.info(f"========== Start New Experiment: {args.run_name} ==========")
    logger.info(
        f"Max training iterations: {args.max_iters}, Resume training: {args.resume}"
    )

    # ====================== 数据加载 ======================
    # 加载内存映射格式训练/验证语料（避免一次性载入全部数据占用显存）
    train_data: npt.NDArray = load_mmap_corpus(args.train_data_path, dtype=np.int64)
    val_data: npt.NDArray = load_mmap_corpus(args.valid_data_path, dtype=np.int64)
    device = try_gpu()
    logger.info(
        f"Train corpus shape: {train_data.shape}, Val corpus shape: {val_data.shape}"
    )
    logger.info(f"Using device: {device}")

    # ====================== 模型、优化器、学习率调度器初始化 ======================
    # 构建Transformer语言模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        device=device,
        dtype=getattr(torch, args.param_dtype),
        theta=args.theta,
    )
    logger.info(
        f"Model initialized, total params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # 自定义AdamW优化器（解耦权重衰减）
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.alpha_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # 带预热的余弦退火学习率调度器（按迭代步数更新lr）
    scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        T_w=args.warmup_steps,
        T_c=args.total_anneal_steps,
    )

    # ====================== 断点续训逻辑 ======================
    start_iter = 0
    if args.resume:
        # 加载权重、优化器状态，恢复迭代步数
        start_iter = load_checkpoint(args.ckpt_path, model, optimizer)
        # 同步调度器迭代计数，保证续训学习率曲线连续
        scheduler.last_epoch = start_iter
        logger.info(
            f"Resumed checkpoint from {args.ckpt_path}, start iteration: {start_iter}"
        )

    # ====================== Wandb可视化初始化 ======================
    wandb.init(
        project="assignment1-basic",
        name=args.run_name,
        config=args,  # 将全部配置录入wandb便于回溯实验
        dir="./output/wandb",
        mode="offline",
    )

    # ====================== 训练主循环 ======================
    # range左闭右开，it为当前迭代序号，总迭代次数：args.max_iters
    for it in range(start_iter, args.max_iters):
        model.train()
        # 采样一个批次训练数据
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # 前向传播计算损失
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 梯度清零 + 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，抑制梯度爆炸
        gradient_clipping(model.parameters(), args.max_l2_norm, args.eps)

        # 参数更新
        optimizer.step()

        it_log = it + 1
        # ---------------- 每隔固定步数验证+日志打印 ----------------
        if it_log % 100 == 0 or it_log == args.max_iters:
            model.eval()
            with torch.no_grad():
                # 验证集前向，不计算梯度节省显存
                vx, vy = get_batch(
                    val_data, args.batch_size, args.context_length, device
                )
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)

                current_lr = scheduler.get_last_lr()[0]
                elapsed_total_sec = time.time() - train_start_time

                # 替换原print，使用规范日志输出
                logger.info(
                    f"Iter {it_log:6d} | train_loss={loss.item():.4f} | val_loss={v_loss.item():.4f} "
                    f"| lr={current_lr:.6e} | elapsed={elapsed_total_sec:.2f}s"
                )

                # 写入wandb日志，满足作业要求：同时记录步数 & 墙钟时间
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": v_loss.item(),
                        "lr": current_lr,
                        "iter": it_log,  # iter记录为已完成总步数，和循环序号做区分
                        "wall_time_seconds": elapsed_total_sec,
                    }
                )

        # ---------------- 定期保存中间检查点 ----------------
        if (it_log) % 1000 == 0:
            ckpt_save_path = os.path.join(args.out_dir, f"ckpt_it{it_log}.pt")
            save_checkpoint(model, optimizer, it, ckpt_save_path)
            logger.info(f"Saved intermediate checkpoint: {ckpt_save_path}")

        # 更新学习率（每迭代一步更新一次，对应iter型调度）
        scheduler.step()

    # ====================== 训练结束，保存最终模型 ======================
    final_ckpt_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(
        model,
        optimizer,
        args.max_iters - 1,
        final_ckpt_path,
    )
    total_train_time = time.time() - train_start_time
    logger.info(f"Training complete! Total wall-clock time: {total_train_time:.2f}s")
    logger.info(f"Final checkpoint saved to: {final_ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    # 解析命令行参数，接收JSON配置文件路径
    parser = argparse.ArgumentParser(description="Transformer LM Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to json config file"
    )
    cli_args = parser.parse_args()

    # 读取JSON配置，自动转为SimpleNamespace，支持 . 属性访问，兼容嵌套字典
    with open(cli_args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # 确保输出文件夹存在
    os.makedirs(cfg_dict.out_dir, exist_ok=True)

    # 启动训练
    main(cfg_dict)
