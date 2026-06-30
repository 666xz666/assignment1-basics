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


def save_config_json(cfg_path: str, cfg_dict: dict):
    """覆写更新 config.json，用于写入自动生成的 run_id"""
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)


def main(args, raw_cfg_dict: dict, cfg_path: str):
    """
    Transformer 语言模型训练主函数
    :param args: 配置对象(SimpleNamespace), 由JSON配置文件解析而来
    :param raw_cfg_dict: 原始字典，用于回写 run_id
    :param cfg_path: 配置文件路径，用于修改保存 run_id
    """
    # 初始化日志
    logger = setup_logger(args.out_dir)
    train_start_time = time.time()

    logger.info(f"========== Start New Experiment: {args.run_name} ==========")
    logger.info(
        f"Max training iterations: {args.max_iters}, Resume training: {args.resume}"
    )

    # ====================== 数据加载 ======================
    train_data: npt.NDArray = load_mmap_corpus(args.train_data_path, dtype=np.int64)
    val_data: npt.NDArray = load_mmap_corpus(args.valid_data_path, dtype=np.int64)
    device = try_gpu()
    logger.info(
        f"Train corpus shape: {train_data.shape}, Val corpus shape: {val_data.shape}"
    )
    logger.info(f"Using device: {device}")

    # ====================== 模型、优化器、学习率调度器初始化 ======================
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

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.alpha_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

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
        start_iter = load_checkpoint(args.ckpt_path, model, optimizer)
        scheduler.last_epoch = start_iter
        logger.info(
            f"Resumed checkpoint from {args.ckpt_path}, loaded finished iter index: {start_iter}"
        )
        # 断点代表该轮已跑完，下一轮迭代+1，解决续训多跑一轮
        start_iter += 1

    # ====================== Wandb 自动绑定 run_id（核心逻辑） ======================
    wandb_id = raw_cfg_dict.get("run_id")
    wandb_init_kwargs = {
        "project": "assignment1-basic",
        "name": args.run_name,
        "config": args,
        "dir": "./output",
        "mode": "offline",
    }

    if args.resume:
        # 续训模式：必须存在 run_id，恢复原有 run，曲线连续
        if not wandb_id:
            raise RuntimeError(
                "Resume mode enabled, but config.json 'run_id' is empty! "
                "Cannot resume wandb run. First train must generate run_id automatically."
            )
        wandb_init_kwargs["resume"] = "must"
        wandb_init_kwargs["id"] = wandb_id
        logger.info(f"Resuming wandb run with id: {wandb_id}")
    else:
        # 首次训练：生成唯一 run_id，写入 config.json 持久化
        if wandb_id is None or wandb_id == "":
            new_run_id = wandb.util.generate_id()
            raw_cfg_dict["run_id"] = new_run_id
            save_config_json(cfg_path, raw_cfg_dict)
            wandb_init_kwargs["id"] = new_run_id
            logger.info(
                f"First run, generated wandb run_id: {new_run_id}, saved to config.json"
            )
        else:
            # 配置已有 run_id 但不是续训（手动预设场景）
            wandb_init_kwargs["id"] = wandb_id
            logger.info(f"Using pre-set run_id from config: {wandb_id}")

    wandb.init(**wandb_init_kwargs)

    # ====================== 训练主循环 ======================
    for it in range(start_iter, args.max_iters):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm, args.eps)
        optimizer.step()

        it_log = it + 1
        if it_log % 100 == 0 or it_log == args.max_iters:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(
                    val_data, args.batch_size, args.context_length, device
                )
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)

                current_lr = scheduler.get_last_lr()[0]
                elapsed_total_sec = time.time() - train_start_time

                logger.info(
                    f"Iter {it_log:6d} | train_loss={loss.item():.4f} | val_loss={v_loss.item():.4f} "
                    f"| lr={current_lr:.6e} | elapsed={elapsed_total_sec:.2f}s"
                )

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": v_loss.item(),
                        "lr": current_lr,
                        "wall_time_seconds": elapsed_total_sec,
                    },
                    step=it_log,  # 关键：强制内置_step = 你的迭代序号it_log
                )

        # 每1000步保存，规避续训开局重复覆盖ckpt
        if it_log % 1000 == 0:
            ckpt_save_path = os.path.join(args.out_dir, f"ckpt_it{it_log}.pt")
            save_checkpoint(model, optimizer, it, ckpt_save_path)
            logger.info(f"Saved intermediate checkpoint: {ckpt_save_path}")

        scheduler.step()

    # 训练结束保存最终断点
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
    parser = argparse.ArgumentParser(description="Transformer LM Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to json config file"
    )
    cli_args = parser.parse_args()
    cfg_path = cli_args.config

    # 读取原始字典 + 转为Namespace
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg_dict = json.load(f)
    cfg_obj = json.loads(
        json.dumps(raw_cfg_dict), object_hook=lambda d: SimpleNamespace(**d)
    )

    os.makedirs(cfg_obj.out_dir, exist_ok=True)
    main(cfg_obj, raw_cfg_dict, cfg_path)
