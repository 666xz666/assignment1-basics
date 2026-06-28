import torch
import wandb

from cs336_basics import SGD


def main():
    log_root = "./output/logs/wandb/lr_tuning_exp"  # 自定义存放根目录
    for lr in [1e1, 1e2, 1e3]:
        wandb.init(
            project="lr-tuning",
            config={"lr": lr},
            dir=log_root,  # 指定wandb内部运行文件存放位置
            resume="allow",
            name=f"lr = {lr:.0e}",
        )

        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            wandb.log({"loss": loss.item(), "step": t})
            loss.backward()
            opt.step()

        wandb.finish()


if __name__ == "__main__":
    main()
