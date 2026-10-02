import argparse
import csv
import random
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size, workers, seed):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    full_train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    full_val = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=eval_tf)
    test_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=eval_tf)

    # 45k train / 5k validation (fixed split via seed)
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()
    train_set = Subset(full_train, perm[:45000])
    val_set = Subset(full_val, perm[45000:])

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def build_model():
    model = resnet18(num_classes=10)
    # Adapt to 32x32 images: 3x3 first conv and no maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/baseline")
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        wandb.init(project="resnet18-cifar10-baseline", name=out_dir.name, config=vars(args))

    train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.workers, args.seed)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_file = open(out_dir / "log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        scheduler.step()

        train_loss, train_acc = loss_sum / n, correct / n
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
        log_file.flush()
        print(f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
            })

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    # Final evaluation on the test set: run once, using the best validation checkpoint
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"best val acc: {best_val:.4f} | TEST acc: {test_acc:.4f}")
    (out_dir / "test_result.txt").write_text(f"best_val_acc={best_val:.4f}\ntest_acc={test_acc:.4f}\n")

    if args.wandb:
        wandb.summary["best_val_acc"] = best_val
        wandb.summary["test_acc"] = test_acc
        wandb.finish()


if __name__ == "__main__":
    main()
