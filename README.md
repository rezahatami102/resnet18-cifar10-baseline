# ResNet18 on CIFAR-10 (Baseline)

A reproducible PyTorch baseline: ResNet18 trained from scratch on CIFAR-10.

## Results

| Split | Accuracy |
|-------|----------|
| Validation (best) | 94.08% |
| Test | 93.25% |

Training curves: [Weights & Biases run](https://wandb.ai/rezahatami102-part-ai-research-center/resnet18-cifar10-baseline/runs/mdigmy2h)

## Setup

- Model: torchvision ResNet18, stem adapted to 32x32 (3x3 conv, no maxpool), trained from scratch
- Data: CIFAR-10, 45k train / 5k validation (fixed split), standard 10k test set
- Augmentation: random crop (padding 4), horizontal flip
- Optimizer: SGD (momentum 0.9, Nesterov), lr 0.1, weight decay 5e-4, cosine schedule
- Training: 30 epochs, batch size 128, bf16 autocast, seed 42
- Model selection on validation; test evaluated once at the end

## Usage

    conda activate ai
    python src/train.py --epochs 30 --out outputs/baseline