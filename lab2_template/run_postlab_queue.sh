#!/bin/bash
# Post-lab Steps 4 & 5: runs every remaining experiment sequentially on one device so
# epoch wall-clock times stay comparable. Each run's full console output goes to logs/.
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs

run() {
    local name=$1; shift
    if grep -q "Training completed" "logs/$name.log" 2>/dev/null; then
        echo "[skip] $name already finished"; return
    fi
    echo "[$(date +%H:%M)] start $name"
    python train.py "$@" > "logs/$name.log" 2>&1 || echo "[$(date +%H:%M)] FAILED $name"
    grep "Best Val" "logs/$name.log"
}

# ---- Step 2: Normalization study (augmentation off, matching Table 7.3a rows 3-5) ----
for norm in none lrn bn; do
    run alexnet_$norm --model alexnet --norm $norm --dataset cifar10 --epochs 10 --no-augment \
        --save-model checkpoints/alexnet_$norm.pth --plot-file plots/loss_alexnet_$norm.png
done

# ---- Step 3: Data augmentation study ----
run alexnet_no_aug --model alexnet --norm bn --dataset cifar10 --epochs 15 --no-augment \
    --save-model checkpoints/alexnet_no_aug.pth --plot-file plots/loss_no_aug.png
run alexnet_with_aug --model alexnet --norm bn --dataset cifar10 --epochs 15 \
    --save-model checkpoints/alexnet_with_aug.pth --plot-file plots/loss_with_aug.png

# ---- Step 4: Dual-dataset benchmark (handout command, augmentation on) ----
run alexnet_tiny_imagenet --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 \
    --save-model checkpoints/alexnet_tiny_imagenet.pth --plot-file plots/loss_tiny_imagenet.png

# ---- Step 5: remaining CIFAR-10 rows of Tables 7.3a/7.3b ----
run lenet --model lenet --dataset cifar10 --epochs 10 --no-augment \
    --save-model checkpoints/lenet_baseline.pth --plot-file plots/loss_lenet.png
run lenet_modern --model lenet_modern --dataset cifar10 --epochs 10 --no-augment \
    --save-model checkpoints/lenet_modern.pth --plot-file plots/loss_lenet_modern.png
run zoo_alexnet --model zoo_alexnet --dataset cifar10 --epochs 10 --no-augment \
    --save-model checkpoints/zoo_alexnet.pth --plot-file plots/loss_zoo_alexnet.png

# ---- Step 5: remaining Tiny ImageNet rows of Table 7.3b ----
run tiny_lenet --model lenet --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_lenet.pth --plot-file plots/loss_tiny_lenet.png
run tiny_lenet_modern --model lenet_modern --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_lenet_modern.pth --plot-file plots/loss_tiny_lenet_modern.png
run tiny_alexnet_none --model alexnet --norm none --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_alexnet_none.pth --plot-file plots/loss_tiny_alexnet_none.png
run tiny_alexnet_lrn --model alexnet --norm lrn --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_alexnet_lrn.pth --plot-file plots/loss_tiny_alexnet_lrn.png
run tiny_alexnet_bn --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_alexnet_bn.pth --plot-file plots/loss_tiny_alexnet_bn.png
run tiny_zoo_alexnet --model zoo_alexnet --dataset tiny_imagenet --epochs 10 --no-augment \
    --save-model checkpoints/tiny_zoo_alexnet.pth --plot-file plots/loss_tiny_zoo_alexnet.png

# ---- Step 5: multi-seed repeats of AlexNet + BatchNorm on CIFAR-10 (seed 42 = logs/alexnet_bn.log) ----
for seed in 123 999; do
    run alexnet_bn_seed$seed --model alexnet --norm bn --dataset cifar10 --epochs 10 --no-augment --seed $seed \
        --save-model checkpoints/alexnet_bn_seed$seed.pth --plot-file plots/loss_alexnet_bn_seed$seed.png
done

# ---- Step 5: complexity benchmark + ablation plot ----
python compare_models.py --dataset cifar10 > logs/compare_models.log 2>&1
python -m utils.visualization --plot-ablation > logs/plot_ablation.log 2>&1
echo "[$(date +%H:%M)] queue complete"
