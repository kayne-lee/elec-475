"""
ELEC 475: Lab 4 Pre-Lab - Test Pretrained FCN-ResNet50 on PASCAL VOC 2012

Loads torchvision's pretrained FCN-ResNet50 (21 VOC classes), downloads the
PASCAL VOC 2012 segmentation dataset via torchvision, and evaluates on the
`val` split with mean Intersection over Union (mIoU).

Example:
    python test_fcn.py                     # full val set (1449 images)
    python test_fcn.py --limit 50          # quick sanity check
    python test_fcn.py --save-vis 8        # also save 8 qualitative examples
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from utils.prelab_dataset import (get_voc_val_loader, voc_colormap,
                                  VOC_CLASSES, NUM_CLASSES, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD)
from utils.prelab_metrics import SegmentationMetric


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_visualization(img: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = voc_colormap()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    # Image was resized for the network; show masks at native resolution.
    img = F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None], size=mask.shape,
                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(axes,
                             [img, cmap[mask.cpu().numpy()], cmap[pred.cpu().numpy()]],
                             ["Image", "Ground truth", "FCN-ResNet50"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def evaluate(args):
    device = get_device()
    print("=" * 70)
    print("ELEC 475 Lab 4 Pre-Lab: FCN-ResNet50 on PASCAL VOC 2012 (val)")
    print(f"Device      : {device}")
    print(f"Data root   : {os.path.abspath(args.data_dir)}")
    print(f"Resize      : shorter side -> {args.resize} (predictions upsampled to native size)")
    print("=" * 70)

    # 1. Model: COCO-trained on the 20 VOC categories (+ background) -> 21 outputs.
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights, aux_loss=True).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded FCN-ResNet50 ({n_params / 1e6:.2f}M params), categories: {len(weights.meta['categories'])}")

    # 2. Data
    loader = get_voc_val_loader(root=args.data_dir, resize=args.resize,
                                download=not args.no_download, num_workers=args.workers)
    total = len(loader) if args.limit is None else min(args.limit, len(loader))
    print(f"VOC 2012 val images: {len(loader.dataset)} (evaluating {total})")

    # 3. Evaluate
    metric = SegmentationMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    start = time.time()
    for i, (img, mask) in enumerate(loader):
        if i >= total:
            break
        img = img.to(device)                      # [1, 3, h, w]
        mask = mask.to(device)                    # [1, H, W]  (native resolution)
        logits = model(img)["out"]                # [1, 21, h, w]
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)               # [1, H, W]
        metric.update(pred, mask)

        if i < args.save_vis:
            save_visualization(img[0], mask[0], pred[0],
                               os.path.join(args.vis_dir, f"sample_{i:03d}.png"))
        if (i + 1) % 100 == 0 or i + 1 == total:
            el = time.time() - start
            print(f"  [{i + 1:4d}/{total}] running mIoU={metric.compute()['miou'] * 100:.2f}%  "
                  f"({(i + 1) / el:.1f} img/s)")

    # 4. Report
    res = metric.compute()
    print("\nPer-class IoU:")
    for name, iou in zip(VOC_CLASSES, res["iou_per_class"]):
        print(f"  {name:<12s} {iou * 100:6.2f}%")
    print("-" * 70)
    print(f"Pixel accuracy      : {res['pixel_acc'] * 100:.2f}%")
    print(f"Mean class accuracy : {res['mean_class_acc'] * 100:.2f}%")
    print(f"mIoU                : {res['miou'] * 100:.2f}%")
    print(f"Elapsed             : {time.time() - start:.1f}s")
    return res


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pretrained FCN-ResNet50 on VOC 2012 val (mIoU).")
    p.add_argument("--data-dir", default="./data", help="where VOC is (or will be) downloaded")
    p.add_argument("--resize", type=int, default=520, help="shorter-side resize for network input")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-download", action="store_true", help="dataset already present")
    p.add_argument("--save-vis", type=int, default=0, help="save N qualitative examples")
    p.add_argument("--vis-dir", default="./plots")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
