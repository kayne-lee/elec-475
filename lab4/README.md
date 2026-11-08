# ELEC 475 Lab 4 – Semantic Segmentation & Knowledge Distillation

## Pre-Lab: Pretrained FCN-ResNet50 on PASCAL VOC 2012

```bash
cd lab4
../.venv/bin/python test_fcn.py                  # full val set, downloads VOC to ./data
../.venv/bin/python test_fcn.py --limit 50       # quick sanity check
../.venv/bin/python test_fcn.py --save-vis 8     # save qualitative examples to ./plots
```

**Files**
- `test_fcn.py`: loads `fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)`, evaluates on VOC 2012 `val`, prints per-class IoU / pixel acc / mIoU.
- `utils/dataset.py`: `VOCSegmentation` loader + eval transform (ImageNet normalization, shorter side resized to 520; masks kept at native resolution).
- `utils/metrics.py`: confusion-matrix mIoU accumulated over the whole dataset, ignoring void pixels (255).

**Protocol:** batch size 1, the image is resized so its shorter side is 520, then the logits are bilinearly upsampled back to the original mask size before argmax. IoU_c = TP / (TP + FP + FN), averaged over the 21 classes.

**Result (VOC 2012 val, 1449 images, Apple MPS):**

| Metric | Value |
|---|---|
| mIoU | **69.62%** |
| Pixel accuracy | 93.12% |
| Mean class accuracy | 78.66% |

Full per-class output: `results_fcn_resnet50.txt`. Sample predictions: `plots/`.

> Note: the official VOC host (host.robots.ox.ac.uk) can be very slow. The identical tar
> (MD5 `6cd6e144f989b92b3379bac3b3de84fd`) is mirrored at
> `http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`. Place it in `./data/` and
> torchvision will verify it and extract it.
