"""
PASCAL VOC 2012 segmentation dataset helpers.

Uses torchvision.datasets.VOCSegmentation. Images are normalized with ImageNet
statistics (what the pretrained FCN-ResNet50 expects); masks are kept as raw
integer class indices in [0, 20] with 255 marking "void" / object boundaries.
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255
NUM_CLASSES = 21

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def voc_colormap(n: int = 256) -> np.ndarray:
    """Standard PASCAL VOC palette (bit-interleaved), shape [n, 3] uint8."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


class SegEvalTransform:
    """
    Joint image/mask transform for evaluation.

    The image is resized so its shorter side is `resize` (bilinear) and
    normalized. The mask is returned at its ORIGINAL resolution so that
    predictions are upsampled back and scored against the true labels.
    """

    def __init__(self, resize: int | None = 520):
        self.resize = resize

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.resize is not None:
            img = TF.resize(img, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        # Mask PNGs are palette-mode; np.array gives the raw class indices.
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


def get_voc_val_loader(root: str = "./data", resize: int | None = 520,
                       download: bool = True, num_workers: int = 2) -> DataLoader:
    """
    VOC 2012 `val` split (1449 images). batch_size is fixed to 1 because
    images have different sizes and masks are kept at native resolution.
    """
    dataset = VOCSegmentation(
        root=root,
        year="2012",
        image_set="val",
        download=download,
        transforms=SegEvalTransform(resize),
    )
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
