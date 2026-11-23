import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegTransform:
    """Resizes an (image, mask) pair to a fixed size and converts both to tensors."""

    def __init__(self, size=320):
        self.size = [size, size]

    def __call__(self, img, mask):
        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


class SegTrainTransform:
    """Training augmentation: resize to size x size, random scale in [lo, hi], random crop
    (padding with void = 255 when the scaled image is smaller), random horizontal flip."""

    def __init__(self, size=320, scale=(0.5, 2.0)):
        self.size, self.scale = size, scale

    def __call__(self, img, mask):
        s = int(self.size * random.uniform(*self.scale))
        img = TF.resize(img, [s, s], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [s, s], interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        if s < self.size:
            pad = self.size - s
            img = torch.nn.functional.pad(img, (0, pad, 0, pad), value=0.0)
            mask = torch.nn.functional.pad(mask, (0, pad, 0, pad), value=255)
        i = random.randint(0, img.shape[1] - self.size)
        j = random.randint(0, img.shape[2] - self.size)
        img = img[:, i:i + self.size, j:j + self.size]
        mask = mask[i:i + self.size, j:j + self.size]
        if random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        return img.contiguous(), mask.contiguous()


def get_voc(data_dir, image_set, size=320, train=False):
    tf = SegTrainTransform(size) if train else SegTransform(size)
    return VOCSegmentation(data_dir, year='2012', image_set=image_set, download=False,
                           transforms=tf)
