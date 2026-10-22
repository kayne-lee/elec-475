"""
ELEC 475: Lab 3 - Pet Nose Localization Dataset
Custom PyTorch Dataset for the oxford-iiit-pet-noses re-annotation of the
Oxford-IIIT Pet Dataset.

Each sample returns (image, label, orig_size):
    image     : (3, 227, 227) float tensor, ImageNet-normalized
    label     : (2,) float tensor, nose position (u, v) in the 227x227 image (pixels)
                u = column (x), v = row (y)
    orig_size : (2,) float tensor, original image (W, H)
"""

import csv
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 227

# ImageNet channel statistics: required by the pretrained AlexNet/VGG16 backbones,
# and used by the provided SnoutNet weights as well.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetNoseDataset(Dataset):
    """
    Loads pet images and nose labels, resized to (3, 227, 227).

    Label file format (one sample per line):
        beagle_145.jpg,"(198, 304)"
    The ordered pair is (u, v) in pixels of the ORIGINAL image.
    """

    def __init__(self, root: str = "data/oxford-iiit-pet-noses", split: str = "train"):
        super().__init__()
        self.img_dir = Path(root) / "images-original" / "images"
        label_file = Path(root) / f"{split}_noses.txt"

        self.samples = []
        if split not in {"train", "test"}:
            raise ValueError(f"Expected split 'train' or 'test', got {split!r}")
        with label_file.open(newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise ValueError(
                        f"{label_file}:{line_number}: expected filename and (u, v)"
                    )
                name, coordinates = row
                try:
                    u, v = (
                        float(value.strip())
                        for value in coordinates.strip().strip("()").split(",")
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{label_file}:{line_number}: invalid coordinates {coordinates!r}"
                    ) from error
                self.samples.append((name.strip(), u, v))
        if not self.samples:
            raise ValueError(f"No samples found in {label_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, u, v = self.samples[idx]
        with Image.open(self.img_dir / name) as source:
            image = source.convert("RGB")
        width, height = image.size
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        label = torch.tensor(
            [u * IMG_SIZE / width, v * IMG_SIZE / height], dtype=torch.float32
        )
        orig_size = torch.tensor([width, height], dtype=torch.float32)
        return image, label, orig_size


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display (returns values in [0, 1])."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    print("Testing PetNoseDataset:")
    for split in ["train", "test"]:
        ds = PetNoseDataset(split=split)
        image, label, orig_size = ds[0]
        print(
            f"  {split:5s}: {len(ds):,} samples | image {tuple(image.shape)} "
            f"| label {label.tolist()} | orig_size {orig_size.tolist()}"
        )
