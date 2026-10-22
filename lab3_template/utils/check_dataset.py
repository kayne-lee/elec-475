"""Validate both complete splits and save eight training labels for visual review."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import IMG_SIZE, PetNoseDataset, denormalize

plt.switch_backend("Agg")


def validate_split(split: str, expected_count: int) -> PetNoseDataset:
    dataset = PetNoseDataset(split=split)
    if len(dataset) != expected_count:
        raise AssertionError(
            f"{split}: expected {expected_count} labels, got {len(dataset)}"
        )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    sample_count = 0
    for images, labels, original_sizes in tqdm(loader, desc=f"Checking {split}"):
        batch_size = images.shape[0]
        if tuple(images.shape) != (batch_size, 3, IMG_SIZE, IMG_SIZE):
            raise AssertionError(f"{split}: invalid image shape {tuple(images.shape)}")
        if tuple(labels.shape) != (batch_size, 2) or tuple(original_sizes.shape) != (
            batch_size,
            2,
        ):
            raise AssertionError(
                f"{split}: labels and original sizes must have shape (B, 2)"
            )
        for name, tensor in [
            ("images", images),
            ("labels", labels),
            ("original sizes", original_sizes),
        ]:
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise AssertionError(f"{split}: {name} must be finite float32 tensors")
        invalid = ((labels < 0) | (labels >= IMG_SIZE)).any(dim=1)
        if invalid.any():
            indices = invalid.nonzero().flatten().tolist()
            names = [dataset.samples[sample_count + index][0] for index in indices]
            raise AssertionError(f"{split}: labels outside the image frame: {names}")
        if (original_sizes <= 0).any():
            raise AssertionError(f"{split}: original image dimensions must be positive")
        sample_count += batch_size
    print(
        f"PASS: {split}: all {sample_count:,} samples have valid shapes and coordinates."
    )
    return dataset


def save_label_plot(dataset: PetNoseDataset) -> None:
    generator = torch.Generator().manual_seed(475)
    indices = torch.randperm(len(dataset), generator=generator)[:8].tolist()
    figure, axes = plt.subplots(2, 4, figsize=(13, 7))
    for axis, index in zip(axes.flat, indices):
        image, label, _ = dataset[index]
        axis.imshow(denormalize(image).permute(1, 2, 0).numpy())
        axis.scatter(
            label[0].item(),
            label[1].item(),
            s=32,
            c="red",
            edgecolors="white",
            linewidths=0.8,
        )
        axis.set_title(dataset.samples[index][0], fontsize=9)
        axis.axis("off")
    figure.suptitle("Training labels: each red dot should sit on a nose")
    figure.tight_layout(pad=1.5, h_pad=3.0, rect=(0, 0, 1, 0.95))
    output = Path("plots/dataset_check.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved {output}; inspect all eight dots before lab checkoff.")


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    training = validate_split("train", 5494)
    validate_split("test", 698)
    save_label_plot(training)


if __name__ == "__main__":
    main()
