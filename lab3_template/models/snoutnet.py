"""
ELEC 475: Lab 3 - SnoutNet Nose Localization Models
Three regression models that predict a pet's nose position (u, v):
    SnoutNet        : Custom 3-block CNN trained from scratch
    SnoutNetAlexNet : SnoutNet-A, ImageNet-pretrained torchvision AlexNet backbone
    SnoutNetVGG16   : SnoutNet-V, ImageNet-pretrained torchvision VGG16 backbone

All models take (B, 3, 227, 227) input and return (B, 2) NORMALIZED coordinates
(u/227, v/227). Layer names must match exactly for the provided weights to load.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16


class SnoutNet(nn.Module):
    """
    SnoutNet Architecture for (3, 227, 227) RGB images.

    Spatial dimension arithmetic (pooling with ceil_mode=True rounds UP):
        Input        : (B, 3, 227, 227)
        Conv1 (p=1)  : (B, 64, 227, 227)
        MaxPool1     : ceil((227 - 4)/4) + 1 = 57 --> (B, 64, 57, 57)
        Conv2 (p=1)  : (B, 128, 57, 57)
        MaxPool2     : ceil((57 - 4)/4) + 1 = 15  --> (B, 128, 15, 15)
        Conv3 (p=1)  : (B, 256, 15, 15)
        MaxPool3     : ceil((15 - 4)/4) + 1 = 4   --> (B, 256, 4, 4)
        Flatten      : 256 * 4 * 4 = 4,096 features
        FC1          : 4,096 -> 1,024
        FC2          : 1,024 -> 1,024
        FC3          : 1,024 -> 2
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SnoutNetAlexNet(nn.Module):
    """
    SnoutNet-A: torchvision AlexNet with its final 1000-class layer replaced
    by a 2-output regression layer.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.backbone = alexnet(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SnoutNetVGG16(nn.Module):
    """
    SnoutNet-V: torchvision VGG16 with its final 1000-class layer replaced
    by a 2-output regression layer.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    from torchinfo import summary
    from utils.dataset import IMG_SIZE, PetNoseDataset

    torch.manual_seed(0)
    # Avoid excessive CPU thread overhead on the small verification batches.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    summary(SnoutNet(), input_size=(1, 3, IMG_SIZE, IMG_SIZE), device="cpu")
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    checks = [
        (SnoutNet, "snoutnet.pth", 5_617_794),
        (SnoutNetAlexNet, "alexnet.pth", 57_012_034),
        (SnoutNetVGG16, "vgg16.pth", 134_268_738),
    ]
    # Build one backbone at a time to keep memory use modest on laptops.
    snoutnet = None
    for model_class, filename, expected_count in checks:
        model = (
            model_class() if model_class is SnoutNet else model_class(pretrained=False)
        )
        model.eval()
        with torch.no_grad():
            output = model(sample)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if tuple(output.shape) != (1, 2) or not torch.isfinite(output).all():
            raise AssertionError(f"{model_class.__name__}: invalid output {output}")
        if parameter_count != expected_count:
            raise AssertionError(
                f"{model_class.__name__}: expected {expected_count:,} parameters, got {parameter_count:,}"
            )
        print(
            f"{model_class.__name__} output {tuple(output.shape)} params {parameter_count:,}"
        )
        state = torch.load(
            f"checkpoints/{filename}", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            if not torch.isfinite(model(sample)).all():
                raise AssertionError(
                    f"{filename}: loaded weights produced non-finite predictions"
                )
        print(f"Loaded {filename}")
        if model_class is SnoutNet:
            snoutnet = model
        del state, model

    dataset = PetNoseDataset(split="test")
    image, label, orig_size = dataset[0]
    with torch.no_grad():
        prediction = snoutnet(image.unsqueeze(0))[0] * orig_size
    ground_truth = label / IMG_SIZE * orig_size
    print("Ground truth:", ground_truth)
    print("Prediction:  ", prediction)
    torch.testing.assert_close(
        ground_truth, torch.tensor([311.0, 152.0]), atol=0.001, rtol=0
    )
    torch.testing.assert_close(
        prediction, torch.tensor([350.3443, 181.7347]), atol=1, rtol=0
    )
    print(
        "PASS: all checkpoints loaded and the known prediction is within 1 original-image pixel."
    )
