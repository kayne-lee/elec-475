# ELEC 475 Lab 3: Pet Nose Localization

The prelab loads the course nose annotations, implements SnoutNet and the
AlexNet/VGG16 variants, and verifies the provided trained checkpoints. No training
is needed for the prelab.

## Set up after cloning

From the repository root, create your own environment and install dependencies:

```bash
cd lab3_template
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dataset, checkpoints and virtual environment are excluded from Git. Download
and extract the course dataset into `data/oxford-iiit-pet-noses/` using the layout
below. Download `snoutnet.pth`, `alexnet.pth` and `vgg16.pth` into `checkpoints/`:

- Dataset: https://drive.google.com/drive/folders/1-IoMxHkbaWlW6w3YA-Fn9ZDv_T6L3pZ4
- Checkpoints: https://drive.google.com/drive/folders/1-OKNBTc_ptVeb49u-2QfNxbb8n2sqKGR

After those files are in place, run the verification commands below. The supplied
plot records the visual check from the initial verified setup.

## Run an existing setup

From the repository root:

```bash
cd lab3_template
source .venv/bin/activate
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
python -m utils.dataset
python -m utils.check_dataset
python -m models.snoutnet
open plots/dataset_check.png
```

Use `.venv/bin/python` as the interpreter in Cursor. Open `lab3_template` as the
project folder to use the included interpreter setting. Run the module commands
from that folder so the relative data/checkpoint paths resolve correctly.

## Files

```text
lab3_template/
├── .venv/                         # Isolated Python environment
├── checkpoints/                   # snoutnet.pth, alexnet.pth, vgg16.pth
├── data/oxford-iiit-pet-noses/
│   ├── images-original/images/    # Original Oxford pet images
│   ├── train_noses.txt            # 5,494 course nose labels
│   └── test_noses.txt             # 698 course nose labels
├── models/snoutnet.py             # Three models and checkpoint verification
├── utils/dataset.py               # Dataset and display denormalization
├── utils/check_dataset.py         # Full-split checks and eight-image plot
└── plots/dataset_check.png         # Visual check of training labels
```

## What the code does

`PetNoseDataset` returns three float32 tensors: an ImageNet-normalized image of
shape `(3, 227, 227)`, a nose label `(u, v)` in resized-image pixels, and the
original image size `(W, H)`. Here `u` is x/column and `v` is y/row. Images are
converted to RGB and resized with PIL bilinear interpolation, matching the
provided weights.

The label scales independently along each axis:

```text
u_resized = u_original * 227 / W
v_resized = v_original * 227 / H
```

The three models return `(B, 2)` normalized coordinates. Multiply a prediction
by `227` for resized-image pixels, or by `(W, H)` for original-image pixels.
Dataset labels are in pixels; model outputs are normalized.

The dataset check reads every sample in both splits, checks tensor shapes,
types, finite values and coordinate bounds, then saves eight deterministic
random training examples. Inspect the saved figure: each red dot should land
on a nose.

The model check prints SnoutNet's layer summary, checks output shapes and
parameter counts, loads each checkpoint with `strict=True`, and checks the known
test prediction. It uses CPU for reproducibility and builds the large backbones
one at a time. `pretrained=False` avoids redundant ImageNet downloads when
loading the supplied course checkpoints.

| Model | Parameters |
| --- | ---: |
| SnoutNet | 5,617,794 |
| SnoutNetAlexNet | 57,012,034 |
| SnoutNetVGG16 | 134,268,738 |

For test sample 0 (`basset_hound_60.jpg`), the original-image ground truth is
`(311, 152)`. The expected SnoutNet prediction is `(350.3443, 181.7347)`, with
a tolerance of one pixel per coordinate. This verifies that the implementation
matches the supplied pipeline; the prediction itself need not equal the label.

## Pooling calculation for checkoff

Each convolution preserves image size. Pooling uses kernel 4, stride 4 and
`ceil_mode=True`, so each spatial dimension becomes `ceil((N - 4) / 4) + 1`:

```text
227 -> ceil(223 / 4) + 1 = 57
 57 -> ceil( 53 / 4) + 1 = 15
 15 -> ceil( 11 / 4) + 1 =  4
```

The last feature map has 256 channels and size 4 by 4, giving 4,096 inputs to
`fc1`. The final regression layer has no output activation.

## Recreate the environment

This setup uses a standard Python venv. Conda and CUDA are not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-lock.txt` records the versions installed and verified on this Mac.
Use `python -m pip install -r requirements-lock.txt` to reproduce those versions
with Python 3.14 on a compatible macOS installation.

## Data sources

The train/test nose annotations come from the course Drive folder linked in the
prelab. The original images come from the fast.ai mirror of the Oxford-IIIT Pet
dataset, rather than downloading thousands of individual Drive files:

- Course annotations: https://drive.google.com/drive/folders/1-IoMxHkbaWlW6w3YA-Fn9ZDv_T6L3pZ4
- Oxford dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Image mirror: https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz

The checkpoint files were extracted from the supplied `checkpoints 1.zip`.
The original ZIPs, PDF, data and checkpoints are kept locally and are not included in this repository.

## Verified on October 22, 2026

- Python 3.14.6, PyTorch 2.14.1, torchvision 0.29.1.
- 7,390 images extracted; a sampled image matched the course copy byte for byte.
- All 5,494 training samples and 698 test samples passed the complete data scan.
- All three models matched the required output shapes and parameter counts and
  loaded the supplied checkpoints with strict matching.
- Test sample 0: ground truth `(311, 152)`, prediction `(350.3442, 181.7347)`.
- Additional checks covered RGB, grayscale, palette and RGBA images, non-square
  coordinate scaling, and malformed-input errors.

The starter package imports can produce a harmless `RuntimeWarning` when running
`python -m models.snoutnet` or `python -m utils.dataset`, as noted in the handout.
