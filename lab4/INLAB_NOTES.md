# Lab 4 In-Lab – Notes for the TA check

## Commands (run from `lab4/` with `.venv` active)

```bash
python download_voc.py                 # train: 1464 val: 1449
python -m models.student               # 3,265,173 params
python test_kd.py                      # 7/7 checks passed
python train.py --kd none     --iters 20 --workers 0
python train.py --kd response --iters 20 --workers 0
python train.py --kd feature  --iters 20 --workers 0
```

`--workers 0` is needed on macOS: `train.py` has no `if __name__ == '__main__':` guard, so
DataLoader worker processes (macOS uses "spawn") re-import the script and crash.
`train.py` was also changed to use Apple MPS when CUDA is unavailable (it was CUDA/CPU only).

## Kick-start results (Apple MPS, batch 16, 320×320)

| Run | avg loss, first 5 | avg loss, last 5 |
|---|---|---|
| none | 2.544 | 1.462 |
| response | 2.730 | 2.154 |
| feature | 4.285 | 2.722 |

Logs are in `kickstart_logs/`, and plots are in `plots/loss_{none,response,feature}.png`.
The losses are on different scales: the response loss includes the teacher's entropy (it can't
reach 0), and the feature loss adds two cosine terms (each in [0, 2]) to the CE.

## Explanation questions

**1. Response-based loss terms and why τ > 1.**
- `α · H(y, σ(z_s; T=1))` is the ordinary per-pixel cross-entropy with the ground-truth labels.
  It keeps the student accurate on the real task and anchors it if the teacher is wrong.
- `β · H(σ(z_t; τ), σ(z_s; τ))` is the cross-entropy between the teacher's and the student's
  softened distributions, which pushes the student to reproduce the teacher's whole probability
  vector at each pixel. It equals KL(p_t‖p_s) plus the teacher's entropy, and the entropy is a
  constant, so minimizing it minimizes the KL.
- At T = 1 the teacher is very confident (close to one-hot), so its output says little more
  than the label. Dividing the logits by τ > 1 flattens the softmax so the small probabilities
  on wrong classes become visible ("dark knowledge"), e.g. that a cat pixel looks a bit like dog
  and not at all like bus. Those relative similarities are information a one-hot label can't
  carry, and they give a richer, smoother training signal.

**2. Why adapters, and why they aren't counted.**
- The student and teacher features have different channel counts (24 vs 512, 576 vs 2048), so
  the per-pixel vectors live in different spaces and can't be compared directly. A learned 1×1
  convolution projects each student feature into the teacher's channel space; the high feature is
  also upsampled from stride 32 to stride 8 to line up spatially.
- The adapters exist only to compute the training loss. At inference the student uses its own
  ASPP + decoder path and never calls them, so they're thrown away after training, add no
  inference cost, and aren't part of the deployed model's 3,265,173 parameters.

**3. What the cosine loss measures; why 0 and 2.**
- At each pixel it measures the angle between the student and teacher feature vectors across
  channels: `cos = f_s·f_t / (‖f_s‖‖f_t‖)`. It compares direction (which channels fire together)
  and ignores magnitude, so scaling either feature doesn't change it. That matters because the
  two networks' activations have very different scales.
- cos ∈ [−1, 1], so `1 − cos ∈ [0, 2]`. Identical (or same-direction) vectors give cos = 1, so the
  loss is 0. Opposite vectors (f_s = −f_t) give cos = −1, so the loss is 1 − (−1) = 2. Orthogonal
  vectors give 1.
