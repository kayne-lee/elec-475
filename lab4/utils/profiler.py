import time
import torch


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def ms_per_image(model, device, size=320, runs=20):
    """Average inference time per 1-image batch, in milliseconds."""
    x = torch.randn(1, 3, size, size, device=device)
    model.eval()
    for _ in range(3):
        model(x)
    _sync(device)
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    _sync(device)
    return 1000 * (time.perf_counter() - t) / runs
