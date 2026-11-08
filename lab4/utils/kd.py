import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Response-based KD loss.
    z_s, z_t: (N, C, H, W) student and teacher logits. y: (N, H, W) labels.

    L = alpha * H(y, softmax(z_s)) + beta * H(softmax(z_t / tau), softmax(z_s / tau))
    """
    # 1. Hard term: usual cross-entropy with the ground truth (T = 1), void pixels ignored.
    hard = F.cross_entropy(z_s, y, ignore_index=ignore_index)

    # 2. Softened teacher probabilities and student log-probabilities over the class dim.
    p_t = F.softmax(z_t / tau, dim=1)            # (N, C, H, W)
    log_p_s = F.log_softmax(z_s / tau, dim=1)    # (N, C, H, W)

    # 3. Per-pixel soft cross-entropy  -sum_c p_t * log p_s, averaged over non-void pixels.
    soft_per_pixel = -(p_t * log_p_s).sum(dim=1)  # (N, H, W)
    valid = y != ignore_index
    soft = soft_per_pixel[valid].mean()

    # 4. Weighted sum.
    return alpha * hard + beta * soft


def feature_kd_loss(f_s, f_t):
    """Feature-based KD loss: 1 - cosine similarity, averaged over all positions.
    f_s: (N, C, h, w) projected student features. f_t: (N, C, H, W) teacher features."""
    # Bring the student feature to the teacher's spatial resolution.
    if f_s.shape[-2:] != f_t.shape[-2:]:
        f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode='bilinear', align_corners=False)
    # Cosine similarity across channels at every pixel -> (N, H, W), in [-1, 1].
    cos = F.cosine_similarity(f_s, f_t, dim=1)
    return (1 - cos).mean()
