"""
Segmentation metrics: confusion-matrix based IoU / mIoU / pixel accuracy.

mIoU is computed the standard (PASCAL VOC / torchvision reference) way: a single
confusion matrix is accumulated over the WHOLE dataset, then
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
and mIoU is the mean over classes. Pixels labelled 255 (void) are ignored.
"""
import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: integer tensors of identical shape (e.g. [B, H, W])."""
        assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
        pred = pred.flatten().cpu()
        target = target.flatten().cpu()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        # rows = ground truth, cols = prediction
        idx = target[valid] * self.num_classes + pred[valid]
        self.confmat += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confmat.double()
        tp = torch.diag(cm)
        gt_total = cm.sum(dim=1)     # TP + FN
        pred_total = cm.sum(dim=0)   # TP + FP
        union = gt_total + pred_total - tp
        iou = tp / union.clamp(min=1)
        present = union > 0          # skip classes absent from both GT and preds
        return {
            "pixel_acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
            "mean_class_acc": (tp / gt_total.clamp(min=1))[gt_total > 0].mean().item(),
            "iou_per_class": iou.tolist(),
            "miou": iou[present].mean().item(),
        }

    def reset(self):
        self.confmat.zero_()
