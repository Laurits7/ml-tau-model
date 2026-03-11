import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Multi-class Focal Loss"""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Class weights (tensor of size num_classes)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class indices
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get predicted probabilities for true classes
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Usage in your model:
# self.focal_loss = FocalLoss(alpha=torch.tensor([0.1, 1.0, 1.0, 1.0]), gamma=2.0)  # Lower weight for background


class SigmoidFocalLoss(nn.Module):
    """Wrapper to make sigmoid_focal_loss behave like a module for consistency."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets.float(),  # targets need to be float for sigmoid
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
