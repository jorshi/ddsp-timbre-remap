"""
Differentiable Loss Functions
"""
import torch


class FeatureDifferenceLoss(torch.nn.Module):
    """
    Loss function that calculates the error between the difference of two features
    and a target difference.
    """

    def __init__(self, loss: callable = torch.nn.L1Loss()):
        super().__init__()
        self.loss = loss

    def forward(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        target_diff: torch.tensor,
        weight: float = 1.0,
    ):
        diff = y_pred - y_true
        error = torch.abs(diff - target_diff)
        error = error * weight
        return torch.mean(error)
