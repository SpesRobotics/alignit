import torch
import torch.nn as nn

class InversePredictionWeightedLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred_pos = pred[:, :3]
        pred_rot = pred[:, 3:]
        target_pos = target[:, :3]
        target_rot = target[:, 3:]

        weights_pos = 1.0 / (torch.abs(pred_pos) + self.epsilon)
        pos_loss = (weights_pos * (pred_pos - target_pos) ** 2).mean()

        rot_loss = torch.mean((pred_rot - target_rot) ** 2)

        loss = pos_loss + rot_loss
        return loss