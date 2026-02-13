"""Custom loss functions for alignment training."""

import torch
import torch.nn as nn


class InverseWeightedPositionLoss(nn.Module):
    """
    Inverse-distance weighted loss for position, MSE for orientation.
    
    Position loss: uses 1/distance weighting so small errors have higher gradient impact.
    Orientation loss: standard MSE.
    
    For predicted output (B, 9): [pos_x, pos_y, pos_z, rot_6d_0...rot_6d_5]
    
    Example:
    - Position error 0.1 m → loss contribution ≈ 100 * error²  (high penalty)
    - Position error 1.0 m → loss contribution ≈ 1 * error²   (low penalty)
    - Orientation: always MSE regardless of magnitude
    """
    
    def __init__(self, pos_weight: float = 1.0, rot_weight: float = 1.0, epsilon: float = 0.01):
        """
        Args:
            pos_weight: Weight multiplier for position loss
            rot_weight: Weight multiplier for rotation loss
            epsilon: Small value to prevent division by zero (default: 0.01)
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values (B, 9) where first 3 are position, last 6 are rotation
            target: Target values (B, 9)
        
        Returns:
            Scalar loss value
        """
        # Split position and rotation
        pred_pos = pred[:, :3]
        pred_rot = pred[:, 3:]
        target_pos = target[:, :3]
        target_rot = target[:, 3:]
        
        # Position loss: inverse-weighted MSE
        # L_pos = (1 / ||target_pos||) * MSE_pos
        pos_distance = torch.norm(target_pos, dim=1, keepdim=True) + self.epsilon
        pos_mse = (pred_pos - target_pos) ** 2
        pos_loss = (pos_mse / pos_distance).mean()
        
        # Rotation loss: standard MSE
        rot_mse = (pred_rot - target_rot) ** 2
        rot_loss = rot_mse.mean()
        
        # Combine with weights
        total_loss = (self.pos_weight * pos_loss + self.rot_weight * rot_loss) / (self.pos_weight + self.rot_weight)
        return total_loss


class LogarithmicMSELoss(nn.Module):
    """
    Inverse-based MSE loss that penalizes smaller errors more heavily.
    
    Instead of: L = MSE(pred, target)
    We compute: L = MSE / (1 + MSE)
    
    This amplifies the importance of small errors in the loss landscape.
    As error approaches 0, gradients remain significant, encouraging precision.
    
    The key property:
    - When MSE → 0: gradient → base (large relative impact)
    - When MSE → ∞: loss → 1 (bounded, handles outliers)
    
    Example gradients (relative to error magnitude):
    - MSE error 0.1 → loss ≈ 0.0909, gradient ≈ 0.826
    - MSE error 1.0 → loss = 0.5, gradient ≈ 0.25
    - MSE error 10.0 → loss ≈ 0.909, gradient ≈ 0.0083
    
    Compare to standard MSE:
    - MSE error 0.1 → loss = 0.1, gradient = 1.0
    - MSE error 1.0 → loss = 1.0, gradient = 1.0
    - MSE error 10.0 → loss = 10.0, gradient = 1.0
    
    The inverse-based version increases the gradient for small errors,
    making the optimizer focus more on achieving high precision.
    """
    
    def __init__(self, base: float = 1.0):
        """
        Args:
            base: Scaling factor for the MSE in numerator. 
                  Higher values make loss increase faster with error.
                  (default: 1.0)
        """
        super().__init__()
        self.base = base
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values (B, output_dim)
            target: Target values (B, output_dim)
        
        Returns:
            Scalar loss value
        """
        mse = torch.mean((pred - target) ** 2)
        # MSE / (1 + MSE) ensures bounded loss and high penalty for small errors
        inverse_loss = (self.base * mse) / (1.0 + self.base * mse)
        return inverse_loss


class WeightedLogMSELoss(nn.Module):
    """
    Weighted logarithmic MSE loss that emphasizes different components.
    
    Useful if translation and rotation errors should be weighted differently.
    
    For SE(3) outputs (3 translation + 6 rotation components):
    - Translation: typically 0-0.1 (meters)
    - Rotation: typically 0-6 (6D representation)
    
    You might want to weight them differently.
    """
    
    def __init__(self, trans_weight: float = 1.0, rot_weight: float = 1.0, base: float = 1.0):
        """
        Args:
            trans_weight: Weight for first 3 components (translation)
            rot_weight: Weight for last 6 components (rotation)
            base: Scaling factor for log compression
        """
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.base = base
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values (B, 9) where first 3 are translation, last 6 are rotation
            target: Target values (B, 9)
        
        Returns:
            Scalar loss value
        """
        # Split into translation and rotation
        pred_trans = pred[:, :3]
        pred_rot = pred[:, 3:]
        target_trans = target[:, :3]
        target_rot = target[:, 3:]
        
        # Compute MSE for each component
        mse_trans = torch.mean((pred_trans - target_trans) ** 2)
        mse_rot = torch.mean((pred_rot - target_rot) ** 2)
        
        # Apply weighting and logarithmic compression
        weighted_mse = (self.trans_weight * mse_trans + 
                       self.rot_weight * mse_rot) / (self.trans_weight + self.rot_weight)
        
        log_loss = torch.log(1.0 + self.base * weighted_mse)
        return log_loss


class SmoothL1LogLoss(nn.Module):
    """
    Hybrid loss combining Smooth L1 (Huber) with logarithmic compression.
    
    Smooth L1 is robust to outliers and combines best of L1 and L2.
    Adding log makes small errors have bigger impact.
    """
    
    def __init__(self, beta: float = 1.0, base: float = 1.0):
        """
        Args:
            beta: Threshold for Smooth L1 (default: 1.0)
            base: Log compression scaling
        """
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.base = base
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values
            target: Target values
        
        Returns:
            Scalar loss value
        """
        smooth_l1_loss = self.smooth_l1(pred, target)
        log_loss = torch.log(1.0 + self.base * smooth_l1_loss)
        return log_loss
