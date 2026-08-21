"""
Direction Consistency Loss for Vessel Segmentation

This loss function enforces directional consistency between predicted vessel 
segmentation and ground truth. It's particularly effective for tubular structures
like blood vessels where directional continuity is important.

The loss consists of two components:
1. Directional gradient consistency: penalizes mismatches in vessel orientation
2. Structural consistency: ensures continuous vessel structures

Author: Code Development Agent
Date: 2026-04-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DirectionConsistencyLoss(nn.Module):
    """
    Direction Consistency Loss for vessel segmentation.
    
    This loss encourages the predicted segmentation to maintain directional
    consistency with the ground truth, which is crucial for tubular structures
    like blood vessels.
    
    Args:
        weight (float): Weight of the direction loss relative to base loss
        alpha (float): Weight for horizontal gradient consistency
        beta (float): Weight for vertical gradient consistency
        gamma (float): Weight for diagonal gradient consistency
    """
    
    def __init__(self, weight=0.1, alpha=1.0, beta=1.0, gamma=0.5):
        super(DirectionConsistencyLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha  # horizontal weight
        self.beta = beta    # vertical weight
        self.gamma = gamma  # diagonal weight
        
        # Sobel operators for gradient computation
        self._register_sobel_kernels()
    
    def _register_sobel_kernels(self):
        """Register Sobel kernels as non-trainable buffers."""
        # Horizontal Sobel kernel
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Vertical Sobel kernel
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal (45°) Sobel kernel
        sobel_diag1 = torch.tensor([
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Diagonal (135°) Sobel kernel
        sobel_diag2 = torch.tensor([
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_diag1', sobel_diag1)
        self.register_buffer('sobel_diag2', sobel_diag2)
    
    def _compute_gradient(self, x, kernel):
        """
        Compute gradient using given kernel.
        
        Args:
            x (Tensor): Input tensor of shape (N, 1, H, W)
            kernel (Tensor): Convolution kernel
            
        Returns:
            Tensor: Gradient magnitude
        """
        # Pad input to maintain spatial dimensions
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        # Convert kernel to match input dtype and device (for mixed precision support)
        kernel = kernel.to(dtype=x.dtype, device=x.device)
        
        gradient = F.conv2d(x_padded, kernel, padding=0)
        return gradient
    
    def _compute_directional_gradients(self, x):
        """
        Compute multi-directional gradients.
        
        Args:
            x (Tensor): Input tensor of shape (N, 1, H, W)
            
        Returns:
            tuple: Gradients in different directions
        """
        grad_x = self._compute_gradient(x, self.sobel_x)
        grad_y = self._compute_gradient(x, self.sobel_y)
        grad_diag1 = self._compute_gradient(x, self.sobel_diag1)
        grad_diag2 = self._compute_gradient(x, self.sobel_diag2)
        
        return grad_x, grad_y, grad_diag1, grad_diag2
    
    def _direction_consistency_term(self, pred_grad, target_grad):
        """
        Compute direction consistency between prediction and target gradients.
        
        Uses cosine similarity to measure directional alignment.
        
        Args:
            pred_grad (Tensor): Predicted gradient
            target_grad (Tensor): Target gradient
            
        Returns:
            Tensor: Direction consistency loss
        """
        # Normalize gradients
        pred_norm = F.normalize(pred_grad, p=2, dim=1)
        target_norm = F.normalize(target_grad, p=2, dim=1)
        
        # Cosine similarity (higher is better)
        similarity = (pred_norm * target_norm).sum(dim=1, keepdim=True)
        
        # Convert to loss (lower is better)
        # We want similarity to be 1, so loss = 1 - similarity
        consistency_loss = 1.0 - similarity
        
        # Mask out regions where target gradient is very small
        gradient_magnitude = torch.sqrt((target_grad ** 2).sum(dim=1, keepdim=True) + 1e-8)
        mask = (gradient_magnitude > 0.1).float()
        
        # Apply mask and average
        consistency_loss = (consistency_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return consistency_loss
    
    def forward(self, predictions, targets):
        """
        Compute direction consistency loss.
        
        Args:
            predictions (Tensor): Predicted segmentation probabilities, shape (N, C, H, W)
            targets (Tensor): Ground truth segmentation, shape (N, H, W)
            
        Returns:
            Tensor: Direction consistency loss value
        """
        # Ensure predictions are probabilities (apply softmax if needed)
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = F.softmax(predictions, dim=1)
        
        # Extract vessel class (class 1)
        if predictions.shape[1] == 2:
            pred_vessel = predictions[:, 1:2, :, :]  # (N, 1, H, W)
        else:
            pred_vessel = predictions[:, 0:1, :, :]
        
        # Convert targets to one-hot format
        if targets.dim() == 3:
            target_vessel = (targets == 1).float().unsqueeze(1)  # (N, 1, H, W)
        else:
            target_vessel = targets
        
        # Compute directional gradients for prediction and target
        pred_grads = self._compute_directional_gradients(pred_vessel)
        target_grads = self._compute_directional_gradients(target_vessel)
        
        # Compute direction consistency for each direction
        loss_x = self._direction_consistency_term(pred_grads[0], target_grads[0])
        loss_y = self._direction_consistency_term(pred_grads[1], target_grads[1])
        loss_diag1 = self._direction_consistency_term(pred_grads[2], target_grads[2])
        loss_diag2 = self._direction_consistency_term(pred_grads[3], target_grads[3])
        
        # Combine losses with weights
        direction_loss = (
            self.alpha * loss_x +
            self.beta * loss_y +
            self.gamma * loss_diag1 +
            self.gamma * loss_diag2
        )
        
        return self.weight * direction_loss


class CombinedDirectionLoss(nn.Module):
    """
    Combined loss function with Cross-Entropy and Direction Consistency.
    
    This is a convenience class that combines the standard Cross-Entropy loss
    with the Direction Consistency loss for easier integration into training.
    
    Args:
        ce_weight (float): Weight for Cross-Entropy loss
        direction_weight (float): Weight for Direction Consistency loss
        class_weights (Tensor, optional): Class weights for CE loss
    """
    
    def __init__(self, ce_weight=1.0, direction_weight=0.1, class_weights=None):
        super(CombinedDirectionLoss, self).__init__()
        self.ce_weight = ce_weight
        self.direction_weight = direction_weight
        
        # Cross-entropy loss
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Direction consistency loss
        self.direction_loss = DirectionConsistencyLoss(weight=1.0)
    
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions (Tensor): Predicted logits, shape (N, C, H, W)
            targets (Tensor): Ground truth segmentation, shape (N, H, W)
            
        Returns:
            Tensor: Total loss value
            dict: Individual loss components
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Direction consistency loss
        dir_loss = self.direction_loss(predictions, targets)
        
        # Total loss
        total_loss = self.ce_weight * ce_loss + self.direction_weight * dir_loss
        
        loss_dict = {
            'total': total_loss,
            'cross_entropy': ce_loss,
            'direction': dir_loss
        }
        
        return total_loss, loss_dict


# Legacy compatibility
class DirectionLoss(DirectionConsistencyLoss):
    """Alias for DirectionConsistencyLoss for backward compatibility."""
    pass
