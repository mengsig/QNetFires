import torch
import torch.nn as nn


class QNet(nn.Module):
    """
    Fully-convolutional Q-network: input channels = 8
      [slope, aspect, canopy_cover, fuel_model, fireline_north, fireline_east, fireline_south, fireline_west]
    Output: Q-map of shape (H, W), then flattened to (H*W) predictions per batch.
    """

    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        # convolutional trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        # Q-value head: 1x1 conv to produce one Q per cell
        self.q_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B,8,H,W)
        returns: qvals_flat (B, H*W)
        """
        feats = self.trunk(x)  # (B,64,H,W)
        q_map = self.q_head(feats).squeeze(1)  # (B,H,W)
        q_flat = q_map.view(-1, self.H * self.W)
        return q_flat
