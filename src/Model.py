import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and skip connections."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important regions."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction using different kernel sizes."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch4 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = F.relu(self.bn(out))
        return out


class EnhancedQNet(nn.Module):
    """
    Enhanced Q-network with residual connections, attention mechanisms, and multi-scale features.
    Input channels = 8: [slope, aspect, canopy_cover, fuel_model, fireline_north, fireline_east, fireline_south, fireline_west]
    Output: Q-map of shape (H, W), then flattened to (H*W) predictions per batch.
    """

    def __init__(self, H, W, use_attention=True, use_residual=True, use_multiscale=True):
        super().__init__()
        self.H, self.W = H, W
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_multiscale = use_multiscale
        
        # Initial feature extraction
        if use_multiscale:
            self.initial_conv = MultiScaleFeatureExtractor(8, 64)
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(8, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        
        # Feature extraction backbone
        if use_residual:
            self.backbone = nn.Sequential(
                ResidualBlock(64, 128),
                ResidualBlock(128, 128),
                ResidualBlock(128, 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, 128),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        
        # Attention mechanisms
        if use_attention:
            self.channel_attention = ChannelAttention(128)
            self.spatial_attention = SpatialAttention(128)
        
        # Final feature processing
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Q-value head: 1x1 conv to produce one Q per cell
        self.q_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B,8,H,W)
        returns: qvals_flat (B, H*W)
        """
        # Initial feature extraction
        features = self.initial_conv(x)  # (B,64,H,W)
        
        # Backbone feature extraction
        features = self.backbone(features)  # (B,128,H,W)
        
        # Apply attention mechanisms
        if self.use_attention:
            features = self.channel_attention(features)
            features = self.spatial_attention(features)
        
        # Final feature processing
        features = self.final_conv(features)  # (B,32,H,W)
        
        # Generate Q-values
        q_map = self.q_head(features).squeeze(1)  # (B,H,W)
        q_flat = q_map.view(-1, self.H * self.W)
        
        return q_flat


class QNet(nn.Module):
    """
    Original fully-convolutional Q-network for backward compatibility.
    Input channels = 8: [slope, aspect, canopy_cover, fuel_model, fireline_north, fireline_east, fireline_south, fireline_west]
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


class DuelingQNet(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    """
    
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        # Value stream
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_head = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        """
        x: (B,8,H,W)
        returns: qvals_flat (B, H*W)
        """
        features = self.feature_extractor(x)  # (B,64,H,W)
        
        # Value stream
        value = self.value_head(features)  # (B,1)
        
        # Advantage stream
        advantage_map = self.advantage_head(features).squeeze(1)  # (B,H,W)
        advantage_flat = advantage_map.view(-1, self.H * self.W)  # (B,H*W)
        
        # Combine value and advantage
        advantage_mean = advantage_flat.mean(dim=1, keepdim=True)  # (B,1)
        q_flat = value + advantage_flat - advantage_mean  # (B,H*W)
        
        return q_flat
