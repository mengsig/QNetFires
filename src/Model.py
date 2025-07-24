import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute spatial attention map
        attention = self.sigmoid(self.conv(x))
        return x * attention


class QNet(nn.Module):
    """
    Advanced Deep Q-Network with residual connections and attention mechanisms.
    Input channels = 8: [slope, aspect, canopy_cover, fuel_model, fireline_north, fireline_east, fireline_south, fireline_west]
    Output: Q-map of shape (H, W), then flattened to (H*W) predictions per batch.
    """

    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for deep feature extraction
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(128, 256)
        self.res_block4 = ResidualBlock(256, 256)
        
        # Spatial attention
        self.attention = SpatialAttention(256)
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature fusion
        self.multiscale_conv1 = nn.Conv2d(64, 32, kernel_size=1, padding=0)  # 1x1 conv
        self.multiscale_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 3x3 conv
        self.multiscale_conv5 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  # 5x5 conv
        
        # Final Q-value prediction head
        self.q_head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 96 = 32*3 from multiscale
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # Final Q-value per cell
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, 8, H, W) - batch of 8-channel landscape observations
        returns: qvals_flat (B, H*W) - Q-values for each spatial location
        """
        # Initial feature extraction
        x = self.initial_conv(x)  # (B, 64, H, W)
        
        # Deep residual feature extraction
        x = self.res_block1(x)    # (B, 128, H, W)
        x = self.res_block2(x)    # (B, 128, H, W)
        x = self.res_block3(x)    # (B, 256, H, W)
        x = self.res_block4(x)    # (B, 256, H, W)
        
        # Apply spatial attention
        x = self.attention(x)     # (B, 256, H, W)
        
        # Feature refinement
        x = self.refinement(x)    # (B, 64, H, W)
        
        # Multi-scale feature fusion
        feat1 = self.multiscale_conv1(x)  # (B, 32, H, W)
        feat3 = self.multiscale_conv3(x)  # (B, 32, H, W)
        feat5 = self.multiscale_conv5(x)  # (B, 32, H, W)
        
        # Concatenate multi-scale features
        multiscale_feats = torch.cat([feat1, feat3, feat5], dim=1)  # (B, 96, H, W)
        
        # Generate Q-values
        q_map = self.q_head(multiscale_feats).squeeze(1)  # (B, H, W)
        
        # Flatten for action selection
        q_flat = q_map.view(-1, self.H * self.W)  # (B, H*W)
        
        return q_flat

    def get_feature_maps(self, x):
        """
        Return intermediate feature maps for visualization/analysis
        """
        features = {}
        
        # Initial features
        x = self.initial_conv(x)
        features['initial'] = x
        
        # Residual features
        x = self.res_block1(x)
        features['res1'] = x
        x = self.res_block2(x)
        features['res2'] = x
        x = self.res_block3(x)
        features['res3'] = x
        x = self.res_block4(x)
        features['res4'] = x
        
        # Attention
        x_att = self.attention(x)
        features['attention'] = x_att
        
        # Refinement
        x = self.refinement(x_att)
        features['refined'] = x
        
        # Final Q-map
        feat1 = self.multiscale_conv1(x)
        feat3 = self.multiscale_conv3(x)
        feat5 = self.multiscale_conv5(x)
        multiscale_feats = torch.cat([feat1, feat3, feat5], dim=1)
        q_map = self.q_head(multiscale_feats).squeeze(1)
        features['q_map'] = q_map
        
        return features


class DuelingQNet(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage estimation
    """
    
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        
        # Shared feature extractor (same as QNet)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            SpatialAttention(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value estimate
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Advantage per spatial location
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (B, 8, H, W)
        returns: Q-values (B, H*W)
        """
        # Extract shared features
        features = self.feature_extractor(x)  # (B, 128, H, W)
        
        # Value stream
        value = self.value_stream(features)  # (B, 1)
        
        # Advantage stream  
        advantages = self.advantage_stream(features).squeeze(1)  # (B, H, W)
        
        # Combine value and advantages using dueling formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        advantages_flat = advantages.view(-1, self.H * self.W)  # (B, H*W)
        advantages_mean = advantages_flat.mean(dim=1, keepdim=True)  # (B, 1)
        
        q_values = value + advantages_flat - advantages_mean  # (B, H*W)
        
        return q_values
