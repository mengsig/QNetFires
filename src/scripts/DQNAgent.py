import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import copy

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """
    State-of-the-art CNN architecture for fuel-break placement prediction.
    
    This network takes landscape data as input and predicts Q-values for 
    each possible fuel-break location. Uses residual connections and 
    attention mechanisms for improved performance.
    """
    
    def __init__(self, input_channels=8, grid_size=50, action_dim=None):
        super(DQNNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.grid_size = grid_size
        self.action_dim = action_dim or (grid_size * grid_size)
        
        # Initial convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(128, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 256)
        
        # Attention mechanism
        self.attention = SpatialAttention(256)
        
        # Feature extraction layers
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Final layers for Q-value prediction
        self.conv_final = nn.Conv2d(256, 128, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(128)
        
        # Output layer - produces Q-value for each spatial location
        self.q_value_head = nn.Conv2d(128, 1, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.2)
        
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
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each spatial location, shape (batch_size, height*width)
        """
        # Initial convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Feature extraction
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Final layers
        x = F.relu(self.bn_final(self.conv_final(x)))
        q_values = self.q_value_head(x)
        
        # Reshape to (batch_size, height*width)
        batch_size = q_values.size(0)
        q_values = q_values.view(batch_size, -1)
        
        return q_values


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                   nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important landscape features."""
    
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        
        # Apply attention
        return x * attention


class ReplayBuffer:
    """Experience replay buffer for storing and sampling training experiences."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for learning fuel-break placement strategies.
    
    This agent uses a state-of-the-art CNN to learn optimal fuel-break
    placement by interacting with fire simulation environments and 
    learning from domirank-based expert demonstrations.
    """
    
    def __init__(self, input_channels=8, grid_size=50, 
                 learning_rate=1e-4, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.grid_size = grid_size
        self.action_dim = grid_size * grid_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize networks
        self.q_network = DQNNetwork(input_channels, grid_size).to(self.device)
        self.target_network = DQNNetwork(input_channels, grid_size).to(self.device)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        
    def preprocess_state(self, landscape_data):
        """
        Preprocess landscape data for input to the network.
        
        Args:
            landscape_data: Dictionary containing landscape arrays
            
        Returns:
            Preprocessed tensor ready for network input
        """
        # Stack all landscape layers
        layers = []
        layer_names = ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']
        
        for name in layer_names:
            if name in landscape_data:
                layer = landscape_data[name]
                if isinstance(layer, np.ndarray):
                    layer = torch.from_numpy(layer).float()
                layers.append(layer)
        
        # Stack and add batch dimension
        state = torch.stack(layers, dim=0).unsqueeze(0)  # (1, channels, height, width)
        return state.to(self.device)
    
    def act(self, state, existing_fuel_breaks=None):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state (landscape data)
            existing_fuel_breaks: Current fuel break locations to avoid duplicates
            
        Returns:
            Selected action (fuel break location)
        """
        if random.random() > self.epsilon:
            # Exploit: use Q-network to select action
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # Mask out existing fuel breaks
                if existing_fuel_breaks is not None:
                    q_values = q_values.clone()
                    flat_breaks = existing_fuel_breaks.flatten()
                    q_values[0, flat_breaks == 1] = -float('inf')
                
                action = q_values.argmax().item()
        else:
            # Explore: select random action
            if existing_fuel_breaks is not None:
                # Select from available positions only
                available_positions = np.where(existing_fuel_breaks.flatten() == 0)[0]
                if len(available_positions) > 0:
                    action = np.random.choice(available_positions)
                else:
                    action = np.random.randint(0, self.action_dim)
            else:
                action = np.random.randint(0, self.action_dim)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state.cpu(), action, reward, next_state.cpu(), done)
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss for monitoring
        self.losses.append(loss.item())
        
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'rewards': self.rewards
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.rewards = checkpoint['rewards']
        print(f"Model loaded from {filepath}")