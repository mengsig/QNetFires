import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import copy
import gc

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 
                         'action', 
                         'reward', 
                         'next_state',
                         'done'])

class OptimizedDQNNetwork(nn.Module):
    """
    Optimized CNN architecture for fuel-break placement prediction.
    
    This network is optimized for better performance with reduced model size
    while maintaining effectiveness. Hidden layers are reduced by factor of 2.
    """
    
    def __init__(self, input_channels=12, grid_size=50, action_dim=None):
        super(OptimizedDQNNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.grid_size = grid_size
        self.action_dim = action_dim or (grid_size * grid_size)
        
        # Optimized convolutional layers (reduced by factor of 2)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Simplified residual blocks
        self.res_block1 = OptimizedResidualBlock(64, 64)
        self.res_block2 = OptimizedResidualBlock(64, 128)
        
        # Simplified attention mechanism
        self.attention = OptimizedSpatialAttention(128)
        
        # Feature extraction layers (reduced)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Final layers for Q-value prediction
        self.conv_final = nn.Conv2d(128, 64, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(64)
        
        # Output layer - produces Q-value for each spatial location
        self.q_value_head = nn.Conv2d(64, 1, kernel_size=1)
        
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


class OptimizedResidualBlock(nn.Module):
    """Optimized residual block with reduced parameters."""
    
    def __init__(self, in_channels, out_channels):
        super(OptimizedResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Identity() if in_channels == out_channels else \
                              nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class OptimizedSpatialAttention(nn.Module):
    """Optimized spatial attention mechanism."""
    
    def __init__(self, channels):
        super(OptimizedSpatialAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        return x * attention


class OptimizedReplayBuffer:
    """Optimized replay buffer with proper memory management."""
    
    def __init__(self, capacity=50000):  # Reduced from 100000
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with proper cleanup."""
        # Ensure tensors are detached from computation graph
        if torch.is_tensor(state):
            state = state.detach().cpu()
        if torch.is_tensor(next_state):
            next_state = next_state.detach().cpu()
        if torch.is_tensor(action):
            action = action.detach().cpu()
        if torch.is_tensor(reward):
            reward = reward.detach().cpu()
        if torch.is_tensor(done):
            done = done.detach().cpu()
            
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.stack([torch.FloatTensor(e.state) for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.stack([torch.FloatTensor(e.next_state) for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer to free memory."""
        self.buffer.clear()
        gc.collect()


class OptimizedDQNAgent:
    """
    Optimized Deep Q-Network agent with proper memory management and reduced model size.
    """
    
    def __init__(self, input_channels=12, grid_size=50, 
                 learning_rate=1e-4, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=50000, batch_size=32):
        
        # Enhanced GPU detection and setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")
        
        print(f"Optimized DQN Agent device: {self.device}")
        
        self.grid_size = grid_size
        self.action_dim = grid_size * grid_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize optimized networks
        self.q_network = OptimizedDQNNetwork(input_channels, grid_size).to(self.device)
        self.target_network = OptimizedDQNNetwork(input_channels, grid_size).to(self.device)
        
        # Print network info
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(f"Optimized Network parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        # Experience replay buffer
        self.memory = OptimizedReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        
        # Memory management
        self.memory_cleanup_frequency = 100
        self.training_step = 0
        
    def preprocess_state(self, landscape_data):
        """Preprocess landscape data into network input format."""
        # Stack all landscape layers
        state_layers = []
        
        # Add landscape data layers
        for key in ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']:
            if key in landscape_data:
                layer = landscape_data[key]
                if isinstance(layer, np.ndarray):
                    layer = torch.FloatTensor(layer)
                state_layers.append(layer)
        
        # Add fireline intensity data (4 layers for 4 directions)
        for direction in ['north', 'south', 'east', 'west']:
            fireline_key = f'fireline_{direction}'
            if fireline_key in landscape_data:
                layer = landscape_data[fireline_key]
                if isinstance(layer, np.ndarray):
                    layer = torch.FloatTensor(layer)
                state_layers.append(layer)
        
        # Stack all layers
        state = torch.stack(state_layers, dim=0)
        
        return state
    
    def act(self, state, existing_fuel_breaks=None):
        """Choose action using epsilon-greedy policy with proper memory management."""
        if random.random() <= self.epsilon:
            # Random action
            return random.randrange(self.action_dim)
        
        # Ensure state is properly formatted
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state)
            
            # Mask invalid actions (existing fuel breaks)
            if existing_fuel_breaks is not None:
                invalid_mask = torch.FloatTensor(existing_fuel_breaks.flatten()).to(self.device)
                q_values = q_values - 1e6 * invalid_mask
            
            action = q_values.argmax().item()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences with proper memory management."""
        if len(self.memory) < self.batch_size:
            return
        
        self.training_step += 1
        
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
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss for monitoring
        self.losses.append(loss.item())
        
        # Periodic memory cleanup
        if self.training_step % self.memory_cleanup_frequency == 0:
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """Clean up GPU memory and Python garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
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
    
    def __del__(self):
        """Cleanup when agent is destroyed."""
        self.cleanup_memory()