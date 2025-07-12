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

class DQNNetwork(nn.Module):
    """
    State-of-the-art CNN architecture for fuel-break placement prediction.
    
    This network takes landscape data as input and predicts Q-values for 
    each possible fuel-break location. Uses residual connections and 
    attention mechanisms for improved performance.
    """
    
    def __init__(self, input_channels=12, grid_size=50, action_dim=None):
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
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        # Only detach if necessary, avoid unnecessary CPU transfers
        if hasattr(state, 'detach'):
            state = state.detach()
        if hasattr(next_state, 'detach'):
            next_state = next_state.detach()
            
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        # Prepare lists for batch creation
        states_list = []
        next_states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        for e in experiences:
            state = e.state
            next_state = e.next_state
            
            # If state has batch dimension, remove it
            if state.dim() == 4 and state.size(0) == 1:
                state = state.squeeze(0)
            if next_state.dim() == 4 and next_state.size(0) == 1:
                next_state = next_state.squeeze(0)
                
            states_list.append(state)
            next_states_list.append(next_state)
            actions_list.append(e.action)
            rewards_list.append(e.reward)
            dones_list.append(e.done)
        
        # Stack tensors efficiently
        states = torch.stack(states_list)
        next_states = torch.stack(next_states_list)
        actions = torch.tensor(actions_list, dtype=torch.long)
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        dones = torch.tensor(dones_list, dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer and free memory."""
        self.buffer.clear()
        gc.collect()


class DQNAgent:
    """
    Deep Q-Network agent for learning fuel-break placement strategies.
    
    This agent uses a state-of-the-art CNN to learn optimal fuel-break
    placement by interacting with fire simulation environments and 
    learning from domirank-based expert demonstrations.
    """
    
    def __init__(self, input_channels=12, grid_size=50, 
                 learning_rate=1e-4, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=32,
                 max_history_size=1000, cleanup_frequency=100):
        
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
        
        print(f"DQN Agent device: {self.device}")
        
        self.grid_size = grid_size
        self.action_dim = grid_size * grid_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.input_channels = input_channels
        
        # Initialize networks
        self.q_network = DQNNetwork(input_channels, grid_size).to(self.device)
        self.target_network = DQNNetwork(input_channels, grid_size).to(self.device)
        
        # Print network info
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(f"Network parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics - LIMITED SIZE to prevent memory leaks
        self.max_history_size = max_history_size
        self.losses = deque(maxlen=self.max_history_size)
        self.rewards = deque(maxlen=self.max_history_size)
        
        # Memory management
        self.cleanup_frequency = cleanup_frequency
        self.training_steps = 0
        
        # Performance optimizations - cache common tensors and pre-warm network
        self._initialize_performance_optimizations()
        
    def _initialize_performance_optimizations(self):
        """Initialize performance optimizations to eliminate early-episode overhead."""
        print("🚀 Initializing performance optimizations...")
        
        # Pre-allocate commonly used tensors
        self._dummy_state = torch.zeros(1, self.input_channels, self.grid_size, self.grid_size, device=self.device)
        self._available_positions_cache = np.arange(self.action_dim)
        
        # Pre-warm PyTorch operations with dummy forward passes
        print("   - Pre-warming neural networks...")
        self.q_network.eval()
        with torch.no_grad():
            # Multiple warmup passes to trigger JIT compilation and CUDA optimizations
            for _ in range(10):
                _ = self.q_network(self._dummy_state)
                _ = self.target_network(self._dummy_state)
        self.q_network.train()
        
        # Pre-warm other operations
        dummy_batch = self._dummy_state.repeat(self.batch_size, 1, 1, 1)
        with torch.no_grad():
            _ = self.q_network(dummy_batch)
            
        # Clear any temporary memory from warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("   ✅ Performance optimizations complete!")
    
    def preprocess_state(self, landscape_data):
        """
        Preprocess landscape data for input to the network.
        
        Args:
            landscape_data: Dictionary containing landscape arrays
            
        Returns:
            Preprocessed tensor ready for network input
        """
        # Stack all landscape layers including fireline intensities
        layers = []
        layer_names = ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm', 
                      'fireline_north', 'fireline_east', 'fireline_south', 'fireline_west']
        
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
        Choose action using epsilon-greedy policy with optimizations.
        
        Args:
            state: Current state (landscape data)
            existing_fuel_breaks: Current fuel break locations to avoid duplicates
            
        Returns:
            Selected action (fuel break location)
        """
        if random.random() > self.epsilon:
            # Exploit: use Q-network to select action (fast path after warmup)
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # Optimized masking for existing fuel breaks
                if existing_fuel_breaks is not None:
                    flat_breaks = existing_fuel_breaks.flatten()
                    if np.any(flat_breaks):  # Only mask if there are existing breaks
                        q_values = q_values.clone()
                        q_values[0, flat_breaks == 1] = -float('inf')
                
                action = q_values.argmax().item()
        else:
            # Explore: optimized random action selection
            if existing_fuel_breaks is not None:
                flat_breaks = existing_fuel_breaks.flatten()
                available_mask = flat_breaks == 0
                if np.any(available_mask):
                    # Use pre-allocated array and boolean indexing for speed
                    available_positions = self._available_positions_cache[available_mask]
                    action = np.random.choice(available_positions)
                else:
                    action = np.random.randint(0, self.action_dim)
            else:
                action = np.random.randint(0, self.action_dim)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Remove batch dimension if present when storing
        if state.dim() == 4 and state.size(0) == 1:
            state = state.squeeze(0)
        if next_state.dim() == 4 and next_state.size(0) == 1:
            next_state = next_state.squeeze(0)
        
        # Keep tensors on GPU, just detach from computation graph
        # Moving to CPU every time is expensive and unnecessary
        self.memory.push(state, action, reward, next_state, done)
    
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
        
        # Store loss for monitoring (limited size)
        self.losses.append(loss.item())
        
        # Increment training steps counter
        self.training_steps += 1
        
        # Periodic memory cleanup
        if self.training_steps % self.cleanup_frequency == 0:
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """Clean up GPU memory to prevent memory leaks."""
        # Only do expensive cleanup when really needed
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
            torch.cuda.empty_cache()
        # Avoid frequent garbage collection which is expensive
        if self.training_steps % 5000 == 0:  # Only every 5000 steps
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
            'losses': list(self.losses),
            'rewards': list(self.rewards)
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = deque(checkpoint['losses'], maxlen=self.max_history_size)
        self.rewards = deque(checkpoint['rewards'], maxlen=self.max_history_size)
        print(f"Model loaded from {filepath}")
    
    def reset_memory(self):
        """Reset replay buffer and training metrics."""
        self.memory.clear()
        self.losses.clear()
        self.rewards.clear()
        self.cleanup_memory()
        print("Agent memory reset")