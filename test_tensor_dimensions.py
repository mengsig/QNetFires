#!/usr/bin/env python3
"""
Test script to verify tensor dimensions in the DQN pipeline
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'scripts'))

from src.scripts.DQNAgent import DQNAgent, ReplayBuffer
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader

def test_tensor_dimensions():
    """Test that tensor dimensions are handled correctly throughout the pipeline."""
    print("Testing tensor dimensions in DQN pipeline...")
    
    # Create a small test setup
    grid_size = 10
    input_channels = 8
    
    # Initialize components
    agent = DQNAgent(input_channels=input_channels, grid_size=grid_size, batch_size=4)
    memory_loader = DomiRankMemoryLoader(grid_size=grid_size)
    
    # Generate a small sample landscape
    memory_loader.generate_sample_data(1)
    landscape_data = memory_loader.load_landscape_data(0)
    
    print("1. Testing preprocess_state...")
    state_tensor = agent.preprocess_state(landscape_data)
    print(f"   State tensor shape: {state_tensor.shape}")
    print(f"   Expected: [1, {input_channels}, {grid_size}, {grid_size}]")
    assert state_tensor.shape == (1, input_channels, grid_size, grid_size), f"Wrong state shape: {state_tensor.shape}"
    
    print("2. Testing network forward pass...")
    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Expected: [1, {grid_size * grid_size}]")
        assert q_values.shape == (1, grid_size * grid_size), f"Wrong Q-values shape: {q_values.shape}"
    
    print("3. Testing memory storage...")
    action = 42
    reward = -100.0
    next_state = state_tensor.clone()
    done = False
    
    agent.remember(state_tensor, action, reward, next_state, done)
    print(f"   Memory buffer size: {len(agent.memory)}")
    
    # Add a few more memories
    for i in range(5):
        agent.remember(state_tensor, action + i, reward - i * 10, next_state, done)
    
    print(f"   Memory buffer size after adding more: {len(agent.memory)}")
    
    print("4. Testing batch sampling...")
    if len(agent.memory) >= agent.batch_size:
        states, actions, rewards, next_states, dones = agent.memory.sample(agent.batch_size)
        
        print(f"   Sampled states shape: {states.shape}")
        print(f"   Expected: [{agent.batch_size}, {input_channels}, {grid_size}, {grid_size}]")
        assert states.shape == (agent.batch_size, input_channels, grid_size, grid_size), f"Wrong sampled states shape: {states.shape}"
        
        print(f"   Sampled actions shape: {actions.shape}")
        print(f"   Sampled rewards shape: {rewards.shape}")
        print(f"   Sampled next_states shape: {next_states.shape}")
        print(f"   Sampled dones shape: {dones.shape}")
        
        print("5. Testing network with batched input...")
        with torch.no_grad():
            batch_q_values = agent.q_network(states)
            print(f"   Batch Q-values shape: {batch_q_values.shape}")
            print(f"   Expected: [{agent.batch_size}, {grid_size * grid_size}]")
            assert batch_q_values.shape == (agent.batch_size, grid_size * grid_size), f"Wrong batch Q-values shape: {batch_q_values.shape}"
        
        print("6. Testing replay training...")
        try:
            agent.replay()
            print("   Replay training successful!")
        except Exception as e:
            print(f"   Replay training failed: {e}")
            raise
    else:
        print("   Not enough memories for batch sampling")
    
    print("\n✅ All tensor dimension tests passed!")

if __name__ == "__main__":
    test_tensor_dimensions()