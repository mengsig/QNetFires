#!/usr/bin/env python3
"""
Demo script for Deep Q-Learning Fuel Break Placement

This script demonstrates the Deep Q-learning approach for fuel break placement
using a simplified configuration suitable for testing and demonstration.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, 'src', 'scripts'))

from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.DQNAgent import DQNAgent


def test_domirank_memory_loader():
    """Test the DomiRank memory loader functionality."""
    print("Testing DomiRank Memory Loader...")
    
    # Initialize memory loader
    memory_loader = DomiRankMemoryLoader(grid_size=25)  # Smaller grid for demo
    
    # Generate sample data
    memory_loader.generate_sample_data(num_landscapes=2)
    
    # Load landscape data
    landscape_data = memory_loader.load_landscape_data(0)
    print(f"Loaded landscape data with keys: {landscape_data.keys()}")
    
    # Load domirank values
    domirank_values = memory_loader.load_domirank_values(0)
    print(f"Domirank values shape: {domirank_values.shape}")
    
    # Create iterative fuel breaks
    percentages = [1, 5, 10, 15, 20]
    fuel_breaks = memory_loader.create_iterative_fuel_breaks(domirank_values, percentages)
    
    print(f"Created fuel breaks for percentages: {list(fuel_breaks.keys())}")
    for pct, breaks in fuel_breaks.items():
        num_breaks = np.sum(breaks)
        total_cells = breaks.size
        actual_pct = (num_breaks / total_cells) * 100
        print(f"  {pct}%: {num_breaks} breaks ({actual_pct:.1f}% actual)")
    
    # Visualize fuel breaks
    try:
        memory_loader.visualize_fuel_breaks(0, percentages)
        print("Visualization created successfully!")
    except Exception as e:
        print(f"Visualization failed (expected in headless environment): {e}")
    
    return memory_loader, landscape_data, domirank_values


def test_dqn_agent():
    """Test the DQN agent functionality."""
    print("\nTesting DQN Agent...")
    
    # Initialize agent with smaller network for demo
    agent = DQNAgent(
        input_channels=8,
        grid_size=25,
        learning_rate=1e-3,
        buffer_size=1000,
        batch_size=8
    )
    
    print(f"Agent initialized with device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Test preprocessing
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(1)
    landscape_data = memory_loader.load_landscape_data(0)
    
    state_tensor = agent.preprocess_state(landscape_data)
    print(f"Preprocessed state shape: {state_tensor.shape}")
    
    # Test action selection
    action = agent.act(state_tensor)
    print(f"Selected action: {action}")
    
    # Test memory storage
    reward = -100.0
    done = False
    agent.remember(state_tensor, action, reward, state_tensor, done)
    print(f"Memory buffer size: {len(agent.memory)}")
    
    return agent


def test_simple_training_loop():
    """Test a simple training loop."""
    print("\nTesting Simple Training Loop...")
    
    # Initialize components
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(2)
    
    agent = DQNAgent(
        input_channels=8,
        grid_size=25,
        learning_rate=1e-3,
        buffer_size=1000,
        batch_size=8,
        epsilon=0.5  # Some exploration
    )
    
    # Generate some training memories
    print("Generating training memories...")
    memories = memory_loader.generate_training_memories(
        landscape_indices=[0, 1],
        percentages=[1, 5, 10],
        num_simulations=2  # Reduced for speed
    )
    
    # Add memories to agent
    for memory in memories:
        agent.remember(
            state=memory['state'],
            action=memory['action'],
            reward=memory['reward'],
            next_state=memory['next_state'],
            done=memory['done']
        )
    
    print(f"Added {len(memories)} memories to replay buffer")
    
    # Simple training loop
    print("Running training steps...")
    initial_epsilon = agent.epsilon
    
    for step in range(20):  # Just a few steps for demo
        if len(agent.memory) >= agent.batch_size:
            agent.replay()
        
        if step % 5 == 0:
            print(f"Step {step}: Epsilon = {agent.epsilon:.3f}, "
                  f"Losses = {len(agent.losses)}")
    
    print(f"Training completed. Epsilon changed from {initial_epsilon:.3f} to {agent.epsilon:.3f}")
    print(f"Total losses recorded: {len(agent.losses)}")
    
    if agent.losses:
        avg_loss = np.mean(agent.losses[-10:])  # Last 10 losses
        print(f"Average loss (last 10 steps): {avg_loss:.4f}")
    
    return agent


def create_demo_visualization():
    """Create a demo visualization showing the fuel break placement process."""
    print("\nCreating Demo Visualization...")
    
    # Initialize memory loader
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(1)
    
    # Load data
    landscape_data = memory_loader.load_landscape_data(0)
    domirank_values = memory_loader.load_domirank_values(0)
    
    # Create fuel breaks at different percentages
    percentages = [1, 5, 10, 15, 20]
    fuel_breaks = memory_loader.create_iterative_fuel_breaks(domirank_values, percentages)
    
    # Create visualization
    fig, axes = plt.subplots(2, len(percentages), figsize=(15, 6))
    
    for i, pct in enumerate(percentages):
        # Top row: landscape features
        im1 = axes[0, i].imshow(landscape_data['slp'], cmap='terrain')
        axes[0, i].set_title(f'Slope')
        axes[0, i].axis('off')
        
        # Bottom row: fuel breaks
        im2 = axes[1, i].imshow(landscape_data['slp'], cmap='terrain', alpha=0.6)
        fuel_break_overlay = axes[1, i].imshow(fuel_breaks[pct], cmap='Reds', alpha=0.8)
        axes[1, i].set_title(f'{pct}% Fuel Breaks')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'demo_fuel_breaks_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Demo visualization saved as {output_file}")
    
    # Create a simple metrics plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Simulate some training metrics
    episodes = np.arange(1, 51)
    rewards = -1000 + 200 * np.exp(episodes / 25) + 50 * np.random.randn(50)
    
    ax.plot(episodes, rewards, 'b-', alpha=0.6, label='Episode Rewards')
    
    # Add moving average
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Demo Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_file = 'demo_training_metrics.png'
    plt.savefig(metrics_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Demo training metrics saved as {metrics_file}")


def main():
    """Main demo function."""
    print("=" * 60)
    print("Deep Q-Learning Fuel Break Placement - Demo")
    print("=" * 60)
    
    try:
        # Test individual components
        memory_loader, landscape_data, domirank_values = test_domirank_memory_loader()
        agent = test_dqn_agent()
        trained_agent = test_simple_training_loop()
        
        # Create visualizations
        create_demo_visualization()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        print("\nKey components tested:")
        print("✓ DomiRank memory loader")
        print("✓ DQN agent initialization")
        print("✓ Landscape data generation")
        print("✓ Iterative fuel break creation")
        print("✓ Experience replay memory")
        print("✓ Simple training loop")
        print("✓ Visualization generation")
        
        print(f"\nNetwork architecture summary:")
        print(f"- Input channels: {agent.q_network.input_channels}")
        print(f"- Grid size: {agent.q_network.grid_size}")
        print(f"- Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
        print(f"- Device: {agent.device}")
        
        print(f"\nOutput files created:")
        print("- demo_fuel_breaks_visualization.png")
        print("- demo_training_metrics.png")
        
        print(f"\nTo run full training, use:")
        print("python src/scripts/train_dqn_fuel_breaks.py --config demo_config.json")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)