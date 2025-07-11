#!/usr/bin/env python3
"""
Deep Q-Learning Training Script for Fuel Break Placement

This script implements an iterative Deep Q-learning approach to learn optimal
fuel break placement strategies. It uses domirank-based expert demonstrations
for initialization and learns to place fuel breaks iteratively from 1% to 20%
of the landscape.

Key Features:
- State-of-the-art CNN architecture with residual connections and attention
- Domirank-based experience replay initialization
- Iterative fuel break placement learning
- Integration with fire simulation for reward evaluation
- Comprehensive training monitoring and visualization
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.DQNAgent import DQNAgent
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.FireEnv import FireEnv
from src.scripts.Simulate import Simulate


class IterativeFuelBreakTrainer:
    """
    Main trainer class for iterative fuel break placement using Deep Q-learning.
    
    This trainer implements a progressive learning approach where the agent
    learns to place fuel breaks iteratively, starting from 1% and going up
    to 20% of the landscape coverage.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.memory_loader = DomiRankMemoryLoader(
            raster_dir=config['raster_dir'],
            grid_size=config['grid_size']
        )
        
        self.agent = DQNAgent(
            input_channels=config['input_channels'],
            grid_size=config['grid_size'],
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            epsilon=config['initial_epsilon'],
            epsilon_min=config['epsilon_min'],
            epsilon_decay=config['epsilon_decay'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'fuel_break_percentages': [],
            'best_reward': -float('inf'),
            'episodes_completed': 0
        }
        
        # Create output directories
        self.setup_output_directories()
        
    def setup_output_directories(self):
        """Create directories for outputs."""
        self.output_dir = self.config['output_dir']
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.memories_dir = os.path.join(self.output_dir, 'memories')
        
        for directory in [self.output_dir, self.models_dir, self.plots_dir, self.memories_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_with_domirank_memories(self):
        """
        Initialize the agent's experience replay buffer with domirank-based memories.
        """
        print("Initializing agent with domirank-based memories...")
        
        # Generate or load memories
        memory_file = os.path.join(self.memories_dir, 'domirank_memories.pt')
        
        if os.path.exists(memory_file) and not self.config['regenerate_memories']:
            print(f"Loading existing memories from {memory_file}")
            memories = self.memory_loader.load_memories(memory_file)
        else:
            print("Generating new domirank memories...")
            # Generate sample data if needed
            if not os.path.exists(self.config['raster_dir']):
                self.memory_loader.generate_sample_data(self.config['num_landscapes'])
            
            # Generate training memories
            landscape_indices = list(range(self.config['num_landscapes']))
            percentages = list(range(1, 21))  # 1% to 20%
            
            memories = self.memory_loader.generate_training_memories(
                landscape_indices=landscape_indices,
                percentages=percentages,
                num_simulations=self.config['memory_simulations']
            )
            
            # Save memories
            self.memory_loader.save_memories(memories, memory_file)
        
        # Add memories to agent's replay buffer
        print("Adding memories to agent's replay buffer...")
        for memory in tqdm(memories, desc="Loading memories"):
            # Ensure states are tensors and have correct dimensions
            state = memory['state']
            next_state = memory['next_state']
            
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state)
                
            # Add batch dimension if needed for the remember method
            if state.dim() == 3:
                state = state.unsqueeze(0)
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)
            
            self.agent.remember(
                state=state,
                action=memory['action'],
                reward=memory['reward'],
                next_state=next_state,
                done=memory['done']
            )
        
        print(f"Initialized replay buffer with {len(self.agent.memory)} experiences")
        
        # Pre-train on memories
        print("Pre-training agent on domirank memories...")
        for _ in tqdm(range(self.config['pretrain_steps']), desc="Pre-training"):
            self.agent.replay()
        
        print("Domirank memory initialization complete!")
    
    def create_environment(self, landscape_data: Dict[str, np.ndarray]) -> FireEnv:
        """Create a fire environment for the given landscape."""
        return FireEnv(
            slope=landscape_data['slp'],
            aspect=landscape_data['asp'],
            dem=landscape_data['dem'],
            cc=landscape_data['cc'],
            cbd=landscape_data['cbd'],
            cbh=landscape_data['cbh'],
            ch=landscape_data['ch'],
            fuel_model=landscape_data['fbfm']
        )
    
    def run_iterative_episode(self, landscape_idx: int, max_fuel_break_percentage: float = 20.0) -> Dict:
        """
        Run a single episode with iterative fuel break placement.
        
        Args:
            landscape_idx: Index of the landscape to use
            max_fuel_break_percentage: Maximum percentage of fuel breaks to place
            
        Returns:
            Episode results dictionary
        """
        # Load landscape data
        landscape_data = self.memory_loader.load_landscape_data(landscape_idx)
        env = self.create_environment(landscape_data)
        
        # Initialize episode
        state = env.reset()
        state_tensor = self.agent.preprocess_state(landscape_data)
        
        current_fuel_breaks = np.zeros((self.config['grid_size'], self.config['grid_size']), dtype=bool)
        episode_reward = 0
        episode_actions = []
        episode_percentages = []
        
        # Iterative fuel break placement
        target_percentage = 1.0  # Start with 1%
        increment = self.config['percentage_increment']
        
        while target_percentage <= max_fuel_break_percentage:
            # Calculate how many fuel breaks we need for this percentage
            total_cells = self.config['grid_size'] * self.config['grid_size']
            target_num_breaks = int(total_cells * target_percentage / 100.0)
            current_num_breaks = np.sum(current_fuel_breaks)
            
            if current_num_breaks < target_num_breaks:
                # Need to place more fuel breaks
                action = self.agent.act(state_tensor, current_fuel_breaks)
                
                # Convert action to row, col
                row = action // self.config['grid_size']
                col = action % self.config['grid_size']
                
                # Place fuel break if not already there
                if not current_fuel_breaks[row, col]:
                    current_fuel_breaks[row, col] = True
                    
                    # Evaluate environment with new fuel breaks
                    fuel_break_action = current_fuel_breaks.flatten().astype(int)
                    next_state, reward, done, info = env.step(fuel_break_action)
                    
                    # Store experience
                    next_state_tensor = state_tensor  # State doesn't change, only fuel breaks
                    self.agent.remember(state_tensor, action, reward, next_state_tensor, done)
                    
                    episode_reward += reward
                    episode_actions.append(action)
                    episode_percentages.append(target_percentage)
                    
                    # Train agent
                    self.agent.replay()
                    
                    print(f"  Placed fuel break at ({row}, {col}), "
                          f"percentage: {target_percentage:.1f}%, reward: {reward:.2f}")
            
            # Move to next percentage increment
            target_percentage += increment
        
        return {
            'total_reward': episode_reward,
            'num_actions': len(episode_actions),
            'actions': episode_actions,
            'percentages': episode_percentages,
            'final_fuel_breaks': current_fuel_breaks.copy(),
            'landscape_idx': landscape_idx
        }
    
    def train(self):
        """Main training loop."""
        print("Starting Deep Q-Learning training for fuel break placement...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Initialize with domirank memories
        self.initialize_with_domirank_memories()
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            print(f"\n--- Episode {episode + 1}/{self.config['num_episodes']} ---")
            
            # Select random landscape for this episode
            landscape_idx = np.random.randint(0, self.config['num_landscapes'])
            
            # Run episode
            episode_results = self.run_iterative_episode(
                landscape_idx=landscape_idx,
                max_fuel_break_percentage=self.config['max_fuel_break_percentage']
            )
            
            # Record metrics
            episode_reward = episode_results['total_reward']
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['fuel_break_percentages'].append(
                episode_results['percentages'][-1] if episode_results['percentages'] else 0
            )
            self.training_metrics['episodes_completed'] = episode + 1
            
            # Track best performance
            if episode_reward > self.training_metrics['best_reward']:
                self.training_metrics['best_reward'] = episode_reward
                # Save best model
                best_model_path = os.path.join(self.models_dir, 'best_model.pt')
                self.agent.save_model(best_model_path)
                print(f"New best reward: {episode_reward:.2f} - Model saved!")
            
            print(f"Episode reward: {episode_reward:.2f}, "
                  f"Actions taken: {episode_results['num_actions']}, "
                  f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Update target network periodically
            if (episode + 1) % self.config['target_update_frequency'] == 0:
                self.agent.update_target_network()
                print("Target network updated")
            
            # Save checkpoint periodically
            if (episode + 1) % self.config['checkpoint_frequency'] == 0:
                checkpoint_path = os.path.join(self.models_dir, f'checkpoint_episode_{episode + 1}.pt')
                self.agent.save_model(checkpoint_path)
                self.save_training_metrics()
                self.plot_training_progress()
                print(f"Checkpoint saved at episode {episode + 1}")
        
        # Final save
        final_model_path = os.path.join(self.models_dir, 'final_model.pt')
        self.agent.save_model(final_model_path)
        self.save_training_metrics()
        self.plot_training_progress()
        
        print("\nTraining completed!")
        print(f"Best reward achieved: {self.training_metrics['best_reward']:.2f}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        
    def save_training_metrics(self):
        """Save training metrics to file."""
        metrics_file = os.path.join(self.output_dir, 'training_metrics.json')
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {}
        for key, value in self.training_metrics.items():
            if isinstance(value, (list, np.ndarray)):
                metrics_to_save[key] = list(value)
            else:
                metrics_to_save[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"Training metrics saved to {metrics_file}")
    
    def plot_training_progress(self):
        """Create plots showing training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        if len(self.training_metrics['episode_rewards']) > 10:
            window = min(50, len(self.training_metrics['episode_rewards']) // 10)
            rewards = np.array(self.training_metrics['episode_rewards'])
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Agent losses
        if self.agent.losses:
            axes[1, 0].plot(self.agent.losses)
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
        
        # Fuel break percentages achieved
        axes[1, 1].plot(self.training_metrics['fuel_break_percentages'])
        axes[1, 1].set_title('Fuel Break Percentages Achieved')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(self.plots_dir, 'training_progress.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress plot saved to {plot_file}")
    
    def evaluate_model(self, model_path: str, num_eval_episodes: int = 10):
        """Evaluate a trained model."""
        print(f"Evaluating model: {model_path}")
        
        # Load model
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        eval_results = []
        
        for episode in range(num_eval_episodes):
            landscape_idx = episode % self.config['num_landscapes']
            
            episode_results = self.run_iterative_episode(
                landscape_idx=landscape_idx,
                max_fuel_break_percentage=self.config['max_fuel_break_percentage']
            )
            
            eval_results.append(episode_results)
            print(f"Eval episode {episode + 1}: Reward = {episode_results['total_reward']:.2f}")
        
        # Calculate statistics
        total_rewards = [r['total_reward'] for r in eval_results]
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"\nEvaluation Results:")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Best reward: {max(total_rewards):.2f}")
        print(f"Worst reward: {min(total_rewards):.2f}")
        
        return eval_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Deep Q-Learning agent for fuel break placement')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--num_landscapes', type=int, default=5, help='Number of landscape samples')
    parser.add_argument('--grid_size', type=int, default=50, help='Grid size for landscapes')
    parser.add_argument('--regenerate_memories', action='store_true', help='Regenerate domirank memories')
    parser.add_argument('--evaluate_only', type=str, help='Path to model for evaluation only')
    
    return parser.parse_args()


def get_default_config():
    """Get default training configuration."""
    return {
        # Environment settings
        'raster_dir': 'cropped_raster',
        'grid_size': 50,
        'input_channels': 12,
        'num_landscapes': 5,
        
        # Training settings
        'num_episodes': 1000,
        'max_fuel_break_percentage': 20.0,
        'percentage_increment': 1.0,
        'memory_simulations': 5,
        'pretrain_steps': 1000,
        
        # Agent settings
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'initial_epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 100000,
        'batch_size': 32,
        
        # Training schedule
        'target_update_frequency': 100,
        'checkpoint_frequency': 100,
        
        # Output settings
        'output_dir': 'outputs',
        'regenerate_memories': False
    }


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.num_episodes:
        config['num_episodes'] = args.num_episodes
    if args.num_landscapes:
        config['num_landscapes'] = args.num_landscapes
    if args.grid_size:
        config['grid_size'] = args.grid_size
    if args.regenerate_memories:
        config['regenerate_memories'] = args.regenerate_memories
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = os.path.join(config['output_dir'], f"dqn_fuel_breaks_{timestamp}")
    
    # Initialize trainer
    trainer = IterativeFuelBreakTrainer(config)
    
    if args.evaluate_only:
        # Evaluation mode
        trainer.evaluate_model(args.evaluate_only)
    else:
        # Training mode
        trainer.train()
        
        # Evaluate best model
        best_model_path = os.path.join(trainer.models_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print("\nEvaluating best model...")
            trainer.evaluate_model(best_model_path)


if __name__ == "__main__":
    main()