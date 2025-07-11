#!/usr/bin/env python3
"""
Parallel Deep Q-Learning Training Script for Fuel Break Placement

This script implements parallel Deep Q-learning using vectorized environments
to run multiple fire simulations simultaneously, providing much faster training
and more diverse experiences.

Key Features:
- Vectorized environments for parallel simulation
- Configurable number of parallel environments
- Both threading and multiprocessing support
- Advanced experience collection strategies
- Real-time performance monitoring
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
import time

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.DQNAgent import DQNAgent
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.VectorizedFireEnv import VectorizedFireEnv, ParallelExperienceCollector
from src.scripts.FireEnv import FireEnv
from src.scripts.Simulate import Simulate


class ParallelFuelBreakTrainer:
    """
    Parallel trainer class for fuel break placement using vectorized environments.
    
    This trainer uses multiple environments running in parallel to collect
    experiences much faster than sequential training, leading to improved
    sample efficiency and faster convergence.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Initializing with config:")
        print(f"   - num_landscapes: {config['num_landscapes']}")
        print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
        print(f"   - memory_simulations: {config['memory_simulations']}")
        
        # Load landscape data for all environments
        self.memory_loader = DomiRankMemoryLoader(
            raster_dir=config['raster_dir'],
            grid_size=config['grid_size']
        )
        
        # Generate or load landscape data - ensure we have enough landscapes
        if not os.path.exists(config['raster_dir']):
            print(f"🏞️  Generating {config['num_landscapes']} landscape files...")
            self.memory_loader.generate_sample_data(config['num_landscapes'])
        
        # Load landscape data for vectorized environments
        self.landscape_data_list = []
        available_landscapes = min(config['num_landscapes'], self._count_available_landscapes())
        
        print(f"📂 Loading {available_landscapes} available landscapes...")
        for i in range(available_landscapes):
            try:
                landscape_data = self.memory_loader.load_landscape_data(i)
                self.landscape_data_list.append(landscape_data)
                if i < 3:  # Only print first few
                    print(f"   ✅ Loaded landscape {i}")
            except Exception as e:
                print(f"   ❌ Failed to load landscape {i}: {e}")
                break
        
        print(f"📊 Landscape data summary:")
        print(f"   - Available landscapes: {len(self.landscape_data_list)}")
        print(f"   - Will cycle through landscapes for {config['num_parallel_envs']} environments")
        
        if len(self.landscape_data_list) == 0:
            raise ValueError("No landscape data available! Check your data files.")
        
        # Initialize DQN agent
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
        
        # Initialize vectorized environment with fire simulation parameters
        self.vectorized_env = VectorizedFireEnv(
            landscape_data_list=self.landscape_data_list,
            num_envs=config['num_parallel_envs'],
            method=config['parallel_method'],
            max_workers=config.get('max_workers', None),
            num_simulations=config['memory_simulations'],
            max_duration=config.get('fire_simulation_max_duration', None)
        )
        
        # Initialize experience collector
        self.experience_collector = ParallelExperienceCollector(
            vectorized_env=self.vectorized_env,
            agent=self.agent,
            collection_batch_size=config.get('collection_batch_size', 64),
            experience_buffer_size=config.get('experience_buffer_size', 2000)
        )
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'parallel_rewards': [],
            'collection_times': [],
            'experiences_per_second': [],
            'environments_reset': [],
            'best_reward': -float('inf'),
            'total_steps': 0,
            'total_experiences': 0
        }
        
        # Create output directories
        self.setup_output_directories()
    
    def _count_available_landscapes(self) -> int:
        """Count how many landscape files are actually available."""
        count = 0
        while True:
            try:
                self.memory_loader.load_landscape_data(count)
                count += 1
                if count > 1000:  # Reasonable upper limit
                    break
            except:
                break
        return count
        
    def setup_output_directories(self):
        """Create directories for outputs."""
        self.output_dir = self.config['output_dir']
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.memories_dir = os.path.join(self.output_dir, 'memories')
        
        for directory in [self.output_dir, self.models_dir, self.plots_dir, self.memories_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_with_domirank_memories(self):
        """Initialize agent with domirank-based memories."""
        print("Initializing agent with domirank-based memories...")
        
        # Generate or load memories
        memory_file = os.path.join(self.memories_dir, 'domirank_memories.pt')
        
        if os.path.exists(memory_file) and not self.config['regenerate_memories']:
            print(f"Loading existing memories from {memory_file}")
            memories = self.memory_loader.load_memories(memory_file)
        else:
            print("Generating new domirank memories...")
            
            # Generate training memories
            landscape_indices = list(range(self.config['num_landscapes']))
            percentages = list(range(1, 21, self.config.get('percentage_increment', 2)))  # More sparse for speed
            
            memories = self.memory_loader.generate_training_memories(
                landscape_indices=landscape_indices,
                percentages=percentages,
                num_simulations=self.config['memory_simulations'],
                max_duration=self.config.get('fire_simulation_max_duration', None)
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
    
    def run_parallel_collection_episode(self, steps_per_episode: int) -> Dict:
        """
        Run one episode of parallel experience collection.
        
        Args:
            steps_per_episode: Number of environment steps to collect
            
        Returns:
            Episode statistics
        """
        print(f"Running parallel collection episode with {self.vectorized_env.num_envs} environments...")
        
        # Collect experiences using the parallel collector
        collection_stats = self.experience_collector.collect_experiences(
            num_steps=steps_per_episode,
            train_frequency=self.config.get('train_frequency', 4)
        )
        
        # Get environment statistics
        env_stats = self.vectorized_env.get_episode_stats()
        
        # Get performance statistics
        perf_stats = self.experience_collector.get_performance_stats()
        
        # Combine all statistics - prioritize collection_stats mean_reward over env_stats
        episode_results = {
            **env_stats,
            **perf_stats,
            **collection_stats,  # Put this last so it overwrites env_stats mean_reward
            'agent_epsilon': self.agent.epsilon,
            'agent_losses_count': len(self.agent.losses)
        }
        
        return episode_results
    
    def train(self):
        """Main parallel training loop."""
        print("Starting Parallel Deep Q-Learning training for fuel break placement...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        print(f"Using {self.vectorized_env.num_envs} parallel environments with {self.config['parallel_method']} method")
        
        # Initialize with domirank memories
        self.initialize_with_domirank_memories()
        
        # Calculate steps per episode based on configuration
        steps_per_episode = self.config.get('steps_per_episode', 50)
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            episode_start_time = time.time()
            
            print(f"\n--- Episode {episode + 1}/{self.config['num_episodes']} ---")
            
            # Run parallel experience collection
            episode_results = self.run_parallel_collection_episode(steps_per_episode)
            
            # Update metrics
            self.training_metrics['parallel_rewards'].append(episode_results['mean_reward'])
            self.training_metrics['collection_times'].append(episode_results['collection_time'])
            self.training_metrics['experiences_per_second'].append(episode_results['experiences_per_second'])
            self.training_metrics['environments_reset'].append(episode_results['environments_reset'])
            self.training_metrics['total_steps'] += episode_results['total_steps']
            self.training_metrics['total_experiences'] += episode_results['experiences_collected']
            
            episode_reward = episode_results['mean_reward']
            
            # Track best performance
            if episode_reward > self.training_metrics['best_reward']:
                self.training_metrics['best_reward'] = episode_reward
                # Save best model
                best_model_path = os.path.join(self.models_dir, 'best_model.pt')
                self.agent.save_model(best_model_path)
                print(f"New best reward: {episode_reward:.2f} - Model saved!")
            
            episode_time = time.time() - episode_start_time
            
            print(f"Episode Results:")
            print(f"  Mean reward: {episode_reward:.2f}")
            print(f"  Total steps: {episode_results['total_steps']}")
            print(f"  Experiences collected: {episode_results['experiences_collected']}")
            print(f"  Training steps: {episode_results['training_steps']}")
            print(f"  Environments reset: {episode_results['environments_reset']}")
            print(f"  Collection time: {episode_results['collection_time']:.2f}s")
            print(f"  Experiences/second: {episode_results['experiences_per_second']:.1f}")
            print(f"  Episode time: {episode_time:.2f}s")
            print(f"  Agent epsilon: {episode_results['agent_epsilon']:.3f}")
            
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
        
        print("\nParallel training completed!")
        print(f"Best reward achieved: {self.training_metrics['best_reward']:.2f}")
        print(f"Total experiences collected: {self.training_metrics['total_experiences']}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        
        # Keep vectorized environment open for evaluation - will be closed later
        
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
        """Create plots showing parallel training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Parallel rewards
        if self.training_metrics['parallel_rewards']:
            axes[0, 0].plot(self.training_metrics['parallel_rewards'])
            axes[0, 0].set_title('Parallel Environment Mean Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Mean Reward')
            axes[0, 0].grid(True)
        
        # Collection times
        if self.training_metrics['collection_times']:
            axes[0, 1].plot(self.training_metrics['collection_times'])
            axes[0, 1].set_title('Experience Collection Times')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].grid(True)
        
        # Experiences per second
        if self.training_metrics['experiences_per_second']:
            axes[0, 2].plot(self.training_metrics['experiences_per_second'])
            axes[0, 2].set_title('Collection Efficiency')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Experiences/Second')
            axes[0, 2].grid(True)
        
        # Environment resets
        if self.training_metrics['environments_reset']:
            axes[1, 0].plot(self.training_metrics['environments_reset'])
            axes[1, 0].set_title('Environment Resets per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Number of Resets')
            axes[1, 0].grid(True)
        
        # Agent losses
        if self.agent.losses:
            recent_losses = self.agent.losses[-1000:]  # Show recent losses
            axes[1, 1].plot(recent_losses)
            axes[1, 1].set_title('Training Losses (Recent)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        # Cumulative experiences
        cumulative_experiences = np.cumsum([len(self.training_metrics['parallel_rewards']) * 
                                          self.config.get('steps_per_episode', 50) * 
                                          self.vectorized_env.num_envs] * len(self.training_metrics['parallel_rewards']))
        if len(cumulative_experiences) > 0:
            axes[1, 2].plot(cumulative_experiences)
            axes[1, 2].set_title('Cumulative Experiences Collected')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Total Experiences')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(self.plots_dir, 'parallel_training_progress.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress plot saved to {plot_file}")
    
    def evaluate_model(self, model_path: str, num_eval_episodes: int = 10):
        """Evaluate a trained model using parallel environments."""
        print(f"Evaluating model: {model_path}")
        
        # Load model
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        eval_results = []
        
        for episode in range(num_eval_episodes):
            print(f"Evaluation episode {episode + 1}/{num_eval_episodes}")
            
            # Run parallel collection for evaluation
            episode_results = self.run_parallel_collection_episode(
                steps_per_episode=self.config.get('eval_steps_per_episode', 30)
            )
            
            eval_results.append(episode_results)
            print(f"  Mean reward: {episode_results['mean_reward']:.2f}")
            print(f"  Experiences/second: {episode_results['experiences_per_second']:.1f}")
        
        # Calculate statistics
        mean_rewards = [r['mean_reward'] for r in eval_results]
        collection_times = [r['collection_time'] for r in eval_results]
        
        avg_reward = np.mean(mean_rewards)
        std_reward = np.std(mean_rewards)
        avg_collection_time = np.mean(collection_times)
        
        print(f"\nEvaluation Results:")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Best reward: {max(mean_rewards):.2f}")
        print(f"Worst reward: {min(mean_rewards):.2f}")
        print(f"Average collection time: {avg_collection_time:.2f}s")
        print(f"Total environments used: {self.vectorized_env.num_envs}")
        
        return eval_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Deep Q-Learning agent with parallel environments')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--num_episodes', type=int, help='Number of training episodes')
    parser.add_argument('--num_parallel_envs', type=int, help='Number of parallel environments')
    parser.add_argument('--parallel_method', type=str, 
                       choices=['threading', 'multiprocessing', 'sequential'],
                       help='Parallelization method')
    parser.add_argument('--evaluate_only', type=str, help='Path to model for evaluation only')
    
    return parser.parse_args()


def get_default_parallel_config():
    """Get default parallel training configuration."""
    return {
        # Environment settings
        'raster_dir': 'cropped_raster',
        'grid_size': 50,
        'input_channels': 8,
        'num_landscapes': 4,
        
        # Parallel training settings
        'num_episodes': 100,
        'num_parallel_envs': 8,
        'parallel_method': 'threading',
        'max_workers': None,
        'steps_per_episode': 50,
        'train_frequency': 4,
        'collection_batch_size': 64,
        'experience_buffer_size': 2000,
        
        # Memory initialization
        'memory_simulations': 3,
        'pretrain_steps': 200,
        'percentage_increment': 2,
        
        # Agent settings
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'initial_epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 50000,
        'batch_size': 32,
        
        # Training schedule
        'target_update_frequency': 20,
        'checkpoint_frequency': 20,
        
        # Output settings
        'output_dir': 'outputs',
        'regenerate_memories': False
    }


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded - num_parallel_envs: {config.get('num_parallel_envs', 'NOT SET')}")
    else:
        print("⚠️  No config file specified, using default configuration")
        config = get_default_parallel_config()
        print(f"📋 Default config - num_parallel_envs: {config['num_parallel_envs']}")
    
    # Override with command line arguments (only if explicitly provided)
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
        print(f"🔧 Command line override - output_dir: {args.output_dir}")
    if args.num_episodes is not None:
        config['num_episodes'] = args.num_episodes
        print(f"🔧 Command line override - num_episodes: {args.num_episodes}")
    if args.num_parallel_envs is not None:
        config['num_parallel_envs'] = args.num_parallel_envs
        print(f"🔧 Command line override - num_parallel_envs: {args.num_parallel_envs}")
    if args.parallel_method is not None:
        config['parallel_method'] = args.parallel_method
        print(f"🔧 Command line override - parallel_method: {args.parallel_method}")
    
    # Set defaults for missing values
    if 'output_dir' not in config:
        config['output_dir'] = 'outputs'
    
    print(f"\n🎯 Final configuration:")
    print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
    print(f"   - memory_simulations: {config.get('memory_simulations', 'NOT SET')}")
    print(f"   - grid_size: {config.get('grid_size', 'NOT SET')}")
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = os.path.join(config['output_dir'], f"parallel_dqn_fuel_breaks_{timestamp}")
    
    # Initialize trainer
    trainer = ParallelFuelBreakTrainer(config)
    
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
    
    # Final cleanup
    print("🧹 Cleaning up resources...")
    trainer.vectorized_env.close()
    print("✅ Cleanup completed")


if __name__ == "__main__":
    main()
