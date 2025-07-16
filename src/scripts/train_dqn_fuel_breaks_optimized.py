#!/usr/bin/env python3
"""
Optimized Parallel Deep Q-Learning Training Script for Fuel Break Placement

This script implements optimized Deep Q-learning using gym.vector.SyncVectorEnv
for proper parallelization, with reduced model size, better memory management,
and random environment sampling from all available landscapes.

Key Optimizations:
- Uses gym.vector.SyncVectorEnv for proper parallelization
- Randomly samples environments from all available landscapes
- Reduced model size (hidden layers by factor of 2)
- Fixed memory leaks with proper cleanup
- Improved performance with GPU memory management
- Better experience collection and training strategies
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
import gc

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.DQNAgent_optimized import OptimizedDQNAgent
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.GymVectorizedFireEnv import OptimizedGymVectorizedFireEnv, OptimizedExperienceCollector


class OptimizedFuelBreakTrainer:
    """
    Optimized trainer class for fuel break placement using gym.vector environments.
    
    This trainer uses properly parallelized environments with random sampling
    from all available landscapes, reduced model size, and optimized memory management.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing OptimizedFuelBreakTrainer with config:")
        print(f"   - num_landscapes: {config['num_landscapes']}")
        print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
        print(f"   - memory_simulations: {config['memory_simulations']}")
        print(f"   - device: {self.device}")
        
        # Load all available landscape data
        self.memory_loader = DomiRankMemoryLoader(
            raster_dir=config['raster_dir'],
            grid_size=config['grid_size']
        )
        
        # Generate or load landscape data
        if not os.path.exists(config['raster_dir']):
            print(f"🏞️  Generating {config['num_landscapes']} landscape files...")
            self.memory_loader.generate_sample_data(config['num_landscapes'])
        
        # Load ALL available landscape data
        self.landscape_data_list = []
        available_landscapes = min(config['num_landscapes'], self._count_available_landscapes())
        
        print(f"📂 Loading {available_landscapes} available landscapes...")
        for i in range(available_landscapes):
            try:
                landscape_data = self.memory_loader.load_landscape_data(i)
                self.landscape_data_list.append(landscape_data)
                if i < 5:  # Only print first few
                    print(f"   ✅ Loaded landscape {i}")
            except Exception as e:
                print(f"   ❌ Failed to load landscape {i}: {e}")
                break
        
        print(f"📊 Landscape data summary:")
        print(f"   - Total available landscapes: {len(self.landscape_data_list)}")
        print(f"   - Will randomly sample {config['num_parallel_envs']} environments")
        
        if len(self.landscape_data_list) == 0:
            raise ValueError("No landscape data available! Check your data files.")
        
        # Initialize optimized DQN agent
        self.agent = OptimizedDQNAgent(
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
        
        # Initialize optimized gym.vector environment
        self.vectorized_env = OptimizedGymVectorizedFireEnv(
            landscape_data_list=self.landscape_data_list,
            num_parallel_envs=config['num_parallel_envs'],
            num_simulations=config['memory_simulations'],
            max_duration=config.get('fire_simulation_max_duration', None),
            random_seed=config.get('random_seed', None)
        )
        
        # Initialize optimized experience collector
        self.experience_collector = OptimizedExperienceCollector(
            vectorized_env=self.vectorized_env,
            agent=self.agent,
            collection_batch_size=config.get('collection_batch_size', 32)
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
            'total_experiences': 0,
            'memory_usage': []
        }
        
        # Create output directories
        self.setup_output_directories()
    
    def _count_available_landscapes(self) -> int:
        """Count available landscape files."""
        raster_dir = self.config['raster_dir']
        slp_dir = os.path.join(raster_dir, 'slp')
        
        if not os.path.exists(slp_dir):
            return 0
        
        # Count .npy files in slp directory
        slp_files = [f for f in os.listdir(slp_dir) if f.endswith('.npy')]
        return len(slp_files)
    
    def setup_output_directories(self):
        """Setup output directories for saving models and results."""
        base_dir = self.config['output_dir']
        os.makedirs(base_dir, exist_ok=True)
        
        self.models_dir = os.path.join(base_dir, 'models')
        self.plots_dir = os.path.join(base_dir, 'plots')
        self.logs_dir = os.path.join(base_dir, 'logs')
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"📁 Output directories created: {base_dir}")
    
    def initialize_with_domirank_memories(self):
        """Initialize agent with domirank-based experiences."""
        print("🧠 Initializing agent with domirank memories...")
        
        # Select a subset of landscapes for memory initialization
        memory_landscapes = min(5, len(self.landscape_data_list))
        selected_indices = np.random.choice(
            len(self.landscape_data_list), 
            size=memory_landscapes, 
            replace=False
        )
        
        try:
            memories = self.memory_loader.generate_training_memories(
                landscape_indices=selected_indices.tolist(),
                percentages=[1, 3, 5, 8, 10, 15, 20],
                num_simulations=self.config['memory_simulations'],
                max_duration=self.config.get('fire_simulation_max_duration', None)
            )
            
            print(f"✅ Generated {len(memories)} initial memories")
            
            # Add memories to agent's replay buffer
            for memory in memories:
                self.agent.remember(
                    state=memory['state'],
                    action=memory['action'],
                    reward=memory['reward'],
                    next_state=memory['next_state'],
                    done=memory['done']
                )
            
            # Pre-train with these memories
            pretrain_steps = self.config.get('pretrain_steps', 50)
            print(f"🏋️ Pre-training with {pretrain_steps} steps...")
            
            for _ in range(pretrain_steps):
                self.agent.replay()
            
            print("✅ Agent initialized with domirank memories")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize with domirank memories: {e}")
            print("   Continuing with random initialization...")
    
    def run_optimized_collection_episode(self, steps_per_episode: int) -> Dict:
        """Run one episode of optimized experience collection."""
        train_frequency = self.config.get('train_frequency', 4)
        
        # Collect experiences
        results = self.experience_collector.collect_experiences(
            num_steps=steps_per_episode,
            train_frequency=train_frequency
        )
        
        # Add memory usage tracking
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            results['memory_usage'] = memory_used
            self.training_metrics['memory_usage'].append(memory_used)
        
        return results
    
    def train(self):
        """Main optimized training loop."""
        print("🚀 Starting Optimized Deep Q-Learning training for fuel break placement...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        print(f"Using {self.vectorized_env.num_envs} parallel environments with gym.vector.SyncVectorEnv")
        
        # Initialize with domirank memories
        self.initialize_with_domirank_memories()
        
        # Calculate steps per episode
        steps_per_episode = self.config.get('steps_per_episode', 50)
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            episode_start_time = time.time()
            
            print(f"\n--- Episode {episode + 1}/{self.config['num_episodes']} ---")
            
            # Run optimized experience collection
            episode_results = self.run_optimized_collection_episode(steps_per_episode)
            
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
                best_model_path = os.path.join(self.models_dir, 'best_model.pt')
                self.agent.save_model(best_model_path)
                print(f"✨ New best reward: {episode_reward:.2f} - Model saved!")
            
            episode_time = time.time() - episode_start_time
            
            # Print detailed results
            print(f"Episode Results:")
            print(f"  Mean reward: {episode_reward:.2f}")
            print(f"  Total steps: {episode_results['total_steps']}")
            print(f"  Experiences collected: {episode_results['experiences_collected']}")
            print(f"  Training steps: {episode_results['training_steps']}")
            print(f"  Episodes completed: {episode_results['episodes_completed']}")
            print(f"  Collection time: {episode_results['collection_time']:.2f}s")
            print(f"  Experiences/second: {episode_results['experiences_per_second']:.1f}")
            print(f"  Episode time: {episode_time:.2f}s")
            print(f"  Agent epsilon: {episode_results['agent_epsilon']:.3f}")
            
            # Memory usage tracking
            if 'memory_usage' in episode_results:
                print(f"  GPU memory usage: {episode_results['memory_usage']:.2f} GB")
            
            # Print enhanced environment diversity statistics
            env_stats = episode_results['performance_stats']
            print(f"  Environment diversity stats:")
            print(f"    Total episodes: {env_stats['total_episodes']}")
            print(f"    Average reward: {env_stats['average_reward']:.2f}")
            print(f"    Current landscapes: {env_stats['current_landscape_indices']}")
            print(f"    Unique landscapes used: {env_stats['unique_environments_used']}/{env_stats['total_available_landscapes']}")
            print(f"    Diversity coverage: {env_stats['diversity_percentage']:.1f}%")
            print(f"    Total environment selections: {env_stats['total_environment_selections']}")
            
            # Update target network periodically
            if (episode + 1) % self.config['target_update_frequency'] == 0:
                self.agent.update_target_network()
                print("🎯 Target network updated")
            
            # Save checkpoint periodically
            if (episode + 1) % self.config['checkpoint_frequency'] == 0:
                checkpoint_path = os.path.join(self.models_dir, f'checkpoint_episode_{episode + 1}.pt')
                self.agent.save_model(checkpoint_path)
                self.save_training_metrics()
                self.plot_training_progress()
                print(f"💾 Checkpoint saved at episode {episode + 1}")
            
            # Periodic cleanup
            if (episode + 1) % 10 == 0:
                self.cleanup_memory()
        
        # Final save
        final_model_path = os.path.join(self.models_dir, 'final_model.pt')
        self.agent.save_model(final_model_path)
        self.save_training_metrics()
        self.plot_training_progress()
        
        print("\n🎉 Optimized training completed!")
        print(f"Best reward achieved: {self.training_metrics['best_reward']:.2f}")
        print(f"Total experiences collected: {self.training_metrics['total_experiences']}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        
        # Final cleanup
        self.cleanup_memory()
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        print("🧹 Performing memory cleanup...")
        
        # Agent cleanup
        self.agent.cleanup_memory()
        
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def save_training_metrics(self):
        """Save training metrics to file."""
        metrics_path = os.path.join(self.logs_dir, 'training_metrics.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.training_metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, list):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"📊 Training metrics saved to {metrics_path}")
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        if self.training_metrics['parallel_rewards']:
            axes[0, 0].plot(self.training_metrics['parallel_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True)
        
        # Plot collection times
        if self.training_metrics['collection_times']:
            axes[0, 1].plot(self.training_metrics['collection_times'])
            axes[0, 1].set_title('Collection Times')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].grid(True)
        
        # Plot experiences per second
        if self.training_metrics['experiences_per_second']:
            axes[1, 0].plot(self.training_metrics['experiences_per_second'])
            axes[1, 0].set_title('Experiences per Second')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Experiences/sec')
            axes[1, 0].grid(True)
        
        # Plot memory usage
        if self.training_metrics['memory_usage']:
            axes[1, 1].plot(self.training_metrics['memory_usage'])
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training progress plot saved to {plot_path}")
    
    def evaluate_model(self, model_path: str, num_eval_episodes: int = 10):
        """Evaluate the trained model."""
        print(f"🔍 Evaluating model: {model_path}")
        
        # Load the model
        self.agent.load_model(model_path)
        
        # Set agent to evaluation mode (no exploration)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        # Run evaluation episodes
        evaluation_rewards = []
        
        for episode in range(num_eval_episodes):
            print(f"   Evaluation episode {episode + 1}/{num_eval_episodes}")
            
            # Reset environment
            observations = self.vectorized_env.reset()
            episode_rewards = np.zeros(self.vectorized_env.num_envs)
            done = np.zeros(self.vectorized_env.num_envs, dtype=bool)
            
            # Run episode
            max_steps = 100
            for step in range(max_steps):
                if done.all():
                    break
                
                # Get actions
                actions = []
                for i, obs in enumerate(observations):
                    if not done[i]:
                        action = self.agent.act(obs)
                        actions.append(action)
                    else:
                        actions.append(0)  # Dummy action for done environments
                
                # Execute actions
                observations, rewards, new_dones, infos = self.vectorized_env.step(actions)
                
                # Update episode rewards
                episode_rewards += rewards * (~done)
                done = done | new_dones
            
            evaluation_rewards.extend(episode_rewards)
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        # Calculate statistics
        mean_reward = np.mean(evaluation_rewards)
        std_reward = np.std(evaluation_rewards)
        
        print(f"📊 Evaluation Results:")
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Best reward: {np.max(evaluation_rewards):.2f}")
        print(f"   Worst reward: {np.min(evaluation_rewards):.2f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'best_reward': np.max(evaluation_rewards),
            'worst_reward': np.min(evaluation_rewards),
            'all_rewards': evaluation_rewards
        }
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        try:
            self.cleanup_memory()
            if hasattr(self, 'vectorized_env'):
                self.vectorized_env.close()
        except:
            pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized Parallel DQN Training for Fuel Break Placement')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--num-episodes', type=int, help='Number of training episodes')
    parser.add_argument('--num-parallel-envs', type=int, help='Number of parallel environments')
    parser.add_argument('--evaluate-only', type=str, help='Path to model for evaluation only')
    return parser.parse_args()


def get_default_optimized_config():
    """Get default configuration for optimized training."""
    return {
        'raster_dir': 'cropped_raster',
        'grid_size': 50,
        'input_channels': 12,
        'num_landscapes': 20,  # Increased for better diversity
        
        'num_episodes': 200,
        'num_parallel_envs': 8,  # Optimized for most systems
        'steps_per_episode': 50,
        'train_frequency': 4,
        'collection_batch_size': 32,
        
        'memory_simulations': 50,  # Reduced for faster initialization
        'pretrain_steps': 50,
        'fire_simulation_max_duration': 120,
        
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'initial_epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 50000,  # Reduced for memory efficiency
        'batch_size': 32,
        
        'target_update_frequency': 10,
        'checkpoint_frequency': 10,
        
        'output_dir': 'optimized_outputs',
        'random_seed': 42,
        'regenerate_memories': True
    }


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("⚠️ No config file specified, using default optimized configuration")
        config = get_default_optimized_config()
    
    # Override with command line arguments
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.num_episodes is not None:
        config['num_episodes'] = args.num_episodes
    if args.num_parallel_envs is not None:
        config['num_parallel_envs'] = args.num_parallel_envs
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = os.path.join(config['output_dir'], f"optimized_dqn_fuel_breaks_{timestamp}")
    
    print(f"\n🎯 Final optimized configuration:")
    print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
    print(f"   - num_landscapes: {config['num_landscapes']}")
    print(f"   - memory_simulations: {config['memory_simulations']}")
    print(f"   - buffer_size: {config['buffer_size']}")
    print(f"   - output_dir: {config['output_dir']}")
    
    # Initialize trainer
    trainer = OptimizedFuelBreakTrainer(config)
    
    if args.evaluate_only:
        # Evaluation mode
        trainer.evaluate_model(args.evaluate_only)
    else:
        # Training mode
        trainer.train()
        
        # Evaluate best model
        best_model_path = os.path.join(trainer.models_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print("\n🔍 Evaluating best model...")
            trainer.evaluate_model(best_model_path)
    
    # Final cleanup
    print("🧹 Cleaning up resources...")
    trainer.cleanup_memory()
    trainer.vectorized_env.close()
    print("✅ Cleanup completed")


if __name__ == "__main__":
    main()