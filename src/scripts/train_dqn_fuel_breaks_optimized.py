#!/usr/bin/env python3
"""
Optimized Parallel Deep Q-Learning Training Script for Fuel Break Placement

This script implements an optimized version of parallel Deep Q-learning that:
1. Uses gym.vector.SyncVectorEnv for better parallelization
2. Randomly selects environments from all loaded landscapes
3. Includes performance optimizations and reduced model size

Key Improvements:
- Uses gym.vector for official vectorized environment support
- Random environment selection instead of cycling
- Reduced model size by factor of 2
- Better memory management and performance
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time
import random
import gym
from gym.vector import SyncVectorEnv

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.DQNAgent import DQNAgent
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.FireEnv import FireEnv
from src.scripts.Simulate import Simulate


class OptimizedFireEnvWrapper(gym.Wrapper):
    """
    Wrapper for FireEnv that optimizes it for use with gym.vector.SyncVectorEnv
    """
    def __init__(self, landscape_data: Dict, num_simulations: int = 10, max_duration: int = None):
        """
        Initialize wrapper with landscape data.
        
        Args:
            landscape_data: Dictionary containing landscape information
            num_simulations: Number of simulations to run
            max_duration: Maximum simulation duration in minutes
        """
        env = FireEnv(
            slope=landscape_data['slp'],
            aspect=landscape_data['asp'],
            dem=landscape_data['dem'],
            cc=landscape_data['cc'],
            cbd=landscape_data['cbd'],
            cbh=landscape_data['cbh'],
            ch=landscape_data['ch'],
            fuel_model=landscape_data['fbfm']
        )
        
        super().__init__(env)
        
        # Store simulation parameters
        self.num_simulations = num_simulations
        self.max_duration = max_duration
        self.landscape_data = landscape_data
        
        # Override environment simulation parameters
        self.env.num_simulations = num_simulations
        self.env.max_duration = max_duration
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        
    def reset(self):
        """Reset environment and return initial observation."""
        self.episode_reward = 0.0
        self.episode_steps = 0
        return self.env.reset()
        
    def step(self, action):
        """Step environment with action."""
        obs, reward, done, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Add episode statistics to info
        info['episode_reward'] = self.episode_reward
        info['episode_steps'] = self.episode_steps
        
        return obs, reward, done, info


def create_env_fn(landscape_data_list: List[Dict], num_simulations: int = 10, max_duration: int = None):
    """
    Create a factory function for creating environments with random landscape selection.
    
    Args:
        landscape_data_list: List of available landscape data
        num_simulations: Number of simulations per environment
        max_duration: Maximum simulation duration
        
    Returns:
        Function that creates a new environment instance
    """
    def _create_env():
        # Randomly select a landscape
        landscape_data = random.choice(landscape_data_list)
        return OptimizedFireEnvWrapper(
            landscape_data=landscape_data,
            num_simulations=num_simulations,
            max_duration=max_duration
        )
    return _create_env


class OptimizedParallelTrainer:
    """
    Optimized parallel trainer using gym.vector.SyncVectorEnv for better performance.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Initializing Optimized Parallel Trainer:")
        print(f"   - num_landscapes: {config['num_landscapes']}")
        print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
        print(f"   - memory_simulations: {config['memory_simulations']}")
        print(f"   - Device: {self.device}")
        
        # Load all available landscape data
        self.memory_loader = DomiRankMemoryLoader(
            raster_dir=config['raster_dir'],
            grid_size=config['grid_size']
        )
        
        # Generate or load landscape data
        if not os.path.exists(config['raster_dir']):
            print(f"🏞️  Generating {config['num_landscapes']} landscape files...")
            self.memory_loader.generate_sample_data(config['num_landscapes'])
        
        # Load all available landscapes
        self.landscape_data_list = []
        available_landscapes = min(config['num_landscapes'], self._count_available_landscapes())
        
        print(f"📂 Loading {available_landscapes} landscapes...")
        for i in range(available_landscapes):
            try:
                landscape_data = self.memory_loader.load_landscape_data(i)
                self.landscape_data_list.append(landscape_data)
                if i < 3:  # Only print first few
                    print(f"   ✅ Loaded landscape {i}")
            except Exception as e:
                print(f"   ❌ Failed to load landscape {i}: {e}")
                break
        
        if len(self.landscape_data_list) == 0:
            raise ValueError("No landscape data available! Check your data files.")
            
        print(f"📊 Loaded {len(self.landscape_data_list)} landscapes")
        
        # Initialize DQN agent with optimized settings
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
        
        # Create vectorized environment using gym.vector
        print(f"🚀 Creating {config['num_parallel_envs']} vectorized environments...")
        env_fns = [
            create_env_fn(
                landscape_data_list=self.landscape_data_list,
                num_simulations=config['memory_simulations'],
                max_duration=config.get('fire_simulation_max_duration', None)
            )
            for _ in range(config['num_parallel_envs'])
        ]
        
        self.vectorized_env = SyncVectorEnv(env_fns)
        
        print(f"✅ Created vectorized environment with {self.vectorized_env.num_envs} environments")
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'parallel_rewards': [],
            'collection_times': [],
            'environments_reset': 0,
            'best_reward': -float('inf'),
            'total_steps': 0,
            'total_experiences': 0
        }
        
        # Performance tracking
        self.step_times = []
        self.training_times = []
        
        # Create output directories
        self.setup_output_directories()
    
    def _count_available_landscapes(self) -> int:
        """Count how many landscape files are actually available."""
        count = 0
        while count < 1000:  # Reasonable upper limit
            try:
                self.memory_loader.load_landscape_data(count)
                count += 1
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
        print("🧠 Initializing agent with domirank-based memories...")
        
        # Generate or load memories
        memory_file = os.path.join(self.memories_dir, 'domirank_memories.pt')
        
        if os.path.exists(memory_file) and not self.config['regenerate_memories']:
            print(f"📁 Loading existing memories from {memory_file}")
            memories = self.memory_loader.load_memories(memory_file)
        else:
            print("🔄 Generating new domirank memories...")
            
            # Use fewer landscapes for memory generation to speed up
            landscape_indices = list(range(min(len(self.landscape_data_list), 10)))
            percentages = list(range(1, 21, self.config.get('percentage_increment', 3)))
            
            memories = self.memory_loader.generate_training_memories(
                landscape_indices=landscape_indices,
                percentages=percentages,
                num_simulations=self.config['memory_simulations'],
                max_duration=self.config.get('fire_simulation_max_duration', None)
            )
            
            # Save memories
            self.memory_loader.save_memories(memories, memory_file)
        
        # Add memories to agent's replay buffer
        print("📚 Adding memories to agent's replay buffer...")
        for memory in tqdm(memories, desc="Loading memories"):
            state = memory['state']
            next_state = memory['next_state']
            
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
            # Add batch dimension if needed
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
        
        print(f"✅ Initialized replay buffer with {len(self.agent.memory)} experiences")
        
        # Pre-train on memories
        print("🎓 Pre-training agent on domirank memories...")
        for _ in tqdm(range(self.config['pretrain_steps']), desc="Pre-training"):
            self.agent.replay()
        
        print("🎯 Domirank memory initialization complete!")
    
    def collect_experiences(self, num_steps: int) -> Dict:
        """
        Collect experiences using vectorized environments.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Collection statistics
        """
        start_time = time.time()
        
        # Reset environments
        observations = self.vectorized_env.reset()
        
        total_rewards = np.zeros(self.vectorized_env.num_envs)
        total_steps = 0
        experiences_collected = 0
        training_steps = 0
        environments_reset = 0
        
        for step in range(num_steps):
            step_start = time.time()
            
            # Convert observations to states for the agent
            states = []
            for i, obs in enumerate(observations):
                # Convert observation to landscape data format for agent
                landscape_data = self._obs_to_landscape_data(obs, i)
                state = self.agent.preprocess_state(landscape_data)
                states.append(state)
            
            # Get actions from agent
            actions = []
            for state in states:
                action = self.agent.act(state)
                actions.append(action)
            
            # Convert actions to environment format
            env_actions = [self._action_to_mask(action, obs.shape) for action, obs in zip(actions, observations)]
            
            # Step all environments
            next_observations, rewards, dones, infos = self.vectorized_env.step(env_actions)
            
            # Convert next observations to states
            next_states = []
            for i, next_obs in enumerate(next_observations):
                landscape_data = self._obs_to_landscape_data(next_obs, i)
                next_state = self.agent.preprocess_state(landscape_data)
                next_states.append(next_state)
            
            # Store experiences
            for i in range(len(observations)):
                self.agent.remember(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=dones[i]
                )
                experiences_collected += 1
            
            # Update rewards
            total_rewards += rewards
            total_steps += 1
            
            # Train agent periodically
            if (step + 1) % self.config.get('train_frequency', 4) == 0:
                self.agent.replay()
                training_steps += 1
            
            # Reset environments that are done
            if any(dones):
                # Count resets
                environments_reset += sum(dones)
                
                # Auto-reset is handled by SyncVectorEnv
                pass
            
            # Update observations for next iteration
            observations = next_observations
            
            # Track step time
            step_time = time.time() - step_start
            self.step_times.append(step_time)
        
        collection_time = time.time() - start_time
        
        return {
            'mean_reward': float(np.mean(total_rewards)),
            'total_steps': total_steps,
            'experiences_collected': experiences_collected,
            'training_steps': training_steps,
            'environments_reset': environments_reset,
            'collection_time': collection_time,
            'experiences_per_second': experiences_collected / collection_time if collection_time > 0 else 0.0,
            'avg_step_time': np.mean(self.step_times[-num_steps:]) if self.step_times else 0.0
        }
    
    def _obs_to_landscape_data(self, obs: np.ndarray, env_id: int) -> Dict:
        """Convert observation to landscape data format."""
        # For now, use the first landscape's data as template
        # In a real implementation, you'd want to track which landscape each env is using
        template = self.landscape_data_list[env_id % len(self.landscape_data_list)]
        return {
            'slp': template['slp'],
            'asp': template['asp'],
            'dem': template['dem'],
            'cc': template['cc'],
            'cbd': template['cbd'],
            'cbh': template['cbh'],
            'ch': template['ch'],
            'fbfm': template['fbfm'],
            'fire_type': obs  # Use observation as fire_type
        }
    
    def _action_to_mask(self, action: int, obs_shape: Tuple) -> np.ndarray:
        """Convert action index to fuel break mask."""
        mask = np.zeros(obs_shape, dtype=bool)
        if action < mask.size:
            idx = np.unravel_index(action, obs_shape)
            mask[idx] = True
        return mask
    
    def train(self):
        """Main optimized training loop."""
        print("🚀 Starting Optimized Parallel Deep Q-Learning training...")
        print(f"📊 Configuration: {json.dumps(self.config, indent=2)}")
        print(f"🔧 Using {self.vectorized_env.num_envs} parallel environments")
        
        # Initialize with domirank memories
        self.initialize_with_domirank_memories()
        
        steps_per_episode = self.config.get('steps_per_episode', 50)
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            episode_start_time = time.time()
            
            print(f"\n--- Episode {episode + 1}/{self.config['num_episodes']} ---")
            
            # Collect experiences
            episode_results = self.collect_experiences(steps_per_episode)
            
            # Update metrics
            self.training_metrics['parallel_rewards'].append(episode_results['mean_reward'])
            self.training_metrics['collection_times'].append(episode_results['collection_time'])
            self.training_metrics['environments_reset'] += episode_results['environments_reset']
            self.training_metrics['total_steps'] += episode_results['total_steps']
            self.training_metrics['total_experiences'] += episode_results['experiences_collected']
            
            episode_reward = episode_results['mean_reward']
            
            # Track best performance
            if episode_reward > self.training_metrics['best_reward']:
                self.training_metrics['best_reward'] = episode_reward
                best_model_path = os.path.join(self.models_dir, 'best_model.pt')
                self.agent.save_model(best_model_path)
                print(f"🎯 New best reward: {episode_reward:.2f} - Model saved!")
            
            episode_time = time.time() - episode_start_time
            self.training_times.append(episode_time)
            
            # Print episode results
            print(f"📊 Episode Results:")
            print(f"   Mean reward: {episode_reward:.2f}")
            print(f"   Total steps: {episode_results['total_steps']}")
            print(f"   Experiences collected: {episode_results['experiences_collected']}")
            print(f"   Training steps: {episode_results['training_steps']}")
            print(f"   Environments reset: {episode_results['environments_reset']}")
            print(f"   Collection time: {episode_results['collection_time']:.2f}s")
            print(f"   Experiences/second: {episode_results['experiences_per_second']:.1f}")
            print(f"   Avg step time: {episode_results['avg_step_time']:.4f}s")
            print(f"   Episode time: {episode_time:.2f}s")
            print(f"   Agent epsilon: {self.agent.epsilon:.3f}")
            print(f"   Replay buffer size: {len(self.agent.memory)}")
            
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
        
        # Final save
        final_model_path = os.path.join(self.models_dir, 'final_model.pt')
        self.agent.save_model(final_model_path)
        self.save_training_metrics()
        self.plot_training_progress()
        
        print("\n✅ Optimized training completed!")
        print(f"🏆 Best reward achieved: {self.training_metrics['best_reward']:.2f}")
        print(f"📈 Total experiences collected: {self.training_metrics['total_experiences']}")
        print(f"⏱️  Average episode time: {np.mean(self.training_times):.2f}s")
        print(f"🎯 Final epsilon: {self.agent.epsilon:.3f}")
        
        # Close vectorized environment
        self.vectorized_env.close()
        
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
        
        # Add performance metrics
        metrics_to_save['avg_episode_time'] = float(np.mean(self.training_times)) if self.training_times else 0.0
        metrics_to_save['avg_step_time'] = float(np.mean(self.step_times)) if self.step_times else 0.0
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"💾 Training metrics saved to {metrics_file}")
    
    def plot_training_progress(self):
        """Create plots showing training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Rewards
        if self.training_metrics['parallel_rewards']:
            axes[0, 0].plot(self.training_metrics['parallel_rewards'])
            axes[0, 0].set_title('Mean Rewards (Optimized)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Mean Reward')
            axes[0, 0].grid(True)
        
        # Collection times
        if self.training_metrics['collection_times']:
            axes[0, 1].plot(self.training_metrics['collection_times'])
            axes[0, 1].set_title('Collection Times')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].grid(True)
        
        # Episode times
        if self.training_times:
            axes[0, 2].plot(self.training_times)
            axes[0, 2].set_title('Episode Times')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Time (seconds)')
            axes[0, 2].grid(True)
        
        # Step times (recent)
        if self.step_times:
            recent_steps = self.step_times[-1000:]
            axes[1, 0].plot(recent_steps)
            axes[1, 0].set_title('Step Times (Recent 1000)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True)
        
        # Agent losses
        if self.agent.losses:
            recent_losses = self.agent.losses[-1000:]
            axes[1, 1].plot(recent_losses)
            axes[1, 1].set_title('Training Losses (Recent)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        # Cumulative experiences
        if self.training_metrics['parallel_rewards']:
            episodes = range(len(self.training_metrics['parallel_rewards']))
            cumulative_experiences = [
                (ep + 1) * self.config.get('steps_per_episode', 50) * self.vectorized_env.num_envs
                for ep in episodes
            ]
            axes[1, 2].plot(cumulative_experiences)
            axes[1, 2].set_title('Cumulative Experiences')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Total Experiences')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(self.plots_dir, 'optimized_training_progress.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training progress plot saved to {plot_file}")
    
    def evaluate_model(self, model_path: str, num_eval_episodes: int = 10):
        """Evaluate a trained model."""
        print(f"🔍 Evaluating model: {model_path}")
        
        # Load model
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        eval_results = []
        
        for episode in range(num_eval_episodes):
            print(f"Evaluation episode {episode + 1}/{num_eval_episodes}")
            
            episode_results = self.collect_experiences(
                self.config.get('eval_steps_per_episode', 30)
            )
            
            eval_results.append(episode_results)
            print(f"  Mean reward: {episode_results['mean_reward']:.2f}")
        
        # Calculate statistics
        mean_rewards = [r['mean_reward'] for r in eval_results]
        
        avg_reward = np.mean(mean_rewards)
        std_reward = np.std(mean_rewards)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Best reward: {max(mean_rewards):.2f}")
        print(f"   Worst reward: {min(mean_rewards):.2f}")
        
        return eval_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized Deep Q-Learning with vectorized environments')
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--num_episodes', type=int, help='Number of training episodes')
    parser.add_argument('--num_parallel_envs', type=int, help='Number of parallel environments')
    parser.add_argument('--evaluate_only', type=str, help='Path to model for evaluation only')
    
    return parser.parse_args()


def get_optimized_config():
    """Get optimized training configuration."""
    return {
        # Environment settings
        'raster_dir': 'cropped_raster',
        'grid_size': 50,
        'input_channels': 12,
        'num_landscapes': 8,  # Load more landscapes for better diversity
        
        # Parallel training settings
        'num_episodes': 200,
        'num_parallel_envs': 16,  # Increased for better parallelization
        'steps_per_episode': 50,
        'train_frequency': 4,
        
        # Memory initialization (optimized)
        'memory_simulations': 3,
        'pretrain_steps': 100,  # Reduced for faster startup
        'percentage_increment': 3,  # Slightly larger for speed
        
        # Agent settings (optimized)
        'learning_rate': 2e-4,  # Slightly higher for faster learning
        'gamma': 0.95,
        'initial_epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 100000,  # Increased for better memory
        'batch_size': 64,  # Increased for better gradients
        
        # Training schedule
        'target_update_frequency': 15,  # More frequent updates
        'checkpoint_frequency': 25,
        
        # Output settings
        'output_dir': 'outputs',
        'regenerate_memories': False,
        
        # Evaluation settings
        'eval_steps_per_episode': 30,
        
        # Fire simulation settings
        'fire_simulation_max_duration': None
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
        print("⚙️  Using optimized default configuration")
        config = get_optimized_config()
    
    # Override with command line arguments
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.num_episodes is not None:
        config['num_episodes'] = args.num_episodes
    if args.num_parallel_envs is not None:
        config['num_parallel_envs'] = args.num_parallel_envs
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = os.path.join(config['output_dir'], f"optimized_dqn_{timestamp}")
    
    print(f"\n🎯 Final Configuration:")
    print(f"   - num_parallel_envs: {config['num_parallel_envs']}")
    print(f"   - num_landscapes: {config['num_landscapes']}")
    print(f"   - batch_size: {config['batch_size']}")
    print(f"   - learning_rate: {config['learning_rate']}")
    
    # Initialize trainer
    trainer = OptimizedParallelTrainer(config)
    
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


if __name__ == "__main__":
    main()