import numpy as np
import torch
import gym
from gym import spaces
from gym.vector import SyncVectorEnv
import random
import time
import os
import sys
from typing import List, Dict, Tuple, Any, Optional
import copy

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.FireEnv import FireEnv
from src.scripts.Simulate import Simulate


class SingleFireEnvWrapper(gym.Env):
    """
    Gym-compatible wrapper for FireEnv that can be used with gym.vector.
    """
    
    def __init__(self, landscape_data: Dict, env_id: int, num_simulations: int = 10, 
                 max_duration: Optional[int] = None):
        super().__init__()
        
        self.landscape_data = landscape_data
        self.env_id = env_id
        self.num_simulations = num_simulations
        self.max_duration = max_duration
        
        # Initialize the underlying FireEnv
        self.fire_env = FireEnv(
            landscape_data=landscape_data,
            num_simulations=num_simulations,
            max_duration=max_duration
        )
        
        # Define action and observation spaces
        grid_size = landscape_data['slp'].shape[0]
        self.action_space = spaces.Discrete(grid_size * grid_size)
        
        # 12 channels: 8 landscape + 4 fireline intensity
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12, grid_size, grid_size),
            dtype=np.float32
        )
        
        # Track episode statistics
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
    def reset(self, **kwargs):
        """Reset environment and return initial observation."""
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
        # Reset the underlying FireEnv
        initial_state = self.fire_env.reset()
        
        # Convert to observation format
        observation = self._state_to_observation(initial_state)
        
        return observation
    
    def step(self, action):
        """Execute action and return next observation, reward, done, info."""
        self.episode_steps += 1
        
        # Execute action in underlying FireEnv
        next_state, reward, done, info = self.fire_env.step(action)
        
        # Convert state to observation
        observation = self._state_to_observation(next_state)
        
        # Add environment-specific info
        info['env_id'] = self.env_id
        info['episode_steps'] = self.episode_steps
        info['episode_time'] = time.time() - self.episode_start_time
        
        if done:
            self.episode_rewards.append(reward)
            info['episode_reward'] = reward
            info['episode_length'] = self.episode_steps
        
        return observation, reward, done, info
    
    def _state_to_observation(self, state):
        """Convert FireEnv state to gym observation format."""
        # Stack all landscape layers
        obs_layers = []
        
        # Add landscape data layers (8 channels)
        for key in ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']:
            if key in state:
                layer = state[key]
                if isinstance(layer, torch.Tensor):
                    layer = layer.cpu().numpy()
                obs_layers.append(layer)
        
        # Add fireline intensity data (4 channels)
        for direction in ['north', 'south', 'east', 'west']:
            fireline_key = f'fireline_{direction}'
            if fireline_key in state:
                layer = state[fireline_key]
                if isinstance(layer, torch.Tensor):
                    layer = layer.cpu().numpy()
                obs_layers.append(layer)
            else:
                # Create zero layer if fireline data is missing
                obs_layers.append(np.zeros_like(obs_layers[0]))
        
        # Stack all layers
        observation = np.stack(obs_layers, axis=0).astype(np.float32)
        
        # Ensure observation shape is correct
        if observation.shape[0] != 12:
            raise ValueError(f"Expected 12 channels, got {observation.shape[0]}")
        
        return observation
    
    def render(self, mode='human'):
        """Render environment (optional)."""
        return self.fire_env.render(mode)
    
    def close(self):
        """Close environment."""
        pass
    
    def get_stats(self):
        """Get environment statistics."""
        return {
            'env_id': self.env_id,
            'episode_rewards': self.episode_rewards,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'total_episodes': len(self.episode_rewards)
        }


class OptimizedGymVectorizedFireEnv:
    """
    Optimized vectorized environment using gym.vector.SyncVectorEnv that randomly
    samples environments from all available landscapes.
    """
    
    def __init__(self, landscape_data_list: List[Dict], num_parallel_envs: int = 8, 
                 num_simulations: int = 10, max_duration: Optional[int] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize vectorized fire environment.
        
        Args:
            landscape_data_list: List of all available landscape data dictionaries
            num_parallel_envs: Number of parallel environments to run
            num_simulations: Number of simulations per environment
            max_duration: Maximum duration for each simulation
            random_seed: Random seed for reproducible environment selection
        """
        self.landscape_data_list = landscape_data_list
        self.num_parallel_envs = num_parallel_envs
        self.num_simulations = num_simulations
        self.max_duration = max_duration
        self.total_available_landscapes = len(landscape_data_list)
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        print(f"🌍 Initializing OptimizedGymVectorizedFireEnv:")
        print(f"   - Total available landscapes: {self.total_available_landscapes}")
        print(f"   - Number of parallel environments: {num_parallel_envs}")
        print(f"   - Simulations per environment: {num_simulations}")
        
        # Randomly select environments from available landscapes
        self.selected_landscape_indices = self._select_random_environments()
        
        # Create environment functions for SyncVectorEnv
        env_fns = []
        for i in range(num_parallel_envs):
            landscape_idx = self.selected_landscape_indices[i]
            landscape_data = self.landscape_data_list[landscape_idx]
            
            # Create environment function
            env_fn = lambda idx=i, data=landscape_data: SingleFireEnvWrapper(
                landscape_data=copy.deepcopy(data),  # Deep copy to avoid interference
                env_id=idx,
                num_simulations=num_simulations,
                max_duration=max_duration
            )
            env_fns.append(env_fn)
        
        # Initialize SyncVectorEnv
        self.vector_env = SyncVectorEnv(env_fns)
        
        # Environment properties
        self.num_envs = num_parallel_envs
        self.observation_space = self.vector_env.observation_space
        self.action_space = self.vector_env.action_space
        
        # Statistics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.environment_resets = 0
        self.performance_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'environments_used': self.selected_landscape_indices.copy()
        }
        
        print(f"✅ OptimizedGymVectorizedFireEnv initialized successfully!")
        print(f"   - Selected landscape indices: {self.selected_landscape_indices}")
    
    def _select_random_environments(self) -> List[int]:
        """Randomly select environments from available landscapes."""
        if self.num_parallel_envs <= self.total_available_landscapes:
            # If we have enough landscapes, sample without replacement
            selected_indices = random.sample(
                range(self.total_available_landscapes), 
                self.num_parallel_envs
            )
        else:
            # If we need more environments than available landscapes, sample with replacement
            selected_indices = [
                random.randint(0, self.total_available_landscapes - 1)
                for _ in range(self.num_parallel_envs)
            ]
        
        return selected_indices
    
    def reset(self, **kwargs):
        """Reset all environments and return initial observations."""
        self.environment_resets += 1
        
        # Optionally reshuffle environments for diversity
        if self.environment_resets % 10 == 0:  # Every 10 resets
            self._reshuffle_environments()
        
        observations = self.vector_env.reset(**kwargs)
        
        return observations
    
    def step(self, actions):
        """Execute actions in all environments."""
        self.total_steps += len(actions)
        
        observations, rewards, dones, infos = self.vector_env.step(actions)
        
        # Update statistics
        self.episode_count += sum(dones)
        
        # Process info dictionaries
        for i, info in enumerate(infos):
            if 'episode_reward' in info:
                self.performance_stats['total_episodes'] += 1
                # Update rolling average reward
                old_avg = self.performance_stats['average_reward']
                count = self.performance_stats['total_episodes']
                self.performance_stats['average_reward'] = (
                    old_avg * (count - 1) + info['episode_reward']
                ) / count
        
        return observations, rewards, dones, infos
    
    def _reshuffle_environments(self):
        """Reshuffle environment assignments for better diversity."""
        print("🔄 Reshuffling environments for better diversity...")
        
        # Select new random environments
        new_indices = self._select_random_environments()
        
        # Close current environments
        self.vector_env.close()
        
        # Create new environment functions
        env_fns = []
        for i in range(self.num_parallel_envs):
            landscape_idx = new_indices[i]
            landscape_data = self.landscape_data_list[landscape_idx]
            
            env_fn = lambda idx=i, data=landscape_data: SingleFireEnvWrapper(
                landscape_data=copy.deepcopy(data),
                env_id=idx,
                num_simulations=self.num_simulations,
                max_duration=self.max_duration
            )
            env_fns.append(env_fn)
        
        # Initialize new SyncVectorEnv
        self.vector_env = SyncVectorEnv(env_fns)
        
        # Update selected indices
        self.selected_landscape_indices = new_indices
        self.performance_stats['environments_used'].extend(new_indices)
        
        print(f"   - New landscape indices: {new_indices}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            'total_episodes': self.performance_stats['total_episodes'],
            'total_steps': self.total_steps,
            'average_reward': self.performance_stats['average_reward'],
            'environment_resets': self.environment_resets,
            'unique_environments_used': len(set(self.performance_stats['environments_used'])),
            'current_landscape_indices': self.selected_landscape_indices,
            'num_parallel_envs': self.num_parallel_envs,
            'total_available_landscapes': self.total_available_landscapes
        }
    
    def close(self):
        """Close all environments."""
        self.vector_env.close()
        print("🧹 OptimizedGymVectorizedFireEnv closed successfully")
    
    def __del__(self):
        """Cleanup when environment is destroyed."""
        try:
            self.close()
        except:
            pass


class OptimizedExperienceCollector:
    """
    Optimized experience collector that works with gym.vector environments.
    """
    
    def __init__(self, vectorized_env: OptimizedGymVectorizedFireEnv, agent, 
                 collection_batch_size: int = 32):
        self.vectorized_env = vectorized_env
        self.agent = agent
        self.collection_batch_size = collection_batch_size
        
        # Initialize environment states
        self.current_observations = None
        self.experiences_collected = 0
        self.training_steps = 0
        
        print(f"📊 OptimizedExperienceCollector initialized:")
        print(f"   - Collection batch size: {collection_batch_size}")
        print(f"   - Number of environments: {vectorized_env.num_envs}")
    
    def collect_experiences(self, num_steps: int, train_frequency: int = 4) -> Dict:
        """Collect experiences from vectorized environments."""
        start_time = time.time()
        
        # Reset environments if needed
        if self.current_observations is None:
            self.current_observations = self.vectorized_env.reset()
        
        total_reward = 0.0
        episodes_completed = 0
        
        for step in range(num_steps):
            # Get actions for all environments
            actions = []
            for i, obs in enumerate(self.current_observations):
                action = self.agent.act(obs)
                actions.append(action)
            
            # Execute actions
            next_observations, rewards, dones, infos = self.vectorized_env.step(actions)
            
            # Store experiences
            for i in range(len(actions)):
                self.agent.remember(
                    state=self.current_observations[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_observations[i],
                    done=dones[i]
                )
                
                total_reward += rewards[i]
                if dones[i]:
                    episodes_completed += 1
            
            # Update current observations
            self.current_observations = next_observations
            
            # Train periodically
            if step % train_frequency == 0:
                self.agent.replay()
                self.training_steps += 1
            
            self.experiences_collected += len(actions)
        
        collection_time = time.time() - start_time
        
        # Get performance stats
        env_stats = self.vectorized_env.get_performance_stats()
        
        return {
            'mean_reward': total_reward / (num_steps * self.vectorized_env.num_envs),
            'total_steps': num_steps * self.vectorized_env.num_envs,
            'experiences_collected': num_steps * self.vectorized_env.num_envs,
            'episodes_completed': episodes_completed,
            'collection_time': collection_time,
            'experiences_per_second': self.experiences_collected / collection_time if collection_time > 0 else 0,
            'training_steps': self.training_steps,
            'agent_epsilon': self.agent.epsilon,
            'environments_reset': env_stats['environment_resets'],
            'performance_stats': env_stats
        }
    
    def reset(self):
        """Reset the experience collector."""
        self.current_observations = None
        self.experiences_collected = 0
        self.training_steps = 0