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
        
        # Initialize the underlying FireEnv with unpacked landscape data
        self.fire_env = FireEnv(
            slope=landscape_data['slp'],
            aspect=landscape_data['asp'],
            dem=landscape_data['dem'],
            cc=landscape_data['cc'],
            cbd=landscape_data['cbd'],
            cbh=landscape_data['cbh'],
            ch=landscape_data['ch'],
            fuel_model=landscape_data['fbfm'],
            ignite_point=None  # Use default center ignition
        )
        
        # Set simulation parameters on the FireEnv instance
        self.fire_env.num_simulations = num_simulations
        self.fire_env.max_duration = max_duration
        
        # Define action and observation spaces
        self.grid_size = landscape_data['slp'].shape[0]
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # 12 channels: 8 landscape + 4 fireline intensity
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12, self.grid_size, self.grid_size),
            dtype=np.float32
        )
        
        # Track episode statistics
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
    def reset(self, **kwargs):
        """Reset environment and return initial observation and info."""
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
        # Reset the underlying FireEnv (returns burned fire map)
        burned_map = self.fire_env.reset()
        
        # Convert to full observation format (12 channels)
        observation = self._construct_full_observation(burned_map)
        
        # Create info dictionary
        info = {
            'env_id': self.env_id,
            'episode_steps': self.episode_steps,
            'episode_time': 0.0
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return next observation, reward, terminated, truncated, info."""
        self.episode_steps += 1
        
        # Convert discrete action to binary mask
        # action is an integer from 0 to (grid_size^2 - 1)
        # Convert to binary mask where only the selected position is True
        action_mask = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        action_mask[action] = 1.0
        
        # Execute action in underlying FireEnv
        burned_map, reward, done, info = self.fire_env.step(action_mask)
        
        # Convert to full observation format (12 channels)
        observation = self._construct_full_observation(burned_map)
        
        # Add environment-specific info
        info['env_id'] = self.env_id
        info['episode_steps'] = self.episode_steps
        info['episode_time'] = time.time() - self.episode_start_time
        
        # Split done into terminated and truncated for newer gym versions
        # FireEnv episodes end when simulation completes (terminated)
        # No time-based truncation in this case
        terminated = done
        truncated = False
        
        if terminated:
            self.episode_rewards.append(reward)
            info['episode_reward'] = reward
            info['episode_length'] = self.episode_steps
        
        return observation, reward, terminated, truncated, info
    
    def _construct_full_observation(self, burned_map):
        """Construct full 12-channel observation from landscape data and current fire state."""
        obs_layers = []
        
        # Add landscape data layers (8 channels)
        for key in ['slp', 'asp', 'dem', 'cc', 'cbd', 'cbh', 'ch', 'fbfm']:
            if key in self.landscape_data:
                layer = self.landscape_data[key]
                if isinstance(layer, torch.Tensor):
                    layer = layer.cpu().numpy()
                obs_layers.append(layer.astype(np.float32))
        
        # Add fireline intensity data (4 channels)
        for direction in ['north', 'south', 'east', 'west']:
            fireline_key = f'fireline_{direction}'
            if fireline_key in self.landscape_data:
                layer = self.landscape_data[fireline_key]
                if isinstance(layer, torch.Tensor):
                    layer = layer.cpu().numpy()
                obs_layers.append(layer.astype(np.float32))
            else:
                # Create zero layer if fireline data is missing
                obs_layers.append(np.zeros_like(obs_layers[0], dtype=np.float32))
        
        # Stack all layers to create 12-channel observation
        observation = np.stack(obs_layers, axis=0)
        
        # Ensure observation shape is correct
        if observation.shape[0] != 12:
            raise ValueError(f"Expected 12 channels, got {observation.shape[0]}. Available keys: {list(self.landscape_data.keys())}")
        
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
        """Randomly select environments from available landscapes with time-based seeding."""
        print(f"🎲 Selecting {self.num_parallel_envs} environments from {self.total_available_landscapes} available landscapes")
        
        # Use time-based seeding for truly random environment selection
        import time
        time_seed = int(time.time() * 1000000) % 2**32  # Use microseconds for better randomness
        env_random = random.Random(time_seed)
        
        if self.num_parallel_envs <= self.total_available_landscapes:
            # If we have enough landscapes, sample without replacement
            selected_indices = env_random.sample(
                range(self.total_available_landscapes), 
                self.num_parallel_envs
            )
            print(f"   - Using sampling without replacement")
        else:
            # If we need more environments than available landscapes, sample with replacement
            selected_indices = [
                env_random.randint(0, self.total_available_landscapes - 1)
                for _ in range(self.num_parallel_envs)
            ]
            print(f"   - Using sampling with replacement")
        
        print(f"   - Selected indices: {selected_indices}")
        return selected_indices
    
    def reset(self, **kwargs):
        """Reset all environments and return initial observations."""
        self.environment_resets += 1
        
        # Randomly reshuffle environments at EVERY reset for maximum diversity
        print(f"🔄 Episode {self.environment_resets}: Randomly selecting {self.num_parallel_envs} environments from {self.total_available_landscapes} landscapes")
        
        # Store previous selection for comparison
        previous_selection = getattr(self, 'selected_landscape_indices', None)
        
        # Reshuffle environments
        self._reshuffle_environments()
        
        # Verify that environments actually changed
        if previous_selection is not None:
            if self.selected_landscape_indices != previous_selection:
                print(f"✅ Environment selection changed: {previous_selection} → {self.selected_landscape_indices}")
            else:
                print(f"⚠️ Environment selection unchanged: {self.selected_landscape_indices}")
        
        # SyncVectorEnv.reset() returns (observations, infos) tuple
        observations, infos = self.vector_env.reset(**kwargs)
        
        # Fix: Ensure observations are properly formatted as arrays
        observations = self._ensure_observations_are_arrays(observations)
        
        return observations
    
    def step(self, actions):
        """Execute actions in all environments."""
        self.total_steps += len(actions)
        
        # SyncVectorEnv.step() returns 5 values: obs, rewards, terminated, truncated, infos
        observations, rewards, terminated, truncated, infos = self.vector_env.step(actions)
        
        # Fix: Ensure observations are properly formatted as arrays
        observations = self._ensure_observations_are_arrays(observations)
        
        # Combine terminated and truncated into done for backward compatibility
        dones = terminated | truncated
        
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
    
    def _create_env_function(self, env_id: int, landscape_data: Dict):
        """Create environment function with proper closure."""
        def env_fn():
            return SingleFireEnvWrapper(
                landscape_data=copy.deepcopy(landscape_data),
                env_id=env_id,
                num_simulations=self.num_simulations,
                max_duration=self.max_duration
            )
        return env_fn
    
    def _reshuffle_environments(self):
        """Reshuffle environment assignments for maximum diversity every episode."""
        # Select new random environments
        new_indices = self._select_random_environments()
        
        # Close current environments
        self.vector_env.close()
        
        # Create new environment functions with proper closure
        env_fns = []
        for i in range(self.num_parallel_envs):
            landscape_idx = new_indices[i]
            landscape_data = self.landscape_data_list[landscape_idx]
            
            # Use proper function closure to avoid lambda capture issues
            env_fn = self._create_env_function(i, landscape_data)
            env_fns.append(env_fn)
        
        # Initialize new SyncVectorEnv
        self.vector_env = SyncVectorEnv(env_fns)
        
        # Update selected indices and performance stats
        self.selected_landscape_indices = new_indices
        self.performance_stats['environments_used'].extend(new_indices)
        
        # Log the selection
        print(f"   - Selected landscapes: {new_indices}")
        unique_landscapes = len(set(self.performance_stats['environments_used']))
        print(f"   - Unique landscapes used so far: {unique_landscapes}/{self.total_available_landscapes}")
    
    def _ensure_observations_are_arrays(self, observations):
        """Ensure observations are properly formatted as arrays, not dictionaries."""
        if isinstance(observations, (list, tuple)):
            processed_observations = []
            for i, obs in enumerate(observations):
                if isinstance(obs, dict):
                    # If observation is a dict, extract the actual array
                    if 'observation' in obs:
                        processed_observations.append(obs['observation'])
                    else:
                        print(f"⚠️ Warning: Dict observation {i} has no 'observation' key. Keys: {list(obs.keys())}")
                        # Try to find the array in the dict
                        array_keys = [k for k, v in obs.items() if hasattr(v, 'shape')]
                        if array_keys:
                            processed_observations.append(obs[array_keys[0]])
                        else:
                            raise ValueError(f"Cannot extract array from dict observation {i}: {obs}")
                else:
                    processed_observations.append(obs)
            return processed_observations
        else:
            return observations

    def get_performance_stats(self) -> Dict:
        """Get performance statistics with enhanced environment diversity tracking."""
        environments_used = self.performance_stats['environments_used']
        unique_environments = set(environments_used)
        
        # Calculate environment usage frequency
        usage_frequency = {}
        for env_idx in environments_used:
            usage_frequency[env_idx] = usage_frequency.get(env_idx, 0) + 1
        
        # Calculate diversity metrics
        diversity_percentage = (len(unique_environments) / self.total_available_landscapes) * 100
        
        return {
            'total_episodes': self.performance_stats['total_episodes'],
            'total_steps': self.total_steps,
            'average_reward': self.performance_stats['average_reward'],
            'environment_resets': self.environment_resets,
            'unique_environments_used': len(unique_environments),
            'diversity_percentage': diversity_percentage,
            'environment_usage_frequency': usage_frequency,
            'current_landscape_indices': self.selected_landscape_indices,
            'num_parallel_envs': self.num_parallel_envs,
            'total_available_landscapes': self.total_available_landscapes,
            'environments_per_episode': self.num_parallel_envs,
            'total_environment_selections': len(environments_used)
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
        
        # Fix: Ensure observations are properly formatted as arrays
        self.current_observations = self._ensure_observations_are_arrays(self.current_observations)
        
        total_reward = 0.0
        episodes_completed = 0
        
        for step in range(num_steps):
            # Get actions for all environments
            actions = []
            for i, obs in enumerate(self.current_observations):
                # Debug: Check if observation is in correct format
                if not hasattr(obs, 'shape'):
                    print(f"⚠️ Warning: Observation {i} has no shape attribute. Type: {type(obs)}")
                    if isinstance(obs, dict):
                        print(f"   Dict keys: {list(obs.keys())}")
                        # Try to extract array from dict if it's wrapped
                        if 'observation' in obs:
                            obs = obs['observation']
                        else:
                            print(f"   Cannot extract array from dict observation!")
                            continue
                
                action = self.agent.act(obs)
                actions.append(action)
            
            # Execute actions
            next_observations, rewards, dones, infos = self.vectorized_env.step(actions)
            
            # Fix: Ensure next observations are properly formatted as arrays
            next_observations = self._ensure_observations_are_arrays(next_observations)
            
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
    
    def _ensure_observations_are_arrays(self, observations):
        """Ensure observations are properly formatted as arrays, not dictionaries."""
        if isinstance(observations, (list, tuple)):
            processed_observations = []
            for i, obs in enumerate(observations):
                if isinstance(obs, dict):
                    # If observation is a dict, extract the actual array
                    if 'observation' in obs:
                        processed_observations.append(obs['observation'])
                    else:
                        print(f"⚠️ Warning: Dict observation {i} has no 'observation' key. Keys: {list(obs.keys())}")
                        # Try to find the array in the dict
                        array_keys = [k for k, v in obs.items() if hasattr(v, 'shape')]
                        if array_keys:
                            processed_observations.append(obs[array_keys[0]])
                        else:
                            raise ValueError(f"Cannot extract array from dict observation {i}: {obs}")
                else:
                    processed_observations.append(obs)
            return processed_observations
        else:
            return observations
    
    def reset(self):
        """Reset the experience collector."""
        self.current_observations = None
        self.experiences_collected = 0
        self.training_steps = 0