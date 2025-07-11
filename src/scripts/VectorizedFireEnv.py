import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
import threading
from typing import List, Dict, Tuple, Any
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.scripts.FireEnv import FireEnv
from src.scripts.Simulate import Simulate


class VectorizedFireEnv:
    """
    Vectorized environment wrapper that runs multiple FireEnv instances in parallel.
    
    This allows the agent to interact with multiple fire simulation environments
    simultaneously, providing much faster training and more diverse experiences.
    """
    
    def __init__(self, landscape_data_list: List[Dict], num_envs: int = None, 
                 method: str = 'threading', max_workers: int = None,
                 num_simulations: int = 10, max_duration: int = None):
        """
        Initialize vectorized fire environment.
        
        Args:
            landscape_data_list: List of landscape data dictionaries
            num_envs: Number of parallel environments (defaults to len(landscape_data_list))
            method: 'threading', 'multiprocessing', or 'sequential'
            max_workers: Maximum number of worker threads/processes
            num_simulations: Number of simulations to run in run_many_simulations
            max_duration: Maximum duration for each simulation (minutes)
        """
        self.landscape_data_list = landscape_data_list
        self.num_envs = num_envs or len(landscape_data_list)
        self.method = method
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.num_simulations = num_simulations
        self.max_duration = max_duration
        
        print(f"🚀 VectorizedFireEnv configuration:")
        print(f"   - Requested environments: {num_envs}")
        print(f"   - Available landscapes: {len(landscape_data_list)}")
        print(f"   - Final num_envs: {self.num_envs}")
        print(f"   - Method: {method}")
        print(f"   - Max workers: {self.max_workers}")
        print(f"   - Fire simulations per step: {num_simulations}")
        print(f"   - Max duration: {max_duration} minutes")
        
        # Create environments
        self.envs = []
        for i in range(self.num_envs):
            landscape_data = landscape_data_list[i % len(landscape_data_list)]
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
            # Store configuration
            env.num_simulations = self.num_simulations
            env.max_duration = self.max_duration
            # Ensure environment is reset for multiprocessing
            env.reset()
            self.envs.append(env)
        
        # Initialize executor based on method
        if method == 'threading':
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif method == 'multiprocessing':
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None  # Sequential execution
        
        print(f"Initialized VectorizedFireEnv with {self.num_envs} environments using {method}")
        
        # Environment state
        self.current_states = None
        self.current_fuel_breaks = [np.zeros((env.H, env.W), dtype=bool) for env in self.envs]
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        
    def reset(self) -> List[np.ndarray]:
        """Reset all environments and return initial observations."""
        if self.method == 'sequential':
            self.current_states = [env.reset() for env in self.envs]
        else:
            # Parallel reset
            futures = [self.executor.submit(env.reset) for env in self.envs]
            self.current_states = [future.result() for future in futures]
        
        # Reset episode tracking
        self.current_fuel_breaks = [np.zeros((env.H, env.W), dtype=bool) for env in self.envs]
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        
        return self.current_states
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """
        Step all environments with the given actions.
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        
        if self.method == 'sequential':
            # Sequential execution
            results = []
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                result = env.step(action)
                results.append(result)
        else:
            # TRUE PARALLEL EXECUTION - All environments run simultaneously
            # Each environment will also run its fire simulations in parallel
            futures = []
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                future = self.executor.submit(env.step, action)
                futures.append(future)
            
            # Wait for all environments to complete
            results = [future.result() for future in futures]
        
        # Unpack results
        observations, rewards, dones, infos = zip(*results)
        
        # Update episode tracking
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.episode_rewards[i] += reward
            self.episode_steps[i] += 1
            
            if done:
                # Reset this environment
                if self.method == 'sequential':
                    self.current_states[i] = self.envs[i].reset()
                else:
                    # For multiprocessing, ensure environment is properly reset
                    future = self.executor.submit(self.envs[i].reset)
                    self.current_states[i] = future.result()
                
                self.current_fuel_breaks[i] = np.zeros((self.envs[i].H, self.envs[i].W), dtype=bool)
                self.episode_rewards[i] = 0.0
                self.episode_steps[i] = 0
        
        self.current_states = list(observations)
        return list(observations), list(rewards), list(dones), list(infos)
    
    def step_async(self, actions: List[np.ndarray]) -> List:
        """
        Submit step operations asynchronously and return futures.
        Useful for advanced async training patterns.
        """
        if self.executor is None:
            raise ValueError("Async stepping requires threading or multiprocessing method")
        
        futures = []
        for env, action in zip(self.envs, actions):
            future = self.executor.submit(env.step, action)
            futures.append(future)
        
        return futures
    
    def wait_for_results(self, futures: List) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """Wait for async step results and process them."""
        results = [future.result() for future in futures]
        observations, rewards, dones, infos = zip(*results)
        
        # Update tracking (similar to step method)
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.episode_rewards[i] += reward
            self.episode_steps[i] += 1
            
            if done:
                if self.method == 'sequential':
                    self.current_states[i] = self.envs[i].reset()
                else:
                    future = self.executor.submit(self.envs[i].reset)
                    self.current_states[i] = future.result()
                
                self.current_fuel_breaks[i] = np.zeros((self.envs[i].H, self.envs[i].W), dtype=bool)
                self.episode_rewards[i] = 0.0
                self.episode_steps[i] = 0
        
        self.current_states = list(observations)
        return list(observations), list(rewards), list(dones), list(infos)
    
    def get_episode_stats(self) -> Dict:
        """Get statistics about current episodes."""
        # Note: episode_rewards get reset after each step since FireEnv episodes are single-step
        # So we calculate mean from the current episode rewards which may be mostly zeros
        current_mean = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_steps': self.episode_steps.copy(),
            'mean_reward': current_mean,
            'mean_steps': np.mean(self.episode_steps),
            'active_environments': self.num_envs
        }
    
    def close(self):
        """Clean up resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        
        for env in self.envs:
            env.close()


class ParallelExperienceCollector:
    """
    Collects experiences from multiple environments in parallel for DQN training.
    
    This class manages the interaction between the DQN agent and vectorized environments,
    ensuring efficient experience collection and proper memory management.
    """
    
    def __init__(self, vectorized_env: VectorizedFireEnv, agent, 
                 collection_batch_size: int = 32, 
                 experience_buffer_size: int = 1000):
        """
        Initialize parallel experience collector.
        
        Args:
            vectorized_env: Vectorized environment instance
            agent: DQN agent instance
            collection_batch_size: Number of experiences to collect before training
            experience_buffer_size: Size of local experience buffer
        """
        self.vectorized_env = vectorized_env
        self.agent = agent
        self.collection_batch_size = collection_batch_size
        self.experience_buffer_size = experience_buffer_size
        
        # Local experience buffer for batched training
        self.experience_buffer = []
        self.total_experiences_collected = 0
        
        # Performance tracking
        self.collection_times = []
        self.step_times = []
        
    def collect_experiences(self, num_steps: int, train_frequency: int = 4) -> Dict:
        """
        Collect experiences from parallel environments.
        
        Args:
            num_steps: Number of environment steps to collect
            train_frequency: Train agent every N collected experiences
            
        Returns:
            Dictionary with collection statistics
        """
        start_time = time.time()
        
        # Reset environments
        observations = self.vectorized_env.reset()
        current_fuel_breaks = [np.zeros((env.H, env.W), dtype=bool) 
                              for env in self.vectorized_env.envs]
        
        stats = {
            'total_steps': 0,
            'total_rewards': [],
            'training_steps': 0,
            'environments_reset': 0,
            'average_step_time': 0.0
        }
        
        for step in range(num_steps):
            step_start = time.time()
            
            # Get actions from agent for all environments
            actions = []
            states_tensors = []
            
            for i, obs in enumerate(observations):
                # Convert observation to landscape format for agent
                landscape_data = self._obs_to_landscape_data(obs, i)
                state_tensor = self.agent.preprocess_state(landscape_data)
                states_tensors.append(state_tensor)
                
                # Get action from agent
                action = self.agent.act(state_tensor, current_fuel_breaks[i])
                
                # Update current fuel breaks for tracking
                row = action // self.vectorized_env.envs[i].W
                col = action % self.vectorized_env.envs[i].W
                current_fuel_breaks[i][row, col] = True
                
                # Create action for FireEnv: full fuel break pattern as flat binary array
                # FireEnv expects the COMPLETE fuel break pattern, not just the new placement
                action_for_env = current_fuel_breaks[i].flatten().astype(int)
                actions.append(action_for_env)
            
            # Step all environments
            next_observations, rewards, dones, infos = self.vectorized_env.step(actions)
            
            # Debug: Print rewards for first few steps (show all environments)
            if step < 3:
                print(f"   🎯 Step {step} rewards: {rewards}")
                for i, info in enumerate(infos):
                    if 'acres_burned' in info:
                        print(f"      Env {i}: {info['acres_burned']:.1f} acres burned, reward: {rewards[i]:.2f}")
                    else:
                        print(f"      Env {i}: No acres_burned info, reward: {rewards[i]:.2f}")
            
            # Store experiences
            for i in range(self.vectorized_env.num_envs):
                experience = {
                    'state': states_tensors[i].cpu(),
                    'action': self._mask_to_action(actions[i]),
                    'reward': rewards[i],
                    'next_state': states_tensors[i].cpu(),  # Could be updated with next obs
                    'done': dones[i],
                    'env_id': i
                }
                
                self.experience_buffer.append(experience)
                self.total_experiences_collected += 1
                
                # Add to agent's memory
                self.agent.remember(
                    state=states_tensors[i],
                    action=self._mask_to_action(actions[i]),
                    reward=rewards[i],
                    next_state=states_tensors[i],  # Same state, different fuel breaks
                    done=dones[i]
                )
            
            # Train agent periodically
            if (step + 1) % train_frequency == 0 and len(self.agent.memory) >= self.agent.batch_size:
                self.agent.replay()
                stats['training_steps'] += 1
            
            # Update observations and reset tracking
            observations = next_observations
            for i, done in enumerate(dones):
                if done:
                    current_fuel_breaks[i] = np.zeros((self.vectorized_env.envs[i].H, 
                                                     self.vectorized_env.envs[i].W), dtype=bool)
                    stats['environments_reset'] += 1
            
            # Record step time
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            stats['total_steps'] += 1
            stats['total_rewards'].extend(rewards)
        
        # Final statistics
        collection_time = time.time() - start_time
        self.collection_times.append(collection_time)
        
        stats['collection_time'] = collection_time
        stats['average_step_time'] = np.mean(self.step_times[-num_steps:]) if self.step_times else 0
        stats['experiences_collected'] = len(self.experience_buffer)
        stats['total_experiences'] = self.total_experiences_collected
        stats['mean_reward'] = np.mean(stats['total_rewards']) if stats['total_rewards'] else 0
        
        # Debug output for mean reward calculation
        if stats['total_rewards']:
            print(f"   📊 Reward calculation debug:")
            print(f"      - Total rewards collected: {len(stats['total_rewards'])}")
            print(f"      - Reward range: [{min(stats['total_rewards']):.1f}, {max(stats['total_rewards']):.1f}]")
            print(f"      - Mean reward: {stats['mean_reward']:.2f}")
        
        return stats
    
    def _obs_to_landscape_data(self, obs: np.ndarray, env_id: int) -> Dict:
        """Convert environment observation back to landscape data format."""
        # Use the stored landscape data for this environment
        landscape_data = self.vectorized_env.landscape_data_list[env_id % len(self.vectorized_env.landscape_data_list)]
        return landscape_data
    

    
    def _mask_to_action(self, fuel_break_mask: np.ndarray) -> int:
        """Convert fuel break mask back to action index (for memory storage)."""
        # Find the last added fuel break (this is approximate)
        # In practice, you'd want to track this more precisely
        positions = np.where(fuel_break_mask.reshape(-1) == 1)[0]
        return positions[-1] if len(positions) > 0 else 0
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the collection process."""
        return {
            'total_experiences_collected': self.total_experiences_collected,
            'average_collection_time': np.mean(self.collection_times) if self.collection_times else 0,
            'average_step_time': np.mean(self.step_times) if self.step_times else 0,
            'experiences_per_second': (self.total_experiences_collected / 
                                     sum(self.collection_times)) if self.collection_times else 0,
            'buffer_size': len(self.experience_buffer),
            'agent_memory_size': len(self.agent.memory)
        }
    
    def clear_local_buffer(self):
        """Clear the local experience buffer."""
        self.experience_buffer.clear()


# Example usage and testing
if __name__ == "__main__":
    # Test the vectorized environment
    from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
    from src.scripts.DQNAgent import DQNAgent
    
    print("Testing VectorizedFireEnv...")
    
    # Generate sample landscapes
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(3)
    
    landscape_data_list = []
    for i in range(3):
        landscape_data = memory_loader.load_landscape_data(i)
        landscape_data_list.append(landscape_data)
    
    # Test different methods
    for method in ['sequential', 'threading']:
        print(f"\nTesting {method} method...")
        
        # Create vectorized environment
        vec_env = VectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_envs=3,
            method=method,
            max_workers=2
        )
        
        # Test basic operations
        observations = vec_env.reset()
        print(f"Reset: {len(observations)} observations")
        
        # Test stepping
        actions = [np.random.randint(0, 625, size=625) for _ in range(3)]  # Random fuel break patterns
        start_time = time.time()
        next_obs, rewards, dones, infos = vec_env.step(actions)
        step_time = time.time() - start_time
        
        print(f"Step time: {step_time:.3f}s")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        
        # Test with agent
        agent = DQNAgent(input_channels=12, grid_size=25, batch_size=4)
        collector = ParallelExperienceCollector(vec_env, agent, collection_batch_size=8)
        
        print("Testing experience collection...")
        stats = collector.collect_experiences(num_steps=5, train_frequency=2)
        print(f"Collection stats: {stats}")
        
        perf_stats = collector.get_performance_stats()
        print(f"Performance stats: {perf_stats}")
        
        vec_env.close()
        
    print("\nVectorizedFireEnv testing completed!")
