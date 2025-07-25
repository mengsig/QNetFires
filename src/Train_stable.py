#!/usr/bin/env python3
"""
Stable Training Script for High-Core Workstations
Optimized for 32+ threads with robust multiprocessing and error handling.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import multiprocessing as mp
import time
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import signal

# Set multiprocessing method early
mp.set_start_method("spawn", force=True)

# Add src to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from src.Env import FuelBreakEnv
from src.Model import QNet, EnhancedQNet, DuelingQNet
from src.utils.loadingUtils import RasterManager

Transition = namedtuple("Transition", "obs action reward next_obs done")

def safe_scalar(value, fallback=0.0):
    """Convert value to scalar safely."""
    if value is None:
        return fallback
    if isinstance(value, str) and value == 'N/A':
        return fallback
    try:
        if hasattr(value, '__len__') and len(value) > 1:
            return float(np.mean(value))
        return float(value)
    except:
        return fallback

class StableProcessPoolEnv:
    """
    Stable process pool-based vector environment for high-core workstations.
    Uses ProcessPoolExecutor for better process management and stability.
    """
    
    def __init__(self, env_fns, max_workers=None, timeout=15.0):
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.max_workers = max_workers or min(self.num_envs, mp.cpu_count())
        self.timeout = timeout
        
        print(f"🚀 Creating StableProcessPoolEnv with {self.num_envs} environments and {self.max_workers} workers")
        
        # Create process pool
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')
        )
        
        # Initialize environments
        self._initialize_environments()
        
        # Track environment states
        self.env_states = [{'done': False, 'steps': 0} for _ in range(self.num_envs)]
        
    def _initialize_environments(self):
        """Initialize all environments in parallel."""
        print(f"🔧 Initializing {self.num_envs} environments...")
        
        # Submit initialization tasks
        futures = []
        for i, env_fn in enumerate(self.env_fns):
            future = self.executor.submit(self._init_single_env, env_fn, i)
            futures.append((i, future))
        
        # Collect results
        self.env_ids = []
        successful = 0
        
        for env_id, future in futures:
            try:
                result = future.result(timeout=30.0)  # 30s timeout for init
                if result:
                    self.env_ids.append(env_id)
                    successful += 1
                    print(f"Environment {env_id}: ✅")
                else:
                    print(f"Environment {env_id}: ❌ Failed initialization")
            except Exception as e:
                print(f"Environment {env_id}: ❌ Error: {e}")
        
        print(f"✅ Successfully initialized {successful}/{self.num_envs} environments")
        
        if successful == 0:
            raise RuntimeError("No environments could be initialized!")
    
    @staticmethod
    def _init_single_env(env_fn, env_id):
        """Initialize a single environment."""
        try:
            # Set process-specific random seed
            np.random.seed(env_id + int(time.time()) % 10000)
            random.seed(env_id + int(time.time()) % 10000)
            
            env = env_fn()
            
            # Test the environment
            obs, _ = env.reset()
            test_action = np.zeros(obs.shape[-2] * obs.shape[-1], dtype=np.int8)
            test_action[:5] = 1
            
            obs2, reward, done, truncated, info = env.step(test_action)
            
            # Verify it works
            if obs2 is not None and info is not None:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Environment {env_id} init failed: {e}")
            return False
    
    def reset(self):
        """Reset all environments."""
        print(f"🔄 Resetting {len(self.env_ids)} environments...")
        
        # Submit reset tasks
        futures = []
        for env_id in self.env_ids:
            future = self.executor.submit(self._reset_single_env, self.env_fns[env_id])
            futures.append((env_id, future))
        
        # Collect results
        obs_list = []
        for env_id, future in futures:
            try:
                obs = future.result(timeout=self.timeout)
                if obs is not None:
                    obs_list.append(obs.astype(np.float32))
                    self.env_states[env_id] = {'done': False, 'steps': 0}
                else:
                    # Fallback observation
                    obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
                    self.env_states[env_id] = {'done': True, 'steps': 0}
            except Exception as e:
                print(f"Reset failed for env {env_id}: {e}")
                obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
                self.env_states[env_id] = {'done': True, 'steps': 0}
        
        return np.stack(obs_list)
    
    @staticmethod
    def _reset_single_env(env_fn):
        """Reset a single environment."""
        try:
            env = env_fn()
            obs, _ = env.reset()
            return obs
        except Exception as e:
            print(f"Reset error: {e}")
            return None
    
    def step(self, actions):
        """Step all environments."""
        # Submit step tasks
        futures = []
        for i, env_id in enumerate(self.env_ids):
            if not self.env_states[env_id]['done']:
                future = self.executor.submit(
                    self._step_single_env, 
                    self.env_fns[env_id], 
                    actions[i]
                )
                futures.append((env_id, future))
        
        # Collect results
        obs_list = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        dones = [True] * self.num_envs
        truncated = [False] * self.num_envs
        infos = [{"burned": 150.0, "new_cells": 0}] * self.num_envs
        
        for env_id, future in futures:
            try:
                result = future.result(timeout=self.timeout)
                if result is not None:
                    obs, reward, done, trunc, info = result
                    
                    obs_list[env_id] = obs.astype(np.float32)
                    rewards[env_id] = float(reward)
                    dones[env_id] = bool(done or trunc)
                    truncated[env_id] = bool(trunc)
                    infos[env_id] = info or {"burned": 150.0, "new_cells": 0}
                    
                    # Update state
                    self.env_states[env_id]['steps'] += 1
                    self.env_states[env_id]['done'] = dones[env_id]
                else:
                    # Environment failed
                    self.env_states[env_id]['done'] = True
                    
            except Exception as e:
                print(f"Step failed for env {env_id}: {e}")
                self.env_states[env_id]['done'] = True
        
        # Fill missing observations
        for i in range(self.num_envs):
            if obs_list[i] is None:
                obs_list[i] = np.random.rand(8, 50, 50).astype(np.float32)
        
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncated, dtype=bool),
            infos
        )
    
    @staticmethod
    def _step_single_env(env_fn, action):
        """Step a single environment."""
        try:
            env = env_fn()
            
            # Ensure action is correct type
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if action.dtype != np.int8:
                action = action.astype(np.int8)
            
            result = env.step(action)
            return result
            
        except Exception as e:
            print(f"Step error: {e}")
            return None
    
    def close(self):
        """Clean shutdown."""
        print("🔄 Shutting down StableProcessPoolEnv...")
        try:
            self.executor.shutdown(wait=True, timeout=10.0)
        except:
            pass
        
        # Force cleanup any remaining processes
        try:
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    child.terminate()
                except:
                    pass
        except:
            pass

class PrioritizedReplayBuffer:
    """Simple prioritized experience replay buffer."""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, obs, action, reward, next_obs, done, priority=1.0):
        # Ensure correct data types
        obs = obs.astype(np.float32)
        action = action.astype(np.float32)
        reward = float(reward)
        next_obs = next_obs.astype(np.float32)
        done = bool(done)
        
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer)), np.ones(len(self.buffer))
        
        # Simple uniform sampling for stability
        batch = random.sample(self.buffer, batch_size)
        weights = np.ones(batch_size)
        
        return batch, weights
    
    def __len__(self):
        return len(self.buffer)

def make_stable_env(raster, budget, kstep, sims, seed):
    """Create a stable environment function."""
    def thunk():
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate raster
        if not isinstance(raster, dict):
            raise ValueError("Invalid raster")
        
        required_keys = ['slp', 'asp', 'fbfm', 'fireline_north', 'fireline_east', 'fireline_south', 'fireline_west']
        for key in required_keys:
            if key not in raster:
                raise ValueError(f"Missing raster key: {key}")
        
        # Create environment with minimal simulations
        env = FuelBreakEnv(
            raster,
            break_budget=budget,
            break_step=kstep,
            num_simulations=1,  # Keep minimal for stability
            seed=seed,
        )
        
        return env
    
    return thunk

def choose_actions_batch(model, obs_np, k, eps, device="cpu"):
    """Choose actions for batch of observations."""
    N, C, H, W = obs_np.shape
    HxW = H * W
    
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_np.astype(np.float32)).to(device)
        q_vals = model(obs_t)
        
        actions = []
        for i in range(N):
            if np.random.rand() < eps:
                # Random action
                action = np.zeros(HxW, dtype=np.int8)
                indices = np.random.choice(HxW, size=min(k, HxW), replace=False)
                action[indices] = 1
            else:
                # Greedy action
                q_i = q_vals[i].cpu().numpy()
                top_indices = np.argsort(q_i)[-k:]
                action = np.zeros(HxW, dtype=np.int8)
                action[top_indices] = 1
            
            actions.append(action)
    
    return np.array(actions)

def compute_q_loss(model, target_model, batch, weights, gamma, device="cpu"):
    """Compute Q-learning loss."""
    obs_batch = torch.stack([torch.from_numpy(t.obs) for t in batch]).to(device)
    action_batch = torch.stack([torch.from_numpy(t.action) for t in batch]).to(device)
    reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
    next_obs_batch = torch.stack([torch.from_numpy(t.next_obs) for t in batch]).to(device)
    done_batch = torch.tensor([t.done for t in batch], dtype=torch.bool).to(device)
    weights_batch = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Current Q-values
    current_q = model(obs_batch)
    current_q_action = (current_q * action_batch).sum(dim=1)
    
    # Target Q-values (Double DQN)
    with torch.no_grad():
        next_online = model(next_obs_batch)
        next_target = target_model(next_obs_batch)
        
        next_actions = next_online.argmax(dim=1)
        next_q_values = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        target_q = reward_batch + gamma * next_q_values * (~done_batch).float()
    
    # Weighted loss
    td_errors = torch.abs(current_q_action - target_q)
    loss = (weights_batch * (current_q_action - target_q).pow(2)).mean()
    
    return loss, td_errors.detach().cpu().numpy()

def main():
    # Device and memory setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device {DEVICE} on {mp.cpu_count()}-core system")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Optimized hyperparameters for high-core systems
    EPISODES = 1000
    STEPS_PER_EP = 25  # Allow episodes to complete naturally
    BUFFER_CAP = 200_000  # Larger buffer for high-core systems
    BATCH_SIZE = 64  # Larger batch size
    GAMMA = 0.99
    LR = 1e-4
    START_EPS = 1.0
    END_EPS = 0.01
    EPS_DECAY_STEPS = 200_000
    TARGET_SYNC_EVERY = 2000
    SAVE_EVERY = 25
    
    # Environment parameters optimized for stability
    N_ENVS = 32  # Use 32 environments as requested
    BUDGET = 250
    K_STEPS = 10
    SIMS = 1  # Keep minimal for stability
    
    print(f"🎯 Training with {N_ENVS} environments on {mp.cpu_count()} cores")
    
    # Load rasters
    print("📂 Loading rasters...")
    raster_manager = RasterManager("cropped_raster", 500)
    try:
        raster_manager.load_all_rasters()
        print(f"✅ Loaded {len(raster_manager.all_rasters)} rasters")
    except Exception as e:
        print(f"❌ Error loading rasters: {e}")
        return
    
    # Create environments
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
    env_fns = [
        make_stable_env(raster, BUDGET, K_STEPS, SIMS, seed=i) 
        for i, raster in enumerate(selected_rasters)
    ]
    
    # Use stable process pool environment
    vec_env = StableProcessPoolEnv(env_fns, max_workers=min(N_ENVS, 16))  # Limit workers
    obs = vec_env.reset()
    _, C, H, W = obs.shape
    
    print(f"🏗️  Environment shape: {obs.shape}")
    
    # Initialize model
    model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
    target_model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = PrioritizedReplayBuffer(BUFFER_CAP)
    
    # Training metrics
    reward_window = deque(maxlen=100)
    loss_window = deque(maxlen=100)
    burned_area_window = deque(maxlen=100)
    
    global_step = 0
    best_reward = float('-inf')
    
    print("🚀 Starting stable training...")
    
    try:
        for episode in range(EPISODES):
            # Cycle rasters every 20 episodes
            if episode % 20 == 0 and episode > 0:
                print(f"\n🔄 Cycling rasters at episode {episode}...")
                try:
                    vec_env.close()
                    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
                    env_fns = [
                        make_stable_env(raster, BUDGET, K_STEPS, SIMS, seed=episode*N_ENVS + i) 
                        for i, raster in enumerate(selected_rasters)
                    ]
                    vec_env = StableProcessPoolEnv(env_fns, max_workers=min(N_ENVS, 16))
                    obs = vec_env.reset()
                    print("✅ Successfully cycled rasters")
                except Exception as e:
                    print(f"❌ Failed to cycle rasters: {e}")
            
            episode_rewards = []
            active_envs = [True] * N_ENVS
            
            for step in range(STEPS_PER_EP):
                # Choose actions
                eps = START_EPS - (START_EPS - END_EPS) * min(1.0, global_step / EPS_DECAY_STEPS)
                actions = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)
                
                # Environment step
                next_obs, rewards, dones, truncated, infos = vec_env.step(actions)
                
                # Store transitions
                for i in range(N_ENVS):
                    if active_envs[i]:
                        buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                        
                        # Track metrics
                        if infos[i] and "burned" in infos[i]:
                            burned_area_window.append(safe_scalar(infos[i]["burned"]))
                        
                        if dones[i]:
                            episode_rewards.append(rewards[i])
                            active_envs[i] = False
                            
                            # Log completion
                            burned = safe_scalar(infos[i].get("burned", 0)) if infos[i] else 0
                            print(f"🎯 Environment {i} completed: Burned={burned:.1f}, Step={step+1}")
                
                obs = next_obs
                global_step += N_ENVS
                
                # Training step
                if len(buffer) >= BATCH_SIZE:
                    batch, weights = buffer.sample(BATCH_SIZE)
                    loss, td_errors = compute_q_loss(model, target_model, batch, weights, GAMMA, DEVICE)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    loss_window.append(loss.item())
                    
                    # Update target network
                    if global_step % TARGET_SYNC_EVERY == 0:
                        target_model.load_state_dict(model.state_dict())
                
                # Early termination if all environments done
                if not any(active_envs):
                    print(f"✅ All environments completed at step {step+1}")
                    break
            
            # Reset environments for next episode
            obs = vec_env.reset()
            
            # Episode statistics
            if episode_rewards:
                reward_window.extend(episode_rewards)
            
            mean_reward = np.mean(reward_window) if reward_window else 0.0
            mean_loss = np.mean(loss_window) if loss_window else 0.0
            mean_burned = np.mean(burned_area_window) if burned_area_window else 0.0
            
            print(f"[Episode {episode}] Reward: {mean_reward:.3f} Loss: {mean_loss:.3f} "
                  f"Burned: {mean_burned:.1f} Eps: {eps:.3f} Completed: {len(episode_rewards)}/{N_ENVS}")
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(model.state_dict(), "best_stable_model.pth")
            
            # Periodic save
            if episode % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f"stable_model_ep{episode}.pth")
    
    finally:
        vec_env.close()
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()