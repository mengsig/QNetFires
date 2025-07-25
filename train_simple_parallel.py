#!/usr/bin/env python3
"""
Simple Parallel Training Script
Uses multiprocessing.Pool for stable parallel execution without AsyncVectorEnv issues.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append('src')

from Env import FuelBreakEnv
from Model import QNet, EnhancedQNet, DuelingQNet
from utils.loadingUtils import RasterManager

Transition = namedtuple("Transition", "obs action reward next_obs done")

def safe_scalar(value, fallback=0.0):
    """Convert value to scalar safely."""
    if value is None:
        return fallback
    try:
        return float(value)
    except:
        return fallback

def step_single_env(args):
    """Step a single environment - used for multiprocessing.Pool"""
    env, action, env_id = args
    try:
        result = env.step(action)
        return env_id, result, None
    except Exception as e:
        # Return error result
        dummy_obs = np.random.rand(8, 50, 50).astype(np.float32)
        error_result = (dummy_obs, -0.1, True, False, {"burned": 150.0, "new_cells": 0})
        return env_id, error_result, str(e)

class SimpleParallelEnv:
    """Simple parallel environment using multiprocessing.Pool"""
    
    def __init__(self, env_fns, n_workers=None):
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.n_workers = n_workers or min(self.num_envs, mp.cpu_count())
        
        # Create environments
        print(f"Creating {self.num_envs} environments...")
        self.envs = []
        for i, env_fn in enumerate(env_fns):
            try:
                env = env_fn()
                self.envs.append(env)
                print(f"Environment {i}: ✅")
            except Exception as e:
                print(f"Environment {i}: ❌ {e}")
                self.envs.append(None)
        
        # Create process pool
        self.pool = mp.Pool(self.n_workers)
        print(f"Created process pool with {self.n_workers} workers")
    
    def reset(self):
        """Reset all environments"""
        obs_list = []
        for i, env in enumerate(self.envs):
            if env is not None:
                try:
                    obs, _ = env.reset()
                    obs_list.append(obs.astype(np.float32))
                except Exception as e:
                    print(f"Env {i} reset failed: {e}")
                    obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
            else:
                obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
        
        return np.stack(obs_list)
    
    def step(self, actions):
        """Step all environments in parallel"""
        # Prepare arguments for pool
        valid_envs = [(env, action, i) for i, (env, action) in enumerate(zip(self.envs, actions)) if env is not None]
        
        if not valid_envs:
            # All environments failed - return dummy results
            dummy_obs = np.random.rand(self.num_envs, 8, 50, 50).astype(np.float32)
            dummy_rewards = np.full(self.num_envs, -0.1, dtype=np.float32)
            dummy_dones = np.ones(self.num_envs, dtype=bool)
            dummy_infos = [{"burned": 150.0, "new_cells": 0} for _ in range(self.num_envs)]
            return dummy_obs, dummy_rewards, dummy_dones, dummy_dones, dummy_infos
        
        # Execute in parallel
        try:
            results = self.pool.map(step_single_env, valid_envs, chunksize=1)
        except Exception as e:
            print(f"Pool execution failed: {e}")
            # Return dummy results
            dummy_obs = np.random.rand(self.num_envs, 8, 50, 50).astype(np.float32)
            dummy_rewards = np.full(self.num_envs, -0.1, dtype=np.float32)
            dummy_dones = np.ones(self.num_envs, dtype=bool)
            dummy_infos = [{"burned": 150.0, "new_cells": 0} for _ in range(self.num_envs)]
            return dummy_obs, dummy_rewards, dummy_dones, dummy_dones, dummy_infos
        
        # Process results
        obs_list = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        dones = [True] * self.num_envs
        infos = [{"burned": 150.0, "new_cells": 0}] * self.num_envs
        
        for env_id, result, error in results:
            if error:
                print(f"Env {env_id} error: {error}")
            
            obs, reward, done, truncated, info = result
            obs_list[env_id] = obs.astype(np.float32)
            rewards[env_id] = float(reward)
            dones[env_id] = bool(done or truncated)
            infos[env_id] = info or {"burned": 150.0, "new_cells": 0}
        
        # Fill missing results
        for i in range(self.num_envs):
            if obs_list[i] is None:
                obs_list[i] = np.random.rand(8, 50, 50).astype(np.float32)
        
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(dones, dtype=bool),  # truncated same as done
            infos
        )
    
    def close(self):
        """Clean shutdown"""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
        
        for env in self.envs:
            if env is not None:
                try:
                    if hasattr(env, 'close'):
                        env.close()
                except:
                    pass

class SimpleReplayBuffer:
    """Simple replay buffer with proper data types."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        # Ensure all data types are correct
        obs = obs.astype(np.float32)
        action = action.astype(np.float32)
        reward = float(reward)
        next_obs = next_obs.astype(np.float32)
        done = bool(done)
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def make_simple_env(raster, budget, kstep, sims, seed):
    """Create simple environment"""
    def thunk():
        random.seed(seed)
        np.random.seed(seed)
        
        env = FuelBreakEnv(
            raster,
            break_budget=budget,
            break_step=kstep,
            num_simulations=1,  # Keep simple
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
                action = np.zeros(HxW, dtype=np.float32)
                indices = np.random.choice(HxW, size=min(k, HxW), replace=False)
                action[indices] = 1.0
            else:
                q_i = q_vals[i].cpu().numpy()
                top_indices = np.argsort(q_i)[-k:]
                action = np.zeros(HxW, dtype=np.float32)
                action[top_indices] = 1.0
            
            actions.append(action)
    
    return np.array(actions, dtype=np.float32)

def compute_q_loss(model, target_model, batch, gamma, device="cpu"):
    """Compute Q-learning loss with proper data types."""
    obs_batch = torch.stack([torch.from_numpy(t.obs) for t in batch]).to(device)
    action_batch = torch.stack([torch.from_numpy(t.action) for t in batch]).to(device)
    reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
    next_obs_batch = torch.stack([torch.from_numpy(t.next_obs) for t in batch]).to(device)
    done_batch = torch.tensor([t.done for t in batch], dtype=torch.bool).to(device)
    
    current_q = model(obs_batch)
    current_q_action = (current_q * action_batch).sum(dim=1)
    
    with torch.no_grad():
        next_q = target_model(next_obs_batch)
        next_q_max = next_q.max(dim=1)[0]
        target_q = reward_batch + gamma * next_q_max * (~done_batch).float()
    
    loss = nn.MSELoss()(current_q_action, target_q)
    return loss

def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")
    
    # Hyperparameters
    EPISODES = 1000
    STEPS_PER_EP = 2
    BUFFER_CAP = 50_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 3e-4
    START_EPS = 1.0
    END_EPS = 0.01
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 1000
    SAVE_EVERY = 25
    
    # Environment parameters
    N_ENVS = 32  # Reasonable number for multiprocessing
    BUDGET = 200
    K_STEPS = 10
    SIMS = 1
    
    print(f"Training with {N_ENVS} environments...")
    
    # Load rasters
    raster_manager = RasterManager("cropped_raster", 500)
    try:
        raster_manager.load_all_rasters()
        print(f"Loaded {len(raster_manager.all_rasters)} rasters")
    except Exception as e:
        print(f"Error loading rasters: {e}")
        return
    
    # Create environments
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
    env_fns = [
        make_simple_env(raster, BUDGET, K_STEPS, SIMS, seed=i) 
        for i, raster in enumerate(selected_rasters)
    ]
    
    vec_env = SimpleParallelEnv(env_fns, n_workers=min(N_ENVS, 8))
    obs = vec_env.reset()
    _, C, H, W = obs.shape
    
    # Initialize model
    model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
    target_model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = SimpleReplayBuffer(BUFFER_CAP)
    
    # Training metrics
    reward_window = deque(maxlen=100)
    loss_window = deque(maxlen=100)
    burned_area_window = deque(maxlen=100)
    
    global_step = 0
    best_reward = float('-inf')
    
    print("Starting simple parallel training...")
    
    try:
        for episode in range(EPISODES):
            episode_rewards = []
            
            for step in range(STEPS_PER_EP):
                # Choose actions
                eps = START_EPS - (START_EPS - END_EPS) * min(1.0, global_step / EPS_DECAY_STEPS)
                actions = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)
                
                # Environment step
                next_obs, rewards, dones, truncated, infos = vec_env.step(actions)
                
                # Store transitions
                for i in range(N_ENVS):
                    buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                    
                    # Track metrics
                    if infos[i] and "burned" in infos[i]:
                        burned_area_window.append(safe_scalar(infos[i]["burned"]))
                    
                    if dones[i]:
                        episode_rewards.append(rewards[i])
                
                obs = next_obs
                global_step += N_ENVS
                
                # Training step
                if len(buffer) >= BATCH_SIZE:
                    batch = buffer.sample(BATCH_SIZE)
                    loss = compute_q_loss(model, target_model, batch, GAMMA, DEVICE)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    loss_window.append(loss.item())
                    
                    # Update target network
                    if global_step % TARGET_SYNC_EVERY == 0:
                        target_model.load_state_dict(model.state_dict())
            
            # Episode statistics
            if episode_rewards:
                reward_window.extend(episode_rewards)
            
            mean_reward = np.mean(reward_window) if reward_window else 0.0
            mean_loss = np.mean(loss_window) if loss_window else 0.0
            mean_burned = np.mean(burned_area_window) if burned_area_window else 0.0
            
            print(f"[Episode {episode}] Reward: {mean_reward:.3f} Loss: {mean_loss:.3f} "
                  f"Burned: {mean_burned:.1f} Eps: {eps:.3f}")
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(model.state_dict(), "best_simple_parallel_model.pth")
            
            # Periodic save
            if episode % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f"simple_parallel_model_ep{episode}.pth")
    
    finally:
        vec_env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    main()