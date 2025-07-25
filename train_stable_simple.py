#!/usr/bin/env python3
"""
Simplified Stable Training Script
Removes complex error handling and reward structure for stable learning.
Uses original simple reward calculation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

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

class SimpleAutoResetWrapper:
    """Simple wrapper without complex error handling."""
    
    def __init__(self, env):
        self.env = env
        self._ret = 0
        self._len = 0
        
    def reset(self, **kw):
        self._ret = 0
        self._len = 0
        return self.env.reset(**kw)
    
    def step(self, a):
        obs, r, d, tr, info = self.env.step(a)
        self._ret += safe_scalar(r)
        self._len += 1
        
        if d or tr:
            info = info or {}
            info["episode_return"] = self._ret
            info["episode_length"] = self._len
            
        return obs, r, d, tr, info

class SimpleFuelBreakEnv(FuelBreakEnv):
    """Simplified FuelBreakEnv with original reward structure."""
    
    def step(self, action):
        """Step with simple, stable reward calculation."""
        try:
            # Original environment step logic
            new_cells = self._get_cells_from_action(action)
            
            if new_cells.size == 0:
                obs = self._make_obs()
                return obs, 0.0, False, False, {"burned": 0.0, "new_cells": 0}
            
            # Update fuel break mask
            for y, x in new_cells:
                self._break_mask[y, x] = True
            
            self._used += new_cells.shape[0]
            
            # Run fire simulation
            try:
                burned = self.sim.run_many_simulations(
                    self.raster, 
                    fireline_mask=self._break_mask,
                    num_simulations=self.num_simulations,
                    seed=self.seed + self._used
                )
                burned = float(burned)
            except Exception:
                # Simple fallback - don't overthink it
                burned = 150.0  # Reasonable middle value
            
            # Simple reward calculation (original style)
            if self._last_burned is None:
                incremental = burned
            else:
                incremental = burned - self._last_burned
            
            self._last_burned = burned
            reward = -incremental / float(self.H * self.W)  # Simple incremental reward
            
            done = self._used >= self.break_budget
            obs = self._make_obs()
            
            return obs, reward, done, False, {
                "burned": burned, 
                "new_cells": new_cells.size
            }
            
        except Exception as e:
            # Simple fallback
            obs = self._make_obs()
            return obs, -0.1, True, False, {"burned": 150.0, "new_cells": 0}

class SimpleReplayBuffer:
    """Simple replay buffer."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def make_simple_env(raster, budget, kstep, sims, seed):
    """Create simple environment."""
    def thunk():
        random.seed(seed)
        np.random.seed(seed)
        
        env = SimpleFuelBreakEnv(
            raster,
            break_budget=budget,
            break_step=kstep,
            num_simulations=1,  # Keep it simple
            seed=seed,
        )
        return SimpleAutoResetWrapper(env)
    
    return thunk

def choose_actions_batch(model, obs_np, k, eps, device="cpu"):
    """Choose actions for batch of observations."""
    N, C, H, W = obs_np.shape
    HxW = H * W
    
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_np).to(device)
        q_vals = model(obs_t)
        
        actions = []
        for i in range(N):
            if np.random.rand() < eps:
                action = np.zeros(HxW)
                indices = np.random.choice(HxW, size=min(k, HxW), replace=False)
                action[indices] = 1
            else:
                q_i = q_vals[i].cpu().numpy()
                top_indices = np.argsort(q_i)[-k:]
                action = np.zeros(HxW)
                action[top_indices] = 1
            
            actions.append(action)
    
    return np.array(actions)

def compute_q_loss(model, target_model, batch, gamma, device="cpu"):
    """Compute Q-learning loss."""
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
        target_q = reward_batch + gamma * next_q_max * (~done_batch)
    
    loss = nn.MSELoss()(current_q_action, target_q)
    return loss

def main():
    # Simple, stable hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")
    
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
    
    # Simple environment setup
    N_ENVS = 4  # Keep it small and simple
    BUDGET = 200
    K_STEPS = 10
    SIMS = 1
    
    # Load rasters
    print("Loading rasters...")
    raster_manager = RasterManager("cropped_raster", 500)
    try:
        raster_manager.load_all_rasters()
        print(f"Loaded {len(raster_manager.all_rasters)} rasters")
    except Exception as e:
        print(f"Error loading rasters: {e}")
        return
    
    # Create simple synchronous environments
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
    envs = [
        make_simple_env(raster, BUDGET, K_STEPS, SIMS, seed=i)() 
        for i, raster in enumerate(selected_rasters)
    ]
    
    # Initialize first observations
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    obs = np.stack(obs_list)
    
    _, C, H, W = obs.shape
    
    # Initialize model (simple version)
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
    
    print(f"Starting simple stable training with {N_ENVS} environments...")
    
    for episode in range(EPISODES):
        episode_rewards = []
        
        for step in range(STEPS_PER_EP):
            # Choose actions
            eps = START_EPS - (START_EPS - END_EPS) * min(1.0, global_step / EPS_DECAY_STEPS)
            actions = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)
            
            # Step all environments synchronously
            next_obs_list = []
            rewards = []
            dones = []
            infos = []
            
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_obs, reward, done, truncated, info = env.step(action)
                next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(done or truncated)
                infos.append(info)
                
                # Reset if done
                if done or truncated:
                    next_obs, _ = env.reset()
                    next_obs_list[-1] = next_obs
            
            next_obs = np.stack(next_obs_list)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)
            
            # Store transitions
            for i in range(N_ENVS):
                buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                
                # Track metrics
                if infos[i] and "burned" in infos[i]:
                    burned_area_window.append(safe_scalar(infos[i]["burned"]))
                
                if dones[i] and "episode_return" in infos[i]:
                    episode_rewards.append(safe_scalar(infos[i]["episode_return"]))
            
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
            torch.save(model.state_dict(), "best_simple_model.pth")
        
        # Periodic save
        if episode % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"simple_model_ep{episode}.pth")
    
    print("Training completed!")

if __name__ == "__main__":
    main()