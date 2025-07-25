#!/usr/bin/env python3
"""
Ultra-Minimal Stable Training - Maximum Stability
- Single environment for debugging
- No complex features
- Pure DQN learning
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Add src to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from src.Env import FuelBreakEnv
from src.Model import QNet
from src.utils.loadingUtils import RasterManager

def safe_scalar(value, fallback=0.0):
    """Convert value to scalar safely."""
    try:
        return float(value)
    except:
        return fallback

# ULTRA-MINIMAL PARAMETERS
EPISODES = 100
STEPS_PER_EP = 25
BATCH_SIZE = 16
GAMMA = 0.99
LR = 1e-4
EPS = 0.1

# SINGLE ENVIRONMENT PARAMETERS
BUDGET = 200
K_STEPS = 8
SIMS = 1

class SimpleBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

def main():
    print("🚀 Ultra-Minimal Training Started")
    
    # Load single raster
    raster_manager = RasterManager("cropped_raster", 500)
    raster_manager.load_all_rasters()
    raster = raster_manager.all_rasters[0]
    
    # Create single environment
    env = FuelBreakEnv(raster, break_budget=BUDGET, break_step=K_STEPS, num_simulations=SIMS, seed=42)
    
    # Initialize model
    obs, _ = env.reset()
    _, H, W = obs.shape
    model = QNet(H, W)
    target_model = QNet(H, W)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = SimpleBuffer(10000)
    
    print(f"✅ Single environment created: {obs.shape}")
    
    for episode in range(EPISODES):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(STEPS_PER_EP):
            # Choose action
            if random.random() < EPS:
                action = np.zeros(H * W, dtype=np.int8)
                indices = np.random.choice(H * W, K_STEPS, replace=False)
                action[indices] = 1
            else:
                with torch.no_grad():
                    q_vals = model(torch.from_numpy(obs).unsqueeze(0)).squeeze().numpy()
                    indices = np.argsort(q_vals)[-K_STEPS:]
                    action = np.zeros(H * W, dtype=np.int8)
                    action[indices] = 1
            
            # Environment step
            next_obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            buffer.push(obs, action.astype(np.float32), reward, next_obs, done)
            
            obs = next_obs
            
            # Training
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                
                obs_batch = torch.stack([torch.from_numpy(t[0]) for t in batch])
                action_batch = torch.stack([torch.from_numpy(t[1]) for t in batch])
                reward_batch = torch.tensor([t[2] for t in batch], dtype=torch.float32)
                next_obs_batch = torch.stack([torch.from_numpy(t[3]) for t in batch])
                done_batch = torch.tensor([t[4] for t in batch], dtype=torch.bool)
                
                current_q = (model(obs_batch) * action_batch).sum(dim=1)
                
                with torch.no_grad():
                    next_q = target_model(next_obs_batch).max(dim=1)[0]
                    target_q = reward_batch + GAMMA * next_q * (~done_batch).float()
                
                loss = nn.MSELoss()(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        
        burned = safe_scalar(info.get("burned", 0)) if info else 0
        print(f"Episode {episode}: Reward={episode_reward:.3f} Burned={burned:.1f}")
    
    print("✅ Ultra-minimal training completed!")

if __name__ == "__main__":
    main()
