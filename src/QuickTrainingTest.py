#!/usr/bin/env python3
"""
Quick training test to verify the entire DQN pipeline works correctly.
Runs a few episodes with mock data to test all components together.
"""

import os
import sys
import torch
import numpy as np
import multiprocessing as mp
from collections import deque, namedtuple

# Add project path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from src.Train import (
    ReplayBuffer, compute_q_loss, choose_actions_batch, 
    create_safe_async_env, make_env, safe_step, AutoResetWrapper
)
from src.Model import QNet, DuelingQNet
from src.utils.loadingUtils import get_available_raster_indices
from src.Env import FuelBreakEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Transition = namedtuple("Transition", "obs action reward next_obs done")

import gym
from gym import spaces

class MockFuelBreakEnv(gym.Env):
    """Mock environment for testing when real raster data is not available"""
    
    def __init__(self, H=50, W=50, break_budget=100, break_step=10, num_simulations=5, seed=None):
        super().__init__()
        self.H, self.W = H, W
        self.break_budget = break_budget
        self.break_step = break_step
        self.num_simulations = num_simulations
        self._steps_used = 0
        self._max_steps = break_budget // break_step
        self._current_step = 0
        
        # Define gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8, H, W), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(H * W)
        
        # Add required gym attributes
        self.metadata = {"render.modes": []}
        self.spec = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def reset(self, **kwargs):
        self._steps_used = 0
        self._current_step = 0
        # Return mock observation (8 channels, H, W)
        obs = np.random.rand(8, self.H, self.W).astype(np.float32)
        return obs, {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.int8).reshape(-1)
        
        # Count new placements
        new_placements = np.sum(action)
        new_placements = min(new_placements, self.break_step)
        self._steps_used += new_placements
        self._current_step += 1
        
        # Mock reward (negative to simulate burned area reduction)
        reward = -np.random.exponential(0.1) + 0.05 * new_placements
        
        # Episode ends when budget is used or max steps reached
        done = self._steps_used >= self.break_budget or self._current_step >= self._max_steps
        
        # Next observation
        next_obs = np.random.rand(8, self.H, self.W).astype(np.float32)
        
        info = {
            "burned": np.random.exponential(100),
            "new_cells": new_placements,
            "steps_used": self._steps_used
        }
        
        # Add episode info when done
        if done:
            info["episode_return"] = np.random.normal(-50, 10)  # Mock episode return
            info["episode_length"] = self._current_step
        
        return next_obs, reward, done, False, info


def make_mock_env(budget, kstep, sims, seed, env_id):
    """Create mock environment function"""
    def _thunk():
        try:
            env = MockFuelBreakEnv(
                break_budget=budget,
                break_step=kstep,
                num_simulations=sims,
                seed=seed + env_id
            )
            # Wrap to auto-reset
            env = AutoResetWrapper(env)
            return env
        except Exception as e:
            print(f"Error creating mock environment {env_id}: {e}")
            raise e
    
    return _thunk


def quick_training_test():
    """Run a quick training test with mock environments"""
    print("Starting Quick Training Test...")
    print("=" * 50)
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Test parameters (smaller for quick test)
    EPISODES = 3
    STEPS_PER_EP = 3
    N_ENVS = 4
    BUDGET = 50
    K_STEPS = 5
    SIMS = 5
    BATCH_SIZE = 16
    BUFFER_CAP = 1000
    
    H, W = 50, 50
    
    print(f"Test Configuration:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Steps per episode: {STEPS_PER_EP}")
    print(f"  Number of environments: {N_ENVS}")
    print(f"  Device: {DEVICE}")
    print()
    
    try:
        # Create mock environments
        env_fns = [
            make_mock_env(BUDGET, K_STEPS, SIMS, seed=i, env_id=i)
            for i in range(N_ENVS)
        ]
        
        # Create vectorized environment
        print("Creating vectorized environment...")
        vec_env, reset_out = create_safe_async_env(env_fns)
        
        # Handle reset output
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
        
        print(f"  Environment created successfully!")
        print(f"  Observation shape: {obs.shape}")
        
        # Initialize models
        print("Initializing models...")
        model = QNet(H, W).to(DEVICE)
        target_model = QNet(H, W).to(DEVICE)
        target_model.load_state_dict(model.state_dict())
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        buffer = ReplayBuffer(BUFFER_CAP)
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        global_step = 0
        eps = 0.8
        loss_window = deque(maxlen=100)
        episode_returns = deque(maxlen=50)
        
        print("\nStarting training loop...")
        
        for meta_ep in range(1, EPISODES + 1):
            print(f"\nMeta-Episode {meta_ep}/{EPISODES}")
            
            for step in range(STEPS_PER_EP):
                print(f"  Step {step + 1}/{STEPS_PER_EP}")
                
                # Choose actions
                actions = choose_actions_batch(model, obs, K_STEPS, eps)
                
                # Environment step
                step_out = safe_step(vec_env, actions)
                
                # Handle step output
                if len(step_out) == 5:
                    next_obs, rewards, dones, truncs, infos = step_out
                    dones = np.logical_or(dones, truncs)
                else:
                    next_obs, rewards, dones, infos = step_out
                
                rewards = np.asarray(rewards, dtype=np.float32)
                if rewards.shape != (N_ENVS,):
                    rewards = rewards.reshape(N_ENVS, -1).sum(axis=1)
                dones = np.asarray(dones, dtype=bool)
                
                # Store transitions
                for i in range(N_ENVS):
                    buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                    
                    # Track episode returns
                    info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                    if info_i and "episode_return" in info_i:
                        episode_returns.append(info_i['episode_return'])
                        print(f"    [env {i}] Episode completed! Return: {info_i['episode_return']:.2f}")
                
                obs = next_obs
                global_step += N_ENVS
                
                # Training step
                if len(buffer) >= BATCH_SIZE:
                    batch = buffer.sample(BATCH_SIZE)
                    loss = compute_q_loss(model, target_model, batch, gamma=0.99, k=K_STEPS)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    
                    loss_window.append(loss.item())
                    
                    print(f"    Loss: {loss.item():.4f}")
            
            # Episode summary
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
            
            print(f"  Episode Summary:")
            print(f"    Buffer size: {len(buffer)}")
            print(f"    Mean loss: {mean_loss:.4f}")
            print(f"    Mean return: {mean_return:.2f}")
            print(f"    Epsilon: {eps:.3f}")
        
        # Clean up
        vec_env.close()
        
        print("\n" + "=" * 50)
        print("✅ QUICK TRAINING TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ Processed {global_step} environment steps")
        print(f"✅ Buffer contains {len(buffer)} transitions")
        print(f"✅ Average loss: {np.mean(loss_window):.4f}")
        print("✅ All components working correctly!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING TEST FAILED: {e}")
        print("❌ Check the error details above")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_training_test()
    
    if success:
        print("\n🎉 The training system is ready for full training!")
        print("You can now run the full training with: python src/Train.py")
    else:
        print("\n⚠️  Please fix the issues above before running full training.")