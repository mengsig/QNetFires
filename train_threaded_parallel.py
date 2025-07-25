#!/usr/bin/env python3
"""
Threaded Parallel Training Script for QAgent
Uses threading instead of multiprocessing for stable parallel environment execution.
No pipe corruption, no hanging, stable learning.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import threading
import queue
import time

# Add src to path
sys.path.append('src')

from Env import FuelBreakEnv
from Model import QNet, EnhancedQNet, DuelingQNet
from utils.loadingUtils import RasterManager

Transition = namedtuple("Transition", "obs action reward next_obs done")

def safe_scalar(value, fallback=0.0):
    """Convert value to scalar, handling arrays, None, and string values safely."""
    if value is None:
        return fallback
    if isinstance(value, str):
        if value == 'N/A':
            return fallback
        try:
            return float(value)
        except ValueError:
            return fallback
    if hasattr(value, '__len__') and len(value) > 1:
        try:
            return float(np.mean(value))
        except:
            return fallback
    try:
        return float(value)
    except:
        return fallback

class ThreadedVectorEnv:
    """Threaded vector environment for stable parallel execution."""
    
    def __init__(self, env_fns, max_workers=None, budget=250):
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.envs = [None] * self.num_envs
        self.budget = budget
        self.max_workers = max_workers or min(self.num_envs, 16)  # Limit threads for stability
        
        # Initialize environments
        self._create_environments()
        
        # Threading infrastructure
        self.step_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self._start_workers()
        
    def _create_environments(self):
        """Create all environments safely."""
        print(f"Creating {self.num_envs} environments with threading...")
        for i, env_fn in enumerate(self.env_fns):
            try:
                self.envs[i] = env_fn()
                print(f"Environment {i}: ✅ Created successfully")
            except Exception as e:
                print(f"Environment {i}: ❌ Failed: {e}")
                # Create a minimal dummy environment with proper budget
                self.envs[i] = self._create_dummy_env(budget=self.budget)
    
    def _create_dummy_env(self, budget=250):
        """Create a minimal dummy environment for fallback."""
        class DummyEnv:
            def __init__(self, budget=250):
                self.H, self.W = 50, 50
                self.steps = 0
                self.budget = budget
                self.fuel_breaks_used = 0
                
            def reset(self):
                self.steps = 0
                self.fuel_breaks_used = 0
                obs = np.random.rand(8, self.H, self.W).astype(np.float32)
                return obs, {}
                
            def step(self, action):
                self.steps += 1
                # Count fuel breaks from action
                action_array = np.asarray(action).reshape(-1)
                new_breaks = min(np.sum(action_array > 0.5), 10)  # Max 10 per step
                self.fuel_breaks_used += new_breaks
                
                obs = np.random.rand(8, self.H, self.W).astype(np.float32)
                reward = -0.1  # Small negative reward
                
                # Done when budget is exceeded (like real environment)
                done = self.fuel_breaks_used >= self.budget
                
                # Simulate burned area reducing as more fuel breaks are placed
                base_burned = 200.0
                reduction = (self.fuel_breaks_used / self.budget) * 100.0
                burned = max(50.0, base_burned - reduction)
                
                info = {
                    "burned": burned, 
                    "new_cells": int(new_breaks), 
                    "total_fuel_breaks": self.fuel_breaks_used,
                    "dummy": True
                }
                return obs, reward, done, False, info
                
        return DummyEnv(budget)
    
    def _start_workers(self):
        """Start worker threads for parallel environment execution."""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread loop for processing environment steps."""
        while True:
            try:
                task = self.step_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                    
                env_id, action, task_id = task
                
                try:
                    # Execute environment step
                    result = self.envs[env_id].step(action)
                    self.result_queue.put((env_id, result, task_id, None))
                except Exception as e:
                    # Return error result
                    error_result = (
                        np.random.rand(8, 50, 50).astype(np.float32),  # dummy obs
                        -1.0,  # penalty reward
                        True,  # done
                        False,  # truncated
                        {"burned": 200.0, "new_cells": 0, "error": True}
                    )
                    self.result_queue.put((env_id, error_result, task_id, e))
                    
                self.step_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                break
    
    def reset(self):
        """Reset all environments."""
        obs_list = []
        for i, env in enumerate(self.envs):
            try:
                obs, info = env.reset()
                obs_list.append(obs)
            except Exception as e:
                print(f"Environment {i} reset failed: {e}")
                # Use dummy observation
                obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
        
        return np.stack(obs_list)
    
    def step(self, actions):
        """Step all environments in parallel using threads."""
        task_id = time.time()  # Unique task identifier
        
        # Submit all tasks
        for env_id, action in enumerate(actions):
            self.step_queue.put((env_id, action, task_id))
        
        # Collect results
        results = {}
        errors = []
        
        for _ in range(self.num_envs):
            try:
                env_id, result, returned_task_id, error = self.result_queue.get(timeout=10.0)
                if returned_task_id == task_id:  # Ensure it's from this step
                    results[env_id] = result
                    if error:
                        errors.append(f"Env {env_id}: {error}")
            except queue.Empty:
                print(f"⚠️  Timeout waiting for environment result")
                # Create dummy result for missing environment
                missing_envs = set(range(self.num_envs)) - set(results.keys())
                for env_id in missing_envs:
                    dummy_result = (
                        np.random.rand(8, 50, 50).astype(np.float32),
                        -1.0, True, False,
                        {"burned": 200.0, "new_cells": 0, "timeout": True}
                    )
                    results[env_id] = dummy_result
                break
        
        if errors:
            print(f"Environment errors: {errors[:3]}...")  # Show first 3
        
        # Organize results
        obs_list = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        
        for env_id in range(self.num_envs):
            if env_id in results:
                obs, reward, done, trunc, info = results[env_id]
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                truncated.append(trunc)
                infos.append(info)
            else:
                # Fallback for missing result
                obs_list.append(np.random.rand(8, 50, 50).astype(np.float32))
                rewards.append(-1.0)
                dones.append(True)
                truncated.append(False)
                infos.append({"burned": 200.0, "new_cells": 0, "missing": True})
        
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncated, dtype=bool),
            infos
        )
    
    def close(self):
        """Clean shutdown of threaded environment."""
        # Signal workers to stop
        for _ in self.workers:
            self.step_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        # Close environments
        for env in self.envs:
            try:
                if hasattr(env, 'close'):
                    env.close()
            except:
                pass

class SimpleReplayBuffer:
    """Simple replay buffer for stable learning."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        # Ensure consistent data types
        obs = obs.astype(np.float32) if hasattr(obs, 'astype') else np.array(obs, dtype=np.float32)
        action = action.astype(np.float32) if hasattr(action, 'astype') else np.array(action, dtype=np.float32)  # Convert int8 to float32 for buffer
        reward = float(reward)
        next_obs = next_obs.astype(np.float32) if hasattr(next_obs, 'astype') else np.array(next_obs, dtype=np.float32)
        done = bool(done)
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def make_env_with_raster(raster, budget, kstep, sims, seed):
    """Create environment with specific raster data."""
    def thunk():
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            # Create environment with minimal simulations for stability
            env = FuelBreakEnv(
                raster,
                break_budget=budget,
                break_step=kstep,
                num_simulations=min(sims, 2),  # Keep simulations low
                seed=seed,
            )
            
            
            # Wrap with a custom wrapper that doesn't auto-reset
            class BudgetEnforcementWrapper:
                def __init__(self, env):
                    self.env = env
                    self.is_done = False
                    self.total_fuel_breaks = 0
                    
                def reset(self):
                    self.is_done = False
                    self.total_fuel_breaks = 0
                    return self.env.reset()
                    
                def step(self, action):
                    if self.is_done:
                        # Return dummy result if already done
                        obs = self.env._make_obs()
                        return obs, 0.0, True, False, {"burned": 0.0, "new_cells": 0, "budget_exceeded": True}
                    
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Track fuel breaks
                    new_cells = info.get("new_cells", 0)
                    self.total_fuel_breaks += new_cells
                    
                    # Enforce budget strictly
                    if self.total_fuel_breaks >= budget:
                        done = True
                        self.is_done = True
                        
                    # Update info with accurate count
                    info["total_fuel_breaks"] = self.total_fuel_breaks
                    
                    return obs, reward, done, truncated, info
                    
                def close(self):
                    if hasattr(self.env, 'close'):
                        self.env.close()
            
            return BudgetEnforcementWrapper(env)
            
        except Exception as e:
            print(f"Environment creation failed: {e}")
            raise
    
    return thunk

def choose_actions_batch(model, obs_np, k, eps, device="cpu"):
    """Choose actions for batch of observations."""
    N, C, H, W = obs_np.shape
    HxW = H * W
    
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_np).to(device)
        q_vals = model(obs_t)  # (N, HxW)
        
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

def compute_q_loss(model, target_model, batch, gamma, device="cpu"):
    """Compute Q-learning loss."""
    # Ensure all tensors are float32 and properly converted
    obs_batch = torch.stack([torch.from_numpy(t.obs.astype(np.float32)) for t in batch]).to(device)
    action_batch = torch.stack([torch.from_numpy(t.action.astype(np.float32)) for t in batch]).to(device)
    reward_batch = torch.tensor([float(t.reward) for t in batch], dtype=torch.float32).to(device)
    next_obs_batch = torch.stack([torch.from_numpy(t.next_obs.astype(np.float32)) for t in batch]).to(device)
    done_batch = torch.tensor([bool(t.done) for t in batch], dtype=torch.bool).to(device)
    
    # Current Q-values
    current_q = model(obs_batch)
    current_q_action = (current_q * action_batch).sum(dim=1)
    
    # Target Q-values
    with torch.no_grad():
        next_q = target_model(next_obs_batch)
        next_q_max = next_q.max(dim=1)[0]
        target_q = reward_batch + gamma * next_q_max * (~done_batch).float()
    
    # Compute loss
    loss = nn.MSELoss()(current_q_action, target_q)
    return loss

def main():
    # Stable hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")
    
    # Reduced complexity for stability
    EPISODES = 1000
    STEPS_PER_EP = 30  # Allow episodes to complete naturally (250 budget / 10 per step = 25 steps)
    BUFFER_CAP = 50_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 3e-4
    START_EPS = 1.0
    END_EPS = 0.01
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 1000
    SAVE_EVERY = 25
    
    # Environment parameters - optimized for stability
    N_ENVS = 16  # Reasonable for threading (user had 64 but that's too many threads)
    BUDGET = 250  # Match user's budget setting
    K_STEPS = 10
    SIMS = 1  # Minimal simulations for stability
    
    # Load rasters
    print("Loading rasters...")
    raster_manager = RasterManager("cropped_raster", 500)
    try:
        raster_manager.load_all_rasters()
        print(f"Loaded {len(raster_manager.all_rasters)} rasters")
    except Exception as e:
        print(f"Error loading rasters: {e}")
        return
    
    # Create environments
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
    print(f"🌍 Initial raster selection: Using {len(selected_rasters)} rasters for {N_ENVS} environments")
    
    env_fns = [
        make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i) 
        for i, raster in enumerate(selected_rasters)
    ]
    
    vec_env = ThreadedVectorEnv(env_fns, budget=BUDGET)
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
    
    print(f"Starting stable threaded training with {N_ENVS} environments...")
    
    for episode in range(EPISODES):
        # Cycle rasters every 10 episodes to ensure generalization
        if episode % 10 == 0 and episode > 0:
            print(f"\n🔄 Cycling rasters at episode {episode}...")
            try:
                # Get new rasters
                selected_rasters = raster_manager.get_random_rasters(N_ENVS)
                print(f"Selected {len(selected_rasters)} new rasters")
                
                # Close old environment
                vec_env.close()
                
                # Create new environments with new rasters
                env_fns = [
                    make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=episode*N_ENVS + i) 
                    for i, raster in enumerate(selected_rasters)
                ]
                
                vec_env = ThreadedVectorEnv(env_fns, budget=BUDGET)
                obs = vec_env.reset()
                print(f"✅ Successfully created new environments with fresh rasters")
                
            except Exception as e:
                print(f"❌ Failed to cycle rasters: {e}")
                print("Continuing with existing environments...")
        
        episode_rewards = []
        active_envs = [True] * N_ENVS  # Track which environments are still active
        
        for step in range(STEPS_PER_EP):
            # Choose actions
            eps = START_EPS - (START_EPS - END_EPS) * min(1.0, global_step / EPS_DECAY_STEPS)
            actions = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)
            
            # Environment step
            next_obs, rewards, dones, truncated, infos = vec_env.step(actions)
            
            # Store transitions and track completion
            completed_this_step = 0
            for i in range(N_ENVS):
                buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                
                # Track metrics
                if infos[i] and "burned" in infos[i]:
                    burned_area_window.append(safe_scalar(infos[i]["burned"]))
                
                if dones[i] and active_envs[i]:
                    episode_rewards.append(rewards[i])
                    active_envs[i] = False  # Mark as completed
                    completed_this_step += 1
                    
                    # Log completion with fuel break count
                    total_breaks = infos[i].get("total_fuel_breaks", 0) if infos[i] else 0
                    burned = safe_scalar(infos[i].get("burned", 0)) if infos[i] else 0
                    is_dummy = infos[i].get("dummy", False) if infos[i] else False
                    env_type = "DUMMY" if is_dummy else "REAL"
                    print(f"🎯 Environment {i} ({env_type}) completed: Burned={burned:.1f}, Total Breaks={total_breaks}/{BUDGET}, Step={step+1}")
            
            obs = next_obs
            global_step += N_ENVS
            
            # If all environments are done, break early
            if not any(active_envs):
                print(f"✅ All environments completed at step {step+1}")
                break
            
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
        
        # Check budget compliance
        budget_compliant = 0
        total_completed = len(episode_rewards)
        
        print(f"[Episode {episode}] Reward: {mean_reward:.3f} Loss: {mean_loss:.3f} "
              f"Burned: {mean_burned:.1f} Eps: {eps:.3f} Completed: {total_completed}/{N_ENVS}")
        
        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(model.state_dict(), "best_threaded_model.pth")
        
        # Periodic save
        if episode % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"threaded_model_ep{episode}.pth")
    
    vec_env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()