#!/usr/bin/env python3
"""
Synchronous training script for QAgent that avoids AsyncVectorEnv hanging issues.
This version processes environments sequentially to prevent multiprocessing deadlocks.
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

class SyncVectorEnv:
    """Synchronous vector environment that processes environments sequentially."""
    
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
    def reset(self):
        obs_list = []
        for env in self.envs:
            try:
                obs, info = env.reset()
                obs_list.append(obs)
            except Exception as e:
                print(f"Environment reset failed: {e}")
                # Use dummy observation
                obs_list.append(np.zeros((8, 50, 50), dtype=np.float32))
        return np.stack(obs_list)
    
    def step(self, actions):
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            try:
                obs, reward, done, truncated, info = env.step(action)
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done or truncated)
                infos.append(info)
            except Exception as e:
                print(f"Environment {i} step failed: {e}")
                # Use dummy values
                obs_list.append(np.zeros((8, 50, 50), dtype=np.float32))
                rewards.append(-1.0)
                dones.append(True)
                infos.append({"error": True})
        
        return (np.stack(obs_list), 
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool),
                np.array([False] * self.num_envs, dtype=bool),  # truncated
                infos)
    
    def close(self):
        for env in self.envs:
            try:
                env.close()
            except:
                pass

class SimpleReplayBuffer:
    """Simple replay buffer for synchronous training."""
    
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)

def compute_q_loss_simple(model, target_model, batch, gamma, k, device="cpu"):
    """Simple Q-learning loss computation."""
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(device)
    next_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(device)
    a_t = torch.from_numpy(np.stack(batch.action)).long().to(device)
    r_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(device)
    d_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(device)

    q_all = model(obs_t)
    mask = a_t.bool()
    q_sel = torch.sum(q_all * mask.float(), dim=1) / k

    with torch.no_grad():
        next_online = model(next_t)
        best_idx = torch.argmax(next_online, dim=1)
        next_target = target_model(next_t)
        q_next = next_target.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        target = r_t + gamma * (1 - d_t.float()) * q_next

    return nn.MSELoss()(q_sel, target)

def choose_actions_simple(model, obs_np, k, eps, device="cpu"):
    """Simple action selection."""
    N, C, H, W = obs_np.shape
    HxW = H * W
    acts = np.zeros((N, HxW), dtype=np.int8)
    
    with torch.no_grad():
        qs = model(torch.from_numpy(obs_np).to(device)).cpu().numpy()
    
    for i in range(N):
        if random.random() < eps:
            idx = np.random.choice(HxW, k, replace=False)
        else:
            qi = qs[i]
            idx = np.argpartition(qi, -k)[-k:]
        acts[i, idx] = 1
    
    return acts

def make_simple_env(raster, budget, kstep, sims, seed):
    """Create a simple environment without multiprocessing."""
    def thunk():
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            from Train import RobustAutoResetWrapper
            env = FuelBreakEnv(raster, break_budget=budget, break_step=kstep, 
                             num_simulations=sims, seed=seed)
            return RobustAutoResetWrapper(env)
        except Exception as e:
            print(f"Environment creation failed: {e}, using dummy environment")
            from Train import DummyEnv
            return DummyEnv(budget, kstep, raster)
    
    return thunk

def main():
    """Run synchronous training."""
    print("QAgent Synchronous Training (No Hanging Issues)")
    print("=" * 50)
    
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")
    
    # Hyperparams - Conservative settings for stability
    EPISODES = 500
    STEPS_PER_EP = 2
    BATCH_SIZE = 16  # Smaller for stability
    GAMMA = 0.99
    LR = 3e-4
    START_EPS = 1.0
    END_EPS = 0.01
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 500
    SAVE_EVERY = 25
    
    # Environment parameters - Conservative for stability
    N_ENVS = 8  # Fewer environments to reduce complexity
    BUDGET = 200
    K_STEPS = 10
    SIMS = 5  # Fewer simulations for stability
    
    # Model settings
    USE_ENHANCED_MODEL = False  # Use basic model for stability
    
    # Raster management
    MAX_RASTERS = 500
    RASTER_ROOT = "cropped_raster"
    
    print("Initializing Raster Manager...")
    raster_manager = RasterManager(RASTER_ROOT, MAX_RASTERS)
    
    # Load rasters
    try:
        all_rasters = raster_manager.load_all_rasters()
        print(f"Successfully loaded {len(all_rasters)} rasters")
        if len(all_rasters) == 0:
            print("No rasters found! Creating dummy rasters for testing...")
            dummy_raster = {
                'slp': np.random.rand(50, 50).astype(np.float32),
                'asp': np.random.rand(50, 50).astype(np.float32),
                'dem': np.random.rand(50, 50).astype(np.float32),
                'cc': np.random.rand(50, 50).astype(np.float32),
                'cbd': np.random.rand(50, 50).astype(np.float32),
                'cbh': np.random.rand(50, 50).astype(np.float32),
                'ch': np.random.rand(50, 50).astype(np.float32),
                'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
                'fireline_north': np.zeros((50, 50), dtype=np.float32),
                'fireline_east': np.zeros((50, 50), dtype=np.float32),
                'fireline_south': np.zeros((50, 50), dtype=np.float32),
                'fireline_west': np.zeros((50, 50), dtype=np.float32),
            }
            raster_manager.all_rasters = [dummy_raster] * N_ENVS
    except Exception as e:
        print(f"Error loading rasters: {e}")
        # Create dummy rasters
        dummy_raster = {
            'slp': np.random.rand(50, 50).astype(np.float32),
            'asp': np.random.rand(50, 50).astype(np.float32),
            'dem': np.random.rand(50, 50).astype(np.float32),
            'cc': np.random.rand(50, 50).astype(np.float32),
            'cbd': np.random.rand(50, 50).astype(np.float32),
            'cbh': np.random.rand(50, 50).astype(np.float32),
            'ch': np.random.rand(50, 50).astype(np.float32),
            'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
            'fireline_north': np.zeros((50, 50), dtype=np.float32),
            'fireline_east': np.zeros((50, 50), dtype=np.float32),
            'fireline_south': np.zeros((50, 50), dtype=np.float32),
            'fireline_west': np.zeros((50, 50), dtype=np.float32),
        }
        raster_manager.all_rasters = [dummy_raster] * N_ENVS
    
    # Get initial batch of rasters
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)
    
    # Create synchronous vector environment
    env_fns = [
        make_simple_env(raster, BUDGET, K_STEPS, SIMS, seed=i) 
        for i, raster in enumerate(selected_rasters)
    ]
    vec_env = SyncVectorEnv(env_fns)
    
    obs = vec_env.reset()
    N_ENVS = obs.shape[0]
    _, C, H, W = obs.shape
    
    # Initialize model
    if USE_ENHANCED_MODEL:
        model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
        tgt = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
        print("Using Memory-Efficient Enhanced DQN")
    else:
        model = QNet(H, W).to(DEVICE)
        tgt = QNet(H, W).to(DEVICE)
        print("Using Basic DQN")
    
    tgt.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    buf = SimpleReplayBuffer(50_000)
    
    global_step = 0
    eps = START_EPS
    loss_win = deque(maxlen=1000)
    reward_win = deque(maxlen=100)  # Episode returns
    step_reward_win = deque(maxlen=1000)  # Step-by-step rewards
    burned_area_win = deque(maxlen=1000)  # Burned areas
    best_avg_reward = float('-inf')
    
    print("Starting synchronous training...")
    
    for ep in range(1, EPISODES + 1):
        print(f"META-EP: {ep}/{EPISODES}")
        
        # Periodically refresh environments
        if ep % 20 == 0 and raster_manager.get_num_loaded_rasters() > N_ENVS:
            print("Refreshing environments...")
            vec_env.close()
            selected_rasters = raster_manager.get_random_rasters(N_ENVS)
            env_fns = [
                make_simple_env(raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS) 
                for i, raster in enumerate(selected_rasters)
            ]
            vec_env = SyncVectorEnv(env_fns)
            obs = vec_env.reset()
        
        for step in range(STEPS_PER_EP):
            acts = choose_actions_simple(model, obs, K_STEPS, eps, DEVICE)
            
            # Synchronous step - no hanging issues
            nxt, rews, dones, truncs, infos = vec_env.step(acts)
            dones = np.logical_or(dones, truncs)
            
            # Store transitions
            for i in range(N_ENVS):
                buf.push(obs[i], acts[i], rews[i], nxt[i], dones[i])
                
                # Always track step rewards
                step_reward_win.append(rews[i])
                
                # Track burned area if available
                if infos[i] and "burned" in infos[i]:
                    burned_area_win.append(infos[i]["burned"])
                
                # Track episode completion
                if infos[i] and "episode_return" in infos[i]:
                    episode_reward = infos[i]['episode_return']
                    reward_win.append(episode_reward)
                    print(f"[env {i}] Episode completed: R={episode_reward:.3f} L={infos[i].get('episode_length', 0)} "
                          f"Burned={infos[i].get('burned', 'N/A'):.1f}")
                elif dones[i]:
                    print(f"[env {i}] Episode ended: Step_reward={rews[i]:.3f} "
                          f"Burned={infos[i].get('burned', 'N/A') if infos[i] else 'N/A':.1f}")
            
            obs = nxt
            global_step += N_ENVS
            
            # Update epsilon
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac
            
            # Training step
            if len(buf) >= BATCH_SIZE:
                batch = buf.sample(BATCH_SIZE)
                loss = compute_q_loss_simple(model, tgt, batch, GAMMA, K_STEPS, DEVICE)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()
                
                loss_win.append(loss.item())
                
                # Update target network
                if global_step % TARGET_SYNC_EVERY == 0:
                    tgt.load_state_dict(model.state_dict())
        
        # Episode statistics with multiple metrics
        mean_loss = float(np.mean(loss_win)) if loss_win else float("nan")
        
        # Episode returns (if episodes complete)
        mean_episode_reward = float(np.mean(reward_win)) if reward_win else float("nan")
        
        # Step rewards (always available)
        mean_step_reward = float(np.mean(step_reward_win)) if step_reward_win else float("nan")
        
        # Burned area (fire spread metric)
        mean_burned_area = float(np.mean(burned_area_win)) if burned_area_win else float("nan")
        
        # Use step reward as primary metric if episode rewards not available
        primary_reward = mean_episode_reward if not np.isnan(mean_episode_reward) else mean_step_reward
        
        print(f"[MetaEp {ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} "
              f"loss={mean_loss:.4f} ep_reward={mean_episode_reward:.3f} "
              f"step_reward={mean_step_reward:.4f} burned_area={mean_burned_area:.1f}")
        
        # Track best performance
        if not np.isnan(primary_reward) and primary_reward > best_avg_reward:
            best_avg_reward = primary_reward
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/qnet_sync_best.pt")
            print(f"🎉 New best model saved! Reward: {primary_reward:.4f}")
        
        # Regular checkpointing
        if ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_reward': best_avg_reward,
                'loss': mean_loss,
            }, f"checkpoints/qnet_sync_ep{ep}.pt")
    
    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_reward': best_avg_reward,
    }, "checkpoints/qnet_sync_final.pt")
    
    vec_env.close()
    print(f"Synchronous training completed! Best average reward: {best_avg_reward:.3f}")

if __name__ == "__main__":
    main()