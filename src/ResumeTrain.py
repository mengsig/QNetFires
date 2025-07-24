#!/usr/bin/env python3
"""
Resume training script for the Deep Q-Network implementation.
This script allows resuming training from a saved checkpoint.
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from collections import deque
import numpy as np

from Train import main as train_main, ReplayBuffer, create_safe_async_env, make_env, safe_step, choose_actions_batch, compute_q_loss
from Model import QNet, DuelingQNet

# -------- project path hack --------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import get_available_raster_indices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(checkpoint_path, model_class=QNet):
    """
    Load a training checkpoint and return all components
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    model_class : class
        Model class to use (QNet or DuelingQNet)
    
    Returns
    -------
    dict: Dictionary containing all loaded components
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract model dimensions from state dict
    # Look for the first conv layer to determine input dimensions
    first_layer_key = None
    for key in checkpoint['model_state_dict'].keys():
        if 'conv' in key.lower() and 'weight' in key:
            first_layer_key = key
            break
    
    if first_layer_key is None:
        raise ValueError("Could not determine model dimensions from checkpoint")
    
    # Assume standard landscape size - this could be made more robust
    H, W = 50, 50  # Default landscape size
    
    # Create models
    model = model_class(H, W).to(DEVICE)
    target_model = model_class(H, W).to(DEVICE)
    
    # Load model states
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    
    # Create optimizer and load state
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore training state
    meta_episode = checkpoint.get('meta_episode', 0)
    global_step = checkpoint.get('global_step', 0)
    eps = checkpoint.get('eps', 1.0)
    
    # Restore loss and episode tracking
    loss_window = deque(checkpoint.get('loss_window', []), maxlen=1000)
    episode_returns = deque(checkpoint.get('episode_returns', []), maxlen=100)
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Meta episode: {meta_episode}")
    print(f"  - Global step: {global_step}")
    print(f"  - Epsilon: {eps:.4f}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return {
        'model': model,
        'target_model': target_model,
        'optimizer': optimizer,
        'meta_episode': meta_episode,
        'global_step': global_step,
        'eps': eps,
        'loss_window': loss_window,
        'episode_returns': episode_returns,
        'H': H,
        'W': W
    }


def resume_training(checkpoint_path, total_episodes=500, model_class=QNet):
    """
    Resume training from a checkpoint
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    total_episodes : int
        Total number of episodes to train for
    model_class : class
        Model class to use (QNet or DuelingQNet)
    """
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Load checkpoint
    loaded = load_checkpoint(checkpoint_path, model_class)
    
    model = loaded['model']
    target_model = loaded['target_model']
    optimizer = loaded['optimizer']
    start_episode = loaded['meta_episode']
    global_step = loaded['global_step']
    eps = loaded['eps']
    loss_window = loaded['loss_window']
    episode_returns = loaded['episode_returns']
    H, W = loaded['H'], loaded['W']
    
    # Training hyperparameters (should match original training)
    EPISODES = total_episodes
    STEPS_PER_EP = 5
    BUFFER_CAP = 50_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    START_EPS = 1.0
    END_EPS = 0.05
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 1000
    SAVE_EVERY = 10
    
    # Environment parameters
    N_ENVS = 16
    BUDGET = 100
    K_STEPS = 10
    SIMS = 25
    
    print(f"\nResuming training from episode {start_episode + 1}/{EPISODES}")
    
    # Set up environments
    try:
        available_indices = get_available_raster_indices("cropped_raster")
        print(f"Found {len(available_indices)} available raster environments")
        if len(available_indices) < N_ENVS:
            available_indices = available_indices * ((N_ENVS // len(available_indices)) + 1)
    except Exception as e:
        print(f"Could not load raster indices: {e}")
        available_indices = list(range(N_ENVS * 2))
    
    env_fns = [
        make_env(available_indices, BUDGET, K_STEPS, SIMS, seed=i, env_id=i) 
        for i in range(N_ENVS)
    ]
    
    vec_env, reset_out = create_safe_async_env(env_fns)
    
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    
    # Create replay buffer (note: previous buffer contents are lost)
    buffer = ReplayBuffer(BUFFER_CAP)
    print(f"⚠ Note: Replay buffer restarted empty (previous buffer contents not saved)")
    
    print(f"\nResuming Deep Q-Network training...")
    print(f"Episodes remaining: {EPISODES - start_episode}")
    
    # Resume training loop
    for meta_ep in range(start_episode + 1, EPISODES + 1):
        print(f"META-EP: {meta_ep}/{EPISODES}")
        
        for step in range(STEPS_PER_EP):
            print(f"{step}/{STEPS_PER_EP}")
            
            actions = choose_actions_batch(model, obs, K_STEPS, eps)
            
            try:
                step_out = safe_step(vec_env, actions)
            except Exception as e:
                print(f"Critical error in environment step: {e}")
                print("Recreating environments...")
                vec_env.close()
                vec_env, reset_out = create_safe_async_env(env_fns)
                if isinstance(reset_out, tuple) and len(reset_out) == 2:
                    obs, _ = reset_out
                else:
                    obs = reset_out
                continue
            
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
                
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                if info_i and "episode_return" in info_i:
                    episode_returns.append(info_i['episode_return'])
                    print(f"[env {i}] return={info_i['episode_return']:.3f} len={info_i['episode_length']}")
            
            obs = next_obs
            global_step += N_ENVS
            
            # Epsilon decay
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac
            
            # Training step
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_q_loss(model, target_model, batch, GAMMA, K_STEPS)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                
                loss_window.append(loss.item())
                
                if global_step % TARGET_SYNC_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                    print(f"Target network updated at step {global_step}")
        
        # Episode statistics
        mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        
        print(f"[MetaEp {meta_ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} "
              f"buffer={len(buffer)} mean_loss={mean_loss:.4f} mean_return={mean_return:.3f}")
        
        # Save checkpoints
        if meta_ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'target_model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'meta_episode': meta_ep,
                'global_step': global_step,
                'eps': eps,
                'loss_window': list(loss_window),
                'episode_returns': list(episode_returns)
            }
            torch.save(checkpoint, f"checkpoints/dqn_checkpoint_ep{meta_ep}.pt")
            print(f"Checkpoint saved at episode {meta_ep}")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'meta_episode': EPISODES,
        'global_step': global_step,
        'training_complete': True
    }
    torch.save(final_checkpoint, "checkpoints/dqn_final.pt")
    
    vec_env.close()
    print("Training resumed and completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Resume DQN training from checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=500, 
                       help="Total episodes to train for (default: 500)")
    parser.add_argument("--model", choices=['qnet', 'dueling'], default='qnet',
                       help="Model architecture to use (default: qnet)")
    
    args = parser.parse_args()
    
    model_class = DuelingQNet if args.model == 'dueling' else QNet
    
    print("=" * 60)
    print("RESUMING DEEP Q-NETWORK TRAINING")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target episodes: {args.episodes}")
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    print()
    
    try:
        resume_training(args.checkpoint, args.episodes, model_class)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()