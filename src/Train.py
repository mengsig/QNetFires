import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from gym.vector import AsyncVectorEnv
import multiprocessing as mp
import signal
import time

from Env import FuelBreakEnv
from Model import QNet

# -------- project path hack --------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import load_all_rasters, get_available_raster_indices  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Transition = namedtuple("Transition", "obs action reward next_obs done")
print(f"Using device {DEVICE}...")


# ---------- Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


# ---------- Loss ----------
def compute_q_loss(model, target_model, batch, gamma, k):
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(DEVICE)
    next_obs_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(DEVICE)
    action_t = torch.from_numpy(np.stack(batch.action)).long().to(DEVICE)
    reward_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(DEVICE)
    done_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(DEVICE)

    q_all = model(obs_t)  # (B,H*W)
    act_mask = action_t.bool()
    q_selected = torch.sum(q_all * act_mask.float(), dim=1) / k

    with torch.no_grad():
        next_q_all_online = model(next_obs_t)
        best_idx = torch.argmax(next_q_all_online, dim=1)
        next_q_all_target = target_model(next_obs_t)
        next_q = next_q_all_target.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        target = reward_t + gamma * (1 - done_t.float()) * next_q

    return nn.MSELoss()(q_selected, target)


# ---------- Env wrappers ----------
import gym


class AutoResetWrapper(gym.Wrapper):
    """
    Auto-resets env when done/truncated is True.
    Adds keys to info:
      info["final_observation"] : obs before reset
      info["episode_return"]    : accumulated reward of finished episode
      info["episode_length"]    : steps in finished episode
    """

    def __init__(self, env):
        super().__init__(env)
        self._ep_ret = 0.0
        self._ep_len = 0

    def reset(self, **kwargs):
        self._ep_ret = 0.0
        self._ep_len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self._ep_ret += float(rew)
        self._ep_len += 1

        if done or trunc:
            info = dict(info)  # copy
            info["final_observation"] = obs
            info["episode_return"] = self._ep_ret
            info["episode_length"] = self._ep_len
            obs, _ = self.env.reset()
            self._ep_ret = 0.0
            self._ep_len = 0

        return obs, rew, done, trunc, info


def make_env(raster_indices, budget, kstep, sims, seed, env_id):
    def _thunk():
        try:
            # Randomly select a raster index for this environment
            raster_idx = random.choice(raster_indices)
            rasters = load_all_rasters("cropped_raster", raster_idx)
            
            env = FuelBreakEnv(
                rasters,
                break_budget=budget,
                break_step=kstep,
                num_simulations=sims,
                seed=seed,
            )
            # wrap to auto-reset
            env = AutoResetWrapper(env)
            return env
        except Exception as e:
            print(f"Error creating environment {env_id}: {e}")
            raise e

    return _thunk


# ---------- Action helper ----------
def choose_actions_batch(model, obs_np, k, eps):
    """
    obs_np: (N,C,H,W)
    returns actions: (N,H*W) int8
    """
    N, C, H, W = obs_np.shape
    HxW = H * W
    actions = np.zeros((N, HxW), dtype=np.int8)

    with torch.no_grad():
        q_all = model(torch.from_numpy(obs_np).to(DEVICE)).cpu().numpy()  # (N,H*W)

    for i in range(N):
        if random.random() < eps:
            idx = np.random.choice(HxW, size=k, replace=False)
        else:
            qi = q_all[i]
            idx = np.argpartition(qi, -k)[-k:]
        actions[i, idx] = 1
    return actions


# ---------- Safe environment handling ----------
def create_safe_async_env(env_fns, max_retries=3):
    """Create AsyncVectorEnv with error handling and retries"""
    for attempt in range(max_retries):
        try:
            vec_env = AsyncVectorEnv(env_fns, context='spawn')
            # Test the environment
            reset_out = vec_env.reset()
            return vec_env, reset_out
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to create environment: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                raise e


def safe_step(vec_env, actions, max_retries=3):
    """Safely execute environment step with error handling"""
    for attempt in range(max_retries):
        try:
            vec_env.step_async(actions)
            step_out = vec_env.step_wait()
            return step_out
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            print(f"Environment step failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1)  # Brief wait before retry
                # Try to reset the problematic environments
                try:
                    vec_env.reset()
                except:
                    pass
            else:
                raise e


# ---------- main ----------
def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Hyperparams
    EPISODES = 500  # Outer training loops ("meta-episodes"); each runs STEPS_PER_EP interaction steps
    STEPS_PER_EP = 5  # Env interaction steps per meta-episode (each step advances ALL N_ENVS envs once)
    BUFFER_CAP = 50_000  # Max number of transitions kept in the replay buffer (old ones are dropped FIFO)
    BATCH_SIZE = (
        32  # Increased batch size for better training stability
    )
    GAMMA = 0.99  # Discount factor for future rewards in the Bellman target
    LR = 1e-4  # Learning rate for the optimizer (Adam here)
    START_EPS = 1.0  # Initial epsilon for ε-greedy action selection (full exploration)
    END_EPS = 0.05  # Final/minimum epsilon after decay (some exploration kept)
    EPS_DECAY_STEPS = 50_000  # Number of env-steps (sum over all envs) to anneal epsilon from START_EPS to END_EPS
    TARGET_SYNC_EVERY = (
        1000  # How many env-steps between copying online net weights to the target net
    )
    SAVE_EVERY = 10  # Save a model checkpoint every this many meta-episodes

    # Vectorized env parameters
    N_ENVS = (
        16  # Number of environments run in parallel (processes with AsyncVectorEnv)
    )
    BUDGET = 100  # FuelBreakEnv: total number of cells you're allowed to place across an episode
    K_STEPS = 10  # FuelBreakEnv: how many new fuel-break cells you can place per step (action size)
    SIMS = 25  # FuelBreakEnv: number of Monte-Carlo fire simulations per step to estimate reward

    # Get available raster indices and ensure random selection
    try:
        available_indices = get_available_raster_indices("cropped_raster")
        print(f"Found {len(available_indices)} available raster environments")
        if len(available_indices) < N_ENVS:
            print(f"Warning: Only {len(available_indices)} rasters available, but need {N_ENVS} environments")
            # Duplicate indices if needed
            available_indices = available_indices * ((N_ENVS // len(available_indices)) + 1)
    except Exception as e:
        print(f"Could not load raster indices: {e}")
        print("Using fallback indices 0 to N_ENVS-1")
        available_indices = list(range(N_ENVS * 2))  # Provide more options for random selection

    # Create environment functions with random raster selection
    env_fns = [
        make_env(available_indices, BUDGET, K_STEPS, SIMS, seed=i, env_id=i) 
        for i in range(N_ENVS)
    ]
    
    # Create vectorized environment with error handling
    vec_env, reset_out = create_safe_async_env(env_fns)

    # Handle reset output
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    N_ENVS = obs.shape[0]
    _, C, H, W = obs.shape

    # Initialize DQN with improved architecture
    model = QNet(H, W).to(DEVICE)
    target_model = QNet(H, W).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    
    # Use Adam optimizer with weight decay for better generalization
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    buffer = ReplayBuffer(BUFFER_CAP)

    global_step = 0
    eps = START_EPS

    # rolling stats
    loss_window = deque(maxlen=1000)
    episode_returns = deque(maxlen=100)

    print("Starting Deep Q-Network training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    for meta_ep in range(1, EPISODES + 1):
        print(f"META-EP: {meta_ep}/{EPISODES}")

        for step in range(STEPS_PER_EP):
            print(f"{step}/{STEPS_PER_EP}")
            
            # Choose actions using epsilon-greedy policy
            actions = choose_actions_batch(model, obs, K_STEPS, eps)

            # Safe environment step
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

            # handle both 4 and 5 return values
            if len(step_out) == 5:
                next_obs, rewards, dones, truncs, infos = step_out
                dones = np.logical_or(dones, truncs)
            else:
                next_obs, rewards, dones, infos = step_out

            rewards = np.asarray(rewards, dtype=np.float32)
            if rewards.shape != (N_ENVS,):
                rewards = rewards.reshape(N_ENVS, -1).sum(axis=1)
            dones = np.asarray(dones, dtype=bool)

            # store transitions in replay buffer
            for i in range(N_ENVS):
                buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

                # Track episode statistics
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                if info_i and "episode_return" in info_i:
                    episode_returns.append(info_i['episode_return'])
                    print(
                        f"[env {i}] return={info_i['episode_return']:.3f} len={info_i['episode_length']}"
                    )

            obs = next_obs
            global_step += N_ENVS

            # epsilon decay (linear annealing)
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac

            # DQN optimization step
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_q_loss(model, target_model, batch, GAMMA, K_STEPS)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                loss_window.append(loss.item())

                # Update target network (Double DQN)
                if global_step % TARGET_SYNC_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                    print(f"Target network updated at step {global_step}")

        # Episode statistics
        mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        
        print(
            f"[MetaEp {meta_ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} "
            f"buffer={len(buffer)} mean_loss={mean_loss:.4f} mean_return={mean_return:.3f}"
        )

        # Save model checkpoints
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
    
    # Clean up
    vec_env.close()
    print("Deep Q-Network training completed successfully!")


if __name__ == "__main__":
    main()
