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
from Model import QNet, EnhancedQNet, DuelingQNet

mp.set_start_method("spawn", force=True)

# -------- project path hack --------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import RasterManager  # noqa: E402


Transition = namedtuple("Transition", "obs action reward next_obs done")


# ---------- Prioritized Experience Replay ----------
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling."""

    def __init__(self, capacity=100_000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return Transition(*zip(*samples)), indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ---------- Standard Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


# ---------- Enhanced Loss Functions ----------
def compute_enhanced_q_loss(model, target_model, batch, gamma, k, budget, weights=None, device="cpu"):
    """
    Enhanced Q-learning loss that balances effectiveness and efficiency.
    
    Args:
        model: Online Q-network
        target_model: Target Q-network
        batch: Batch of transitions
        gamma: Discount factor
        k: Number of fuel breaks per step
        budget: Total fuel break budget
        weights: Importance sampling weights (for prioritized replay)
        device: Computing device
    
    Returns:
        loss: Combined loss value
        td_errors: TD errors for priority updates
        metrics: Dictionary of loss components for logging
    """
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(device)
    next_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(device)
    a_t = torch.from_numpy(np.stack(batch.action)).long().to(device)
    r_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(device)
    d_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(device)

    q_all = model(obs_t)
    mask = a_t.bool()
    q_sel = torch.sum(q_all * mask.float(), dim=1) / k

    with torch.no_grad():
        # Double DQN: use online network to select actions, target network to evaluate
        next_online = model(next_t)
        best_idx = torch.argmax(next_online, dim=1)
        next_target = target_model(next_t)
        q_next = next_target.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        target = r_t + gamma * (1 - d_t.float()) * q_next

    # Standard TD loss
    td_errors = target - q_sel
    td_loss = td_errors.pow(2)

    # Efficiency bonus: encourage using fewer fuel breaks when possible
    # Count number of actions taken in each sample
    num_actions = torch.sum(mask.float(), dim=1) / k
    efficiency_factor = 1.0 - (num_actions / budget)  # Higher when using fewer breaks
    efficiency_bonus = 0.1 * efficiency_factor * torch.abs(td_errors)
    
    # Effectiveness penalty: penalize high Q-values when rewards are poor
    effectiveness_penalty = 0.05 * torch.clamp(q_sel - target, min=0) ** 2
    
    # Combined loss with balance between effectiveness and efficiency
    combined_loss = td_loss + effectiveness_penalty - efficiency_bonus

    if weights is not None:
        loss = (weights * combined_loss).mean()
    else:
        loss = combined_loss.mean()

    # Metrics for logging
    metrics = {
        'td_loss': td_loss.mean().item(),
        'efficiency_bonus': efficiency_bonus.mean().item(),
        'effectiveness_penalty': effectiveness_penalty.mean().item(),
        'combined_loss': combined_loss.mean().item(),
        'avg_q_value': q_sel.mean().item(),
        'avg_target': target.mean().item()
    }

    return loss, td_errors.abs().detach().cpu().numpy(), metrics


def compute_q_loss(model, target_model, batch, gamma, k, weights=None, device="cpu"):
    """Standard Q-learning loss with optional importance sampling weights."""
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(device)
    next_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(device)
    a_t = torch.from_numpy(np.stack(batch.action)).long().to(device)
    r_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(device)
    d_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(device)

    q_all = model(obs_t)
    mask = a_t.bool()
    q_sel = torch.sum(q_all * mask.float(), dim=1) / k

    with torch.no_grad():
        # Double DQN: use online network to select actions, target network to evaluate
        next_online = model(next_t)
        best_idx = torch.argmax(next_online, dim=1)
        next_target = target_model(next_t)
        q_next = next_target.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        target = r_t + gamma * (1 - d_t.float()) * q_next

    td_errors = target - q_sel

    if weights is not None:
        loss = (weights * td_errors.pow(2)).mean()
    else:
        loss = nn.MSELoss()(q_sel, target)

    return loss, td_errors.abs().detach().cpu().numpy()


# ---------- Env wrapper & maker ----------
import gym


class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ret = 0
        self._len = 0

    def reset(self, **kw):
        self._ret = 0
        self._len = 0
        return self.env.reset(**kw)

    def step(self, a):
        obs, r, d, tr, info = self.env.step(a)
        self._ret += float(r)
        self._len += 1
        if d or tr:
            info = dict(info)
            info["episode_return"] = self._ret
            info["episode_length"] = self._len
            obs, _ = self.env.reset()
            self._ret = 0
            self._len = 0
        return obs, r, d, tr, info


class RobustAutoResetWrapper(gym.Wrapper):
    """Auto-reset wrapper with error handling for robust training."""

    def __init__(self, env):
        super().__init__(env)
        self._ret = 0
        self._len = 0
        self._error_count = 0
        self._max_errors = 5

    def reset(self, **kw):
        self._ret = 0
        self._len = 0
        try:
            return self.env.reset(**kw)
        except Exception as e:
            print(f"Environment reset failed: {e}")
            self._error_count += 1
            if self._error_count > self._max_errors:
                # Return dummy observation if too many errors
                return self._get_dummy_obs(), {}
            return self.reset(**kw)

    def step(self, a):
        try:
            obs, r, d, tr, info = self.env.step(a)
            self._ret += float(r)
            self._len += 1

            if d or tr:
                info = dict(info)
                info["episode_return"] = self._ret
                info["episode_length"] = self._len
                try:
                    obs, _ = self.env.reset()
                except Exception as e:
                    print(f"Environment reset after episode failed: {e}")
                    obs = self._get_dummy_obs()
                self._ret = 0
                self._len = 0
            return obs, r, d, tr, info

        except Exception as e:
            print(f"Environment step failed: {e}")
            self._error_count += 1

            # Return safe dummy values
            obs = self._get_dummy_obs()
            reward = -1.0  # Negative reward for failed step
            done = True
            info = {
                "episode_return": self._ret,
                "episode_length": self._len,
                "error": True,
            }

            # Reset for next episode
            self._ret = 0
            self._len = 0

            return obs, reward, done, False, info

    def _get_dummy_obs(self):
        """Return a dummy observation that matches the expected shape."""
        try:
            return np.zeros((8, 50, 50), dtype=np.float32)
        except:
            return np.zeros((8, 50, 50), dtype=np.float32)


class DummyEnv(gym.Env):
    """Dummy environment that doesn't crash, for fallback when real env fails."""

    def __init__(self, budget, kstep, raster):
        super().__init__()
        self.budget = budget
        self.kstep = kstep
        self.H, self.W = 50, 50  # Default size

        # Try to get actual size from raster
        try:
            if isinstance(raster, dict) and "slp" in raster:
                self.H, self.W = raster["slp"].shape
        except:
            pass

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8, self.H, self.W), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiBinary(self.H * self.W)

        self.steps_taken = 0

    def reset(self, **kwargs):
        self.steps_taken = 0
        obs = (
            np.random.rand(8, self.H, self.W).astype(np.float32) * 0.1
        )  # Small random values
        return obs, {}

    def step(self, action):
        self.steps_taken += 1

        # Simple reward based on action
        action = np.asarray(action, dtype=np.int8).reshape(-1)
        num_actions = np.sum(action)

        # Give small negative reward (simulate fuel break cost)
        reward = -0.01 * num_actions

        # Episode ends when budget is reached or after reasonable number of steps
        done = (self.steps_taken >= self.budget // self.kstep) or (
            self.steps_taken > 50
        )

        # Next observation
        obs = np.random.rand(8, self.H, self.W).astype(np.float32) * 0.1

        info = {"burned": 100.0, "new_cells": min(num_actions, self.kstep)}

        return obs, reward, done, False, info


def make_env_with_raster(raster, budget, kstep, sims, seed):
    """Create environment with specific raster data."""

    def thunk():
        # Set random seed for reproducibility in subprocess
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        try:
            env = FuelBreakEnv(
                raster,
                break_budget=budget,
                break_step=kstep,
                num_simulations=sims,
                seed=seed,
            )
            return RobustAutoResetWrapper(env)
        except Exception as e:
            print(
                f"Warning: Environment creation failed with {e}, creating dummy environment"
            )
            # Create a dummy environment that doesn't crash
            return DummyEnv(budget, kstep, raster)

    return thunk


# ---------- Action helper ----------
def choose_actions_batch(model, obs_np, k, eps, device="cpu"):
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


# ---------- Learning Rate Scheduler ----------
class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ---------- main ----------
def main():
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")

    # CUDA memory optimization
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(
            f"Initial GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Hyperparams
    EPISODES = 1000  # Increased episodes to utilize all rasters
    STEPS_PER_EP = 3  # Increased steps per episode
    BUFFER_CAP = 100_000  # Increased buffer capacity
    BATCH_SIZE = 32  # Reduced from 64 to save memory
    GAMMA = 0.99
    LR = 3e-4  # Adjusted learning rate
    START_EPS = 1.0
    END_EPS = 0.01  # Lower final epsilon for better exploitation
    EPS_DECAY_STEPS = 100_000  # Longer epsilon decay
    TARGET_SYNC_EVERY = 1000  # Less frequent target updates for stability
    SAVE_EVERY = 25  # More frequent saving

    # Advanced training features
    USE_PRIORITIZED_REPLAY = True
    USE_ENHANCED_MODEL = True  # Use EnhancedQNet instead of basic QNet
    USE_DUELING = False  # Set to True to use DuelingQNet
    USE_LR_SCHEDULER = True

    # Memory optimization settings
    MEMORY_EFFICIENT = True  # Enable memory optimizations
    GRADIENT_ACCUMULATION_STEPS = (
        2  # Accumulate gradients to simulate larger batch size
    )

    # Vectorized env parameters
    N_ENVS = 16  # Reduced from 32 to save memory
    BUDGET = 200
    K_STEPS = 10
    SIMS = 10  # Reduced from 25 to improve stability

    # Raster management
    MAX_RASTERS = 500
    RASTER_ROOT = "cropped_raster"

    print("Initializing Raster Manager...")
    raster_manager = RasterManager(RASTER_ROOT, MAX_RASTERS)

    # Try to load all rasters
    try:
        all_rasters = raster_manager.load_all_rasters()
        print(f"Successfully loaded {len(all_rasters)} rasters")
        if len(all_rasters) == 0:
            print("No rasters found! Creating dummy raster for testing...")
            # Create a dummy raster for testing
            dummy_raster = {
                "slp": np.random.rand(50, 50).astype(np.float32),
                "asp": np.random.rand(50, 50).astype(np.float32),
                "dem": np.random.rand(50, 50).astype(np.float32),
                "cc": np.random.rand(50, 50).astype(np.float32),
                "cbd": np.random.rand(50, 50).astype(np.float32),
                "cbh": np.random.rand(50, 50).astype(np.float32),
                "ch": np.random.rand(50, 50).astype(np.float32),
                "fbfm": np.random.randint(1, 14, (50, 50)).astype(np.float32),
                "fireline_north": np.zeros((50, 50), dtype=np.float32),
                "fireline_east": np.zeros((50, 50), dtype=np.float32),
                "fireline_south": np.zeros((50, 50), dtype=np.float32),
                "fireline_west": np.zeros((50, 50), dtype=np.float32),
            }
            raster_manager.all_rasters = [dummy_raster] * N_ENVS
    except Exception as e:
        print(f"Error loading rasters: {e}")
        print("Creating dummy rasters for testing...")
        dummy_raster = {
            "slp": np.random.rand(50, 50).astype(np.float32),
            "asp": np.random.rand(50, 50).astype(np.float32),
            "dem": np.random.rand(50, 50).astype(np.float32),
            "cc": np.random.rand(50, 50).astype(np.float32),
            "cbd": np.random.rand(50, 50).astype(np.float32),
            "cbh": np.random.rand(50, 50).astype(np.float32),
            "ch": np.random.rand(50, 50).astype(np.float32),
            "fbfm": np.random.randint(1, 14, (50, 50)).astype(np.float32),
            "fireline_north": np.zeros((50, 50), dtype=np.float32),
            "fireline_east": np.zeros((50, 50), dtype=np.float32),
            "fireline_south": np.zeros((50, 50), dtype=np.float32),
            "fireline_west": np.zeros((50, 50), dtype=np.float32),
        }
        raster_manager.all_rasters = [dummy_raster] * N_ENVS

    # Get initial batch of rasters
    selected_rasters = raster_manager.get_random_rasters(N_ENVS)

    # Create environments with selected rasters
    env_fns = [
        make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i)
        for i, raster in enumerate(selected_rasters)
    ]
    vec_env = AsyncVectorEnv(env_fns)

    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    N_ENVS = obs.shape[0]
    _, C, H, W = obs.shape

    # Initialize model with memory considerations
    if USE_DUELING:
        model = DuelingQNet(H, W).to(DEVICE)
        tgt = DuelingQNet(H, W).to(DEVICE)
        print("Using Dueling DQN architecture")
    elif USE_ENHANCED_MODEL:
        if MEMORY_EFFICIENT:
            # Use lighter version of enhanced model for memory efficiency
            model = EnhancedQNet(
                H, W, use_attention=False, use_residual=True, use_multiscale=False
            ).to(DEVICE)
            tgt = EnhancedQNet(
                H, W, use_attention=False, use_residual=True, use_multiscale=False
            ).to(DEVICE)
            print("Using Memory-Efficient Enhanced DQN architecture (residual only)")
        else:
            model = EnhancedQNet(
                H, W, use_attention=True, use_residual=True, use_multiscale=True
            ).to(DEVICE)
            tgt = EnhancedQNet(
                H, W, use_attention=True, use_residual=True, use_multiscale=True
            ).to(DEVICE)
            print(
                "Using Full Enhanced DQN architecture with attention and residual connections"
            )
    else:
        model = QNet(H, W).to(DEVICE)
        tgt = QNet(H, W).to(DEVICE)
        print("Using basic DQN architecture")

    tgt.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # Learning rate scheduler
    if USE_LR_SCHEDULER:
        total_steps = EPISODES * STEPS_PER_EP * 10  # Approximate total training steps
        scheduler = CosineAnnealingWarmup(
            opt, warmup_steps=1000, total_steps=total_steps
        )

    # Initialize replay buffer
    if USE_PRIORITIZED_REPLAY:
        buf = PrioritizedReplayBuffer(BUFFER_CAP)
        print("Using Prioritized Experience Replay")
    else:
        buf = ReplayBuffer(BUFFER_CAP)
        print("Using standard Experience Replay")

    global_step = 0
    eps = START_EPS
    loss_win = deque(maxlen=1000)
    reward_win = deque(maxlen=100)
    
    # Enhanced metrics tracking
    burned_area_win = deque(maxlen=100)
    fuel_breaks_used_win = deque(maxlen=100)
    episode_length_win = deque(maxlen=100)
    
    # Loss component tracking
    td_loss_win = deque(maxlen=1000)
    efficiency_bonus_win = deque(maxlen=1000)
    effectiveness_penalty_win = deque(maxlen=1000)
    q_value_win = deque(maxlen=1000)

    # Training metrics
    best_avg_reward = float("-inf")
    best_avg_burned_area = float("inf")
    episodes_since_improvement = 0

    # Gradient accumulation tracking
    accumulated_loss = 0.0
    accumulation_steps = 0

    for ep in range(1, EPISODES + 1):
        print(f"META-EP: {ep}/{EPISODES}")

        # Periodically refresh environments with new rasters
        if ep % 10 == 0 and raster_manager.get_num_loaded_rasters() > N_ENVS:
            print("Refreshing environments with new rasters...")
            vec_env.close()
            selected_rasters = raster_manager.get_random_rasters(N_ENVS)
            env_fns = [
                make_env_with_raster(
                    raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS
                )
                for i, raster in enumerate(selected_rasters)
            ]
            vec_env = AsyncVectorEnv(env_fns)
            reset_out = vec_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        episode_rewards = []

        for step in range(STEPS_PER_EP):
            acts = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)

            try:
                vec_env.step_async(acts)

                # Add timeout to prevent hanging
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Environment step timed out")

                # Set timeout for step_wait (30 seconds)
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)

                try:
                    out = vec_env.step_wait()
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                # Handle different gym API versions
                if len(out) == 5:
                    nxt, rews, dones, truncs, infos = out
                    dones = np.logical_or(dones, truncs)
                else:
                    nxt, rews, dones, infos = out

                rews = np.asarray(rews, dtype=np.float32)
                if rews.shape != (N_ENVS,):
                    rews = rews.reshape(N_ENVS, -1).sum(axis=1)
                dones = np.asarray(dones, dtype=bool)

            except (EOFError, BrokenPipeError, ConnectionResetError, TimeoutError) as e:
                print(f"Environment communication error: {e}")
                print("Recreating environments...")

                # Close the problematic environment with force
                try:
                    vec_env.close()
                except:
                    pass

                # Kill any hanging processes
                import multiprocessing

                for p in multiprocessing.active_children():
                    try:
                        p.terminate()
                        p.join(timeout=2)
                        if p.is_alive():
                            p.kill()
                    except:
                        pass

                # Wait a moment for cleanup
                import time

                time.sleep(2)

                # Recreate environments
                selected_rasters = raster_manager.get_random_rasters(N_ENVS)
                env_fns = [
                    make_env_with_raster(
                        raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS + step
                    )
                    for i, raster in enumerate(selected_rasters)
                ]
                vec_env = AsyncVectorEnv(env_fns)
                reset_out = vec_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

                # Skip this step and continue
                continue

            # Store transitions and collect metrics
            for i in range(N_ENVS):
                buf.push(obs[i], acts[i], rews[i], nxt[i], dones[i])
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                
                # Track per-step metrics
                if info_i and isinstance(info_i, dict):
                    if "burned" in info_i:
                        burned_area_win.append(float(info_i["burned"]))
                    if "new_cells" in info_i:
                        fuel_breaks_used_win.append(float(info_i["new_cells"]))
                
                # Track episode completion metrics
                if info_i and "episode_return" in info_i:
                    episode_reward = float(info_i["episode_return"])
                    episode_length = int(info_i.get("episode_length", 0))
                    
                    # Ensure we have valid values before adding to windows
                    if not (np.isnan(episode_reward) or np.isinf(episode_reward)):
                        episode_rewards.append(episode_reward)
                        reward_win.append(episode_reward)
                        episode_length_win.append(episode_length)
                        
                        print(f"[env {i}] Episode completed: R={episode_reward:.3f}, L={episode_length}, "
                              f"Burned={info_i.get('burned', 'N/A')}")
                    else:
                        print(f"[env {i}] Invalid episode reward: {episode_reward}, skipping...")
                else:
                    # Track intermediate step rewards to prevent all NaN
                    if not (np.isnan(rews[i]) or np.isinf(rews[i])):
                        reward_win.append(float(rews[i]))

            obs = nxt
            global_step += N_ENVS

            # Update epsilon
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac

            # Training step with gradient accumulation
            if len(buf) >= BATCH_SIZE:
                if USE_PRIORITIZED_REPLAY:
                    batch, indices, weights = buf.sample(BATCH_SIZE)
                    weights = weights.to(DEVICE)
                    loss, td_errors, metrics = compute_enhanced_q_loss(
                        model, tgt, batch, GAMMA, K_STEPS, BUDGET, weights, DEVICE
                    )
                    buf.update_priorities(
                        indices, td_errors + 1e-6
                    )  # Small epsilon for numerical stability
                else:
                    batch = buf.sample(BATCH_SIZE)
                    loss, _, metrics = compute_enhanced_q_loss(
                        model, tgt, batch, GAMMA, K_STEPS, BUDGET, device=DEVICE
                    )

                # Gradient accumulation
                if MEMORY_EFFICIENT and GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    loss.backward()
                    accumulated_loss += loss.item()
                    accumulation_steps += 1

                    if accumulation_steps >= GRADIENT_ACCUMULATION_STEPS:
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=10.0
                        )
                        opt.step()
                        opt.zero_grad()

                        if USE_LR_SCHEDULER:
                            scheduler.step()

                        loss_win.append(accumulated_loss)
                        
                        # Track loss components if available
                        if 'metrics' in locals() and isinstance(metrics, dict):
                            td_loss_win.append(metrics.get('td_loss', 0.0))
                            efficiency_bonus_win.append(metrics.get('efficiency_bonus', 0.0))
                            effectiveness_penalty_win.append(metrics.get('effectiveness_penalty', 0.0))
                            q_value_win.append(metrics.get('avg_q_value', 0.0))
                        
                        accumulated_loss = 0.0
                        accumulation_steps = 0

                        # Update target network
                        if global_step % TARGET_SYNC_EVERY == 0:
                            tgt.load_state_dict(model.state_dict())
                else:
                    # Standard training step
                    opt.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    opt.step()

                    if USE_LR_SCHEDULER:
                        scheduler.step()

                    loss_win.append(loss.item())
                    
                    # Track loss components if available
                    if 'metrics' in locals() and isinstance(metrics, dict):
                        td_loss_win.append(metrics.get('td_loss', 0.0))
                        efficiency_bonus_win.append(metrics.get('efficiency_bonus', 0.0))
                        effectiveness_penalty_win.append(metrics.get('effectiveness_penalty', 0.0))
                        q_value_win.append(metrics.get('avg_q_value', 0.0))

                    # Update target network
                    if global_step % TARGET_SYNC_EVERY == 0:
                        tgt.load_state_dict(model.state_dict())

        # Episode statistics with robust NaN handling
        mean_loss = float(np.mean(loss_win)) if len(loss_win) > 0 else 0.0
        mean_reward = float(np.mean(reward_win)) if len(reward_win) > 0 else 0.0
        mean_burned_area = float(np.mean(burned_area_win)) if len(burned_area_win) > 0 else 0.0
        mean_fuel_breaks = float(np.mean(fuel_breaks_used_win)) if len(fuel_breaks_used_win) > 0 else 0.0
        mean_episode_length = float(np.mean(episode_length_win)) if len(episode_length_win) > 0 else 0.0
        
        # Loss component statistics
        mean_td_loss = float(np.mean(td_loss_win)) if len(td_loss_win) > 0 else 0.0
        mean_efficiency_bonus = float(np.mean(efficiency_bonus_win)) if len(efficiency_bonus_win) > 0 else 0.0
        mean_effectiveness_penalty = float(np.mean(effectiveness_penalty_win)) if len(effectiveness_penalty_win) > 0 else 0.0
        mean_q_value = float(np.mean(q_value_win)) if len(q_value_win) > 0 else 0.0
        
        current_lr = opt.param_groups[0]["lr"] if USE_LR_SCHEDULER else LR

        # Enhanced logging with performance metrics
        print(f"\n=== META-EPISODE {ep}/{EPISODES} SUMMARY ===")
        print(f"Training: steps={STEPS_PER_EP * N_ENVS}, eps={eps:.3f}, lr={current_lr:.2e}")
        print(f"Rewards: mean={mean_reward:.3f}, episodes_completed={len(episode_rewards)}")
        print(f"Environment: burned_area={mean_burned_area:.1f}, fuel_breaks_used={mean_fuel_breaks:.1f}, ep_length={mean_episode_length:.1f}")
        print(f"Loss Components: total={mean_loss:.4f}, td={mean_td_loss:.4f}, eff_bonus={mean_efficiency_bonus:.4f}, eff_penalty={mean_effectiveness_penalty:.4f}")
        print(f"Q-Values: mean={mean_q_value:.3f}")
        print("=" * 50)

        # Track best performance (considering both reward and burned area)
        performance_improved = False
        
        if mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            performance_improved = True
            print(f"🎉 New best average reward: {best_avg_reward:.3f}")
            
        if mean_burned_area < best_avg_burned_area and mean_burned_area > 0:
            best_avg_burned_area = mean_burned_area
            performance_improved = True
            print(f"🔥 New best (lowest) average burned area: {best_avg_burned_area:.1f}")
        
        if performance_improved:
            episodes_since_improvement = 0
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/qnet_best.pt")
            print(f"💾 Saving best model (reward: {best_avg_reward:.3f}, burned_area: {best_avg_burned_area:.1f})")
        else:
            episodes_since_improvement += 1

        # Regular checkpointing
        if ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_reward": best_avg_reward,
                "best_burned_area": best_avg_burned_area,
                "loss": mean_loss,
                "mean_reward": mean_reward,
                "mean_burned_area": mean_burned_area,
                "mean_fuel_breaks": mean_fuel_breaks,
                "episodes_since_improvement": episodes_since_improvement,
                "global_step": global_step,
                "epsilon": eps
            }
            torch.save(checkpoint, f"checkpoints/qnet_ep{ep}.pt")
            print(f"💾 Checkpoint saved for epoch {ep} (reward: {mean_reward:.3f}, burned: {mean_burned_area:.1f})")

            # Clear CUDA cache periodically
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                print(
                    f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )

    # Final save
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_reward": best_avg_reward,
        "best_burned_area": best_avg_burned_area,
        "total_episodes": EPISODES,
        "global_step": global_step,
        "final_epsilon": eps
    }
    torch.save(final_checkpoint, "checkpoints/qnet_final.pt")
    print(f"💾 Final model saved! Best reward: {best_avg_reward:.3f}, Best burned area: {best_avg_burned_area:.1f}")

    vec_env.close()
    print(f"Training completed! Best average reward: {best_avg_reward:.3f}")


if __name__ == "__main__":
    main()
