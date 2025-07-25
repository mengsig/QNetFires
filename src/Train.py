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


# ---------- Spatial Analysis Helper Functions ----------
def compute_spatial_metrics(actions_batch, H, W, device="cpu"):
    """
    Compute spatial structure metrics for fuel break placement.
    
    Args:
        actions_batch: Batch of action masks (B, H*W)
        H, W: Grid dimensions
        device: Computing device
    
    Returns:
        connectivity_score: How connected the fuel breaks are (higher = more line-like)
        compactness_penalty: Penalty for blob-like structures (higher = more blob-like)
        edge_ratio: Ratio of perimeter to area (higher = more line-like)
    """
    batch_size = actions_batch.shape[0]
    actions_2d = actions_batch.view(batch_size, H, W).float()
    
    # Connectivity score: count adjacent fuel breaks
    # Use convolution to count neighbors
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).to(device)
    kernel = kernel.view(1, 1, 3, 3)
    
    # Pad actions for convolution
    padded_actions = torch.nn.functional.pad(actions_2d.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)
    neighbor_counts = torch.nn.functional.conv2d(padded_actions, kernel, padding=0)
    neighbor_counts = neighbor_counts.squeeze(1)
    
    # Connectivity: sum of (fuel_break * number_of_neighbors)
    connectivity = torch.sum(actions_2d * neighbor_counts, dim=(1, 2))
    total_breaks = torch.sum(actions_2d, dim=(1, 2)) + 1e-6  # Avoid division by zero
    connectivity_score = connectivity / total_breaks
    
    # Simplified perimeter calculation using neighbor differences
    # Shift actions in 4 directions and count differences
    actions_padded = torch.nn.functional.pad(actions_2d, (1, 1, 1, 1), mode='constant', value=0)
    
    # Calculate differences with neighbors (edge detection)
    diff_up = torch.abs(actions_2d - actions_padded[:, :-2, 1:-1])
    diff_down = torch.abs(actions_2d - actions_padded[:, 2:, 1:-1])
    diff_left = torch.abs(actions_2d - actions_padded[:, 1:-1, :-2])
    diff_right = torch.abs(actions_2d - actions_padded[:, 1:-1, 2:])
    
    # Sum all edge differences (perimeter approximation)
    perimeter = torch.sum(diff_up + diff_down + diff_left + diff_right, dim=(1, 2))
    
    # Edge ratio: perimeter / area (higher for lines, lower for blobs)
    edge_ratio = perimeter / total_breaks
    
    # Compactness penalty: area^2 / perimeter (higher for blobs)
    compactness_penalty = (total_breaks**2) / (perimeter + 1e-6)
    
    return connectivity_score, compactness_penalty, edge_ratio


# ---------- Enhanced Loss Functions ----------
def compute_curriculum_q_loss(model, target_model, batch, gamma, k, budget, global_step=0, weights=None, device="cpu"):
    """
    Curriculum-based Q-learning loss that transitions from geometric guidance to performance optimization.
    
    Phase 1 (0-30k steps): Strong geometric guidance to escape blob local minima
    Phase 2 (30k-60k steps): Gradual transition to performance-based learning  
    Phase 3 (60k+ steps): Pure performance optimization
    
    Final loss components:
    - Immediate Improvement (40%): Each fuel break should reduce burned area
    - Total Efficiency (40%): Overall reduction from baseline
    - Fuel Break Efficiency (20%): Penalizes wasteful placement
    
    Args:
        model: Online Q-network
        target_model: Target Q-network
        batch: Batch of transitions
        gamma: Discount factor
        k: Number of fuel breaks per step
        budget: Total fuel break budget
        global_step: Current training step (for curriculum progression)
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

    batch_size, action_dim = a_t.shape
    H = W = int(np.sqrt(action_dim))  # Assuming square grid
    
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

    # Standard TD loss (always present)
    td_errors = target - q_sel
    td_loss = td_errors.pow(2)

    # === CURRICULUM PROGRESSION ===
    # Phase 1: 0-30k steps (strong geometric guidance)
    # Phase 2: 30k-60k steps (transition period) 
    # Phase 3: 60k+ steps (pure performance)
    
    geometric_phase_end = 30000
    transition_phase_end = 60000
    
    if global_step < geometric_phase_end:
        # Phase 1: Strong geometric guidance (like before)
        geometric_weight = 1.0
        performance_weight = 0.3
        curriculum_phase = "GEOMETRIC"
    elif global_step < transition_phase_end:
        # Phase 2: Linear transition
        progress = (global_step - geometric_phase_end) / (transition_phase_end - geometric_phase_end)
        geometric_weight = 1.0 - progress  # 1.0 -> 0.0
        performance_weight = 0.3 + 0.7 * progress  # 0.3 -> 1.0
        curriculum_phase = "TRANSITION"
    else:
        # Phase 3: Pure performance optimization
        geometric_weight = 0.0
        performance_weight = 1.0
        curriculum_phase = "PERFORMANCE"

    # === GEOMETRIC COMPONENTS (fade out over time) ===
    geometric_loss = torch.zeros_like(td_loss)
    connectivity_score = torch.zeros(batch_size).to(device)
    compactness_penalty_val = torch.zeros(batch_size).to(device)
    edge_ratio = torch.zeros(batch_size).to(device)
    
    if geometric_weight > 0:
        # Compute spatial metrics only when needed
        connectivity_score, compactness_penalty_val, edge_ratio = compute_spatial_metrics(
            mask.float(), H, W, device
        )
        
        # Progressive anti-blob strategy: start very strong, then fade
        # Early phase: Heavy blob punishment to force escape from local minima
        # Later phase: Lighter guidance to allow performance optimization
        
        if global_step < geometric_phase_end // 2:  # First half of geometric phase (0-15k steps)
            # VERY STRONG anti-blob bias
            blob_penalty_weight = 0.5  # 5x stronger than before
            connectivity_bonus_weight = 0.25
            line_bonus_weight = 0.15
            phase_name = "ANTI-BLOB"
        else:  # Second half of geometric phase (15k-30k steps)
            # Moderate geometric guidance
            blob_penalty_weight = 0.2
            connectivity_bonus_weight = 0.15
            line_bonus_weight = 0.1
            phase_name = "GEOMETRIC"
        
        # Apply the weights
        connectivity_bonus = connectivity_bonus_weight * connectivity_score * torch.abs(td_errors)
        blob_penalty = blob_penalty_weight * compactness_penalty_val * td_errors.pow(2)
        line_bonus = line_bonus_weight * edge_ratio * torch.abs(td_errors)
        
        # Add extra punishment for very compact structures in early phase
        if global_step < geometric_phase_end // 2:
            # Extra penalty for extremely blob-like structures (compactness > 5)
            extreme_blob_penalty = 0.3 * torch.clamp(compactness_penalty_val - 5.0, min=0) * td_errors.pow(2)
            blob_penalty = blob_penalty + extreme_blob_penalty
        
        geometric_loss = blob_penalty - connectivity_bonus - line_bonus

    # === PERFORMANCE COMPONENTS (strengthen over time) ===
    
    # 1. Immediate Improvement (40% weight in final phase)
    # Reward negative rewards (fire reduction) more strongly
    immediate_improvement = 0.4 * torch.clamp(-r_t, min=0) * torch.abs(td_errors)
    
    # 2. Total Efficiency (40% weight in final phase)  
    # Encourage consistent performance across episodes
    # Use cumulative reward as proxy for total efficiency
    baseline_performance = -0.1  # Assume baseline without fuel breaks
    efficiency_factor = torch.clamp((baseline_performance - r_t) / abs(baseline_performance), min=0, max=2)
    total_efficiency = 0.4 * efficiency_factor * torch.abs(td_errors)
    
    # 3. Fuel Break Efficiency (20% weight in final phase)
    # Penalize using too many fuel breaks for small improvements
    num_actions = torch.sum(mask.float(), dim=1)
    fuel_break_efficiency = num_actions / (budget + 1e-6)  # Normalized usage
    
    # Efficiency milestones (bonus for significant reductions)
    milestone_bonus = torch.zeros_like(r_t)
    reduction_ratio = torch.clamp(-r_t / abs(baseline_performance), min=0, max=1)
    
    # Milestone bonuses: 30%, 50%, 70% reduction
    milestone_30 = (reduction_ratio >= 0.3).float() * 0.05
    milestone_50 = (reduction_ratio >= 0.5).float() * 0.08  
    milestone_70 = (reduction_ratio >= 0.7).float() * 0.12
    milestone_bonus = (milestone_30 + milestone_50 + milestone_70) * torch.abs(td_errors)
    
    # Wasteful placement penalty
    wasteful_penalty = 0.2 * fuel_break_efficiency * td_errors.pow(2)
    
    performance_loss = wasteful_penalty - immediate_improvement - total_efficiency - milestone_bonus

    # === COMBINED LOSS ===
    combined_loss = (td_loss + 
                    geometric_weight * geometric_loss + 
                    performance_weight * performance_loss)

    if weights is not None:
        loss = (weights * combined_loss).mean()
    else:
        loss = combined_loss.mean()

    # Comprehensive metrics for monitoring
    metrics = {
        'td_loss': td_loss.mean().item(),
        'geometric_loss': geometric_loss.mean().item(),
        'performance_loss': performance_loss.mean().item(),
        'combined_loss': combined_loss.mean().item(),
        'immediate_improvement': immediate_improvement.mean().item(),
        'total_efficiency': total_efficiency.mean().item(),
        'wasteful_penalty': wasteful_penalty.mean().item(),
        'milestone_bonus': milestone_bonus.mean().item(),
        'avg_q_value': q_sel.mean().item(),
        'avg_target': target.mean().item(),
        'avg_reward': r_t.mean().item(),
        'avg_fuel_breaks_used': num_actions.mean().item(),
        'avg_efficiency_ratio': efficiency_factor.mean().item(),
        
        # Geometric metrics (will be zero in performance phase)
        'avg_connectivity': connectivity_score.mean().item(),
        'avg_compactness': compactness_penalty_val.mean().item(), 
        'avg_edge_ratio': edge_ratio.mean().item(),
        'blob_penalty': blob_penalty.mean().item() if geometric_weight > 0 else 0.0,
        'connectivity_bonus': connectivity_bonus.mean().item() if geometric_weight > 0 else 0.0,
        'line_bonus': line_bonus.mean().item() if geometric_weight > 0 else 0.0,
        
        # Curriculum tracking
        'geometric_weight': geometric_weight,
        'performance_weight': performance_weight,
        'curriculum_phase': curriculum_phase,
        'detailed_phase': phase_name if geometric_weight > 0 else "PERFORMANCE",
        'global_step': global_step
    }

    return loss, td_errors.abs().detach().cpu().numpy(), metrics


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
    EPISODES = 20000  # Increased episodes to utilize all rasters
    STEPS_PER_EP = 10  # Increased steps per episode
    BUFFER_CAP = 100_000  # Increased buffer capacity
    BATCH_SIZE = 32  # Reduced from 64 to save memory
    GAMMA = 0.997
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
    USE_SPATIAL_LOSS = True  # Use spatial-aware loss to encourage line structures

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
            opt, warmup_steps=500, total_steps=total_steps
        )

    # Initialize replay buffer
    if USE_PRIORITIZED_REPLAY:
        buf = PrioritizedReplayBuffer(BUFFER_CAP)
        print("Using Prioritized Experience Replay")
    else:
        buf = ReplayBuffer(BUFFER_CAP)
        print("Using standard Experience Replay")
        
    if USE_SPATIAL_LOSS:
        print("🎓 Using Enhanced Curriculum-Based Loss Function:")
        print("   Phase 1a (0-15k steps): 🚫 HEAVY anti-blob punishment to escape local minima")
        print("   Phase 1b (15k-30k steps): 🎯 Moderate geometric guidance for structure")
        print("   Phase 2 (30k-60k steps): 🔄 Gradual transition to performance optimization")
        print("   Phase 3 (60k+ steps): 🎖️ Pure performance focus (immediate improvement + total efficiency)")
    else:
        print("📊 Using Enhanced Loss Function (basic efficiency/effectiveness)")

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
    
    # Curriculum and performance tracking
    immediate_improvement_win = deque(maxlen=1000)
    total_efficiency_win = deque(maxlen=1000)
    wasteful_penalty_win = deque(maxlen=1000)
    milestone_bonus_win = deque(maxlen=1000)
    geometric_loss_win = deque(maxlen=1000)
    performance_loss_win = deque(maxlen=1000)
    
    # Spatial structure tracking (for geometric phase)
    connectivity_win = deque(maxlen=1000)
    compactness_win = deque(maxlen=1000)
    edge_ratio_win = deque(maxlen=1000)
    blob_penalty_win = deque(maxlen=1000)
    connectivity_bonus_win = deque(maxlen=1000)
    line_bonus_win = deque(maxlen=1000)

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
                        burned_val = info_i["burned"]
                        if isinstance(burned_val, (np.ndarray, list)):
                            if np.size(burned_val) == 1:
                                burned_val = burned_val.item() if hasattr(burned_val, 'item') else burned_val[0]
                            else:
                                burned_val = float(np.mean(burned_val))  # Take mean for multi-element arrays
                        burned_area_win.append(float(burned_val))
                    if "new_cells" in info_i:
                        cells_val = info_i["new_cells"]
                        if isinstance(cells_val, (np.ndarray, list)):
                            if np.size(cells_val) == 1:
                                cells_val = cells_val.item() if hasattr(cells_val, 'item') else cells_val[0]
                            else:
                                cells_val = float(np.sum(cells_val))  # Take sum for multi-element arrays
                        fuel_breaks_used_win.append(float(cells_val))
                
                # Track episode completion metrics
                if info_i and "episode_return" in info_i:
                    reward_val = info_i["episode_return"]
                    if isinstance(reward_val, (np.ndarray, list)):
                        if np.size(reward_val) == 1:
                            reward_val = reward_val.item() if hasattr(reward_val, 'item') else reward_val[0]
                        else:
                            reward_val = float(np.sum(reward_val))  # Take sum for multi-element arrays
                    episode_reward = float(reward_val)
                    
                    length_val = info_i.get("episode_length", 0)
                    if isinstance(length_val, (np.ndarray, list)):
                        if np.size(length_val) == 1:
                            length_val = length_val.item() if hasattr(length_val, 'item') else length_val[0]
                        else:
                            length_val = int(np.max(length_val))  # Take max for multi-element arrays
                    episode_length = int(length_val)
                    
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
                    reward_val = rews[i]
                    if isinstance(reward_val, (np.ndarray, list)):
                        if np.size(reward_val) == 1:
                            reward_val = reward_val.item() if hasattr(reward_val, 'item') else reward_val[0]
                        else:
                            reward_val = float(np.sum(reward_val))  # Take sum for multi-element arrays
                    if not (np.isnan(reward_val) or np.isinf(reward_val)):
                        reward_win.append(float(reward_val))

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
                    if USE_SPATIAL_LOSS:
                        loss, td_errors, metrics = compute_curriculum_q_loss(
                            model, tgt, batch, GAMMA, K_STEPS, BUDGET, global_step, weights, DEVICE
                        )
                    else:
                        loss, td_errors, metrics = compute_enhanced_q_loss(
                            model, tgt, batch, GAMMA, K_STEPS, BUDGET, weights, DEVICE
                        )
                    buf.update_priorities(
                        indices, td_errors + 1e-6
                    )  # Small epsilon for numerical stability
                else:
                    batch = buf.sample(BATCH_SIZE)
                    if USE_SPATIAL_LOSS:
                        loss, _, metrics = compute_curriculum_q_loss(
                            model, tgt, batch, GAMMA, K_STEPS, BUDGET, global_step, device=DEVICE
                        )
                    else:
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
                            
                            # Track curriculum metrics
                            immediate_improvement_win.append(metrics.get('immediate_improvement', 0.0))
                            total_efficiency_win.append(metrics.get('total_efficiency', 0.0))
                            wasteful_penalty_win.append(metrics.get('wasteful_penalty', 0.0))
                            milestone_bonus_win.append(metrics.get('milestone_bonus', 0.0))
                            geometric_loss_win.append(metrics.get('geometric_loss', 0.0))
                            performance_loss_win.append(metrics.get('performance_loss', 0.0))
                            
                            # Track spatial metrics (geometric phase)
                            connectivity_win.append(metrics.get('avg_connectivity', 0.0))
                            compactness_win.append(metrics.get('avg_compactness', 0.0))
                            edge_ratio_win.append(metrics.get('avg_edge_ratio', 0.0))
                            blob_penalty_win.append(metrics.get('blob_penalty', 0.0))
                            connectivity_bonus_win.append(metrics.get('connectivity_bonus', 0.0))
                            line_bonus_win.append(metrics.get('line_bonus', 0.0))
                        
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
                        
                        # Track curriculum metrics
                        immediate_improvement_win.append(metrics.get('immediate_improvement', 0.0))
                        total_efficiency_win.append(metrics.get('total_efficiency', 0.0))
                        wasteful_penalty_win.append(metrics.get('wasteful_penalty', 0.0))
                        milestone_bonus_win.append(metrics.get('milestone_bonus', 0.0))
                        geometric_loss_win.append(metrics.get('geometric_loss', 0.0))
                        performance_loss_win.append(metrics.get('performance_loss', 0.0))
                        
                        # Track spatial metrics (geometric phase)
                        connectivity_win.append(metrics.get('avg_connectivity', 0.0))
                        compactness_win.append(metrics.get('avg_compactness', 0.0))
                        edge_ratio_win.append(metrics.get('avg_edge_ratio', 0.0))
                        blob_penalty_win.append(metrics.get('blob_penalty', 0.0))
                        connectivity_bonus_win.append(metrics.get('connectivity_bonus', 0.0))
                        line_bonus_win.append(metrics.get('line_bonus', 0.0))

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
        
        # Curriculum and performance statistics
        mean_immediate_improvement = float(np.mean(immediate_improvement_win)) if len(immediate_improvement_win) > 0 else 0.0
        mean_total_efficiency = float(np.mean(total_efficiency_win)) if len(total_efficiency_win) > 0 else 0.0
        mean_wasteful_penalty = float(np.mean(wasteful_penalty_win)) if len(wasteful_penalty_win) > 0 else 0.0
        mean_milestone_bonus = float(np.mean(milestone_bonus_win)) if len(milestone_bonus_win) > 0 else 0.0
        mean_geometric_loss = float(np.mean(geometric_loss_win)) if len(geometric_loss_win) > 0 else 0.0
        mean_performance_loss = float(np.mean(performance_loss_win)) if len(performance_loss_win) > 0 else 0.0
        
        # Spatial structure statistics (for geometric phase)
        mean_connectivity = float(np.mean(connectivity_win)) if len(connectivity_win) > 0 else 0.0
        mean_compactness = float(np.mean(compactness_win)) if len(compactness_win) > 0 else 0.0
        mean_edge_ratio = float(np.mean(edge_ratio_win)) if len(edge_ratio_win) > 0 else 0.0
        mean_blob_penalty = float(np.mean(blob_penalty_win)) if len(blob_penalty_win) > 0 else 0.0
        mean_connectivity_bonus = float(np.mean(connectivity_bonus_win)) if len(connectivity_bonus_win) > 0 else 0.0
        mean_line_bonus = float(np.mean(line_bonus_win)) if len(line_bonus_win) > 0 else 0.0
        
        current_lr = opt.param_groups[0]["lr"] if USE_LR_SCHEDULER else LR

        # Enhanced logging with curriculum progress
        print(f"\n=== META-EPISODE {ep}/{EPISODES} SUMMARY ===")
        print(f"Training: steps={STEPS_PER_EP * N_ENVS}, eps={eps:.3f}, lr={current_lr:.2e}, global_step={global_step}")
        print(f"Rewards: mean={mean_reward:.3f}, episodes_completed={len(episode_rewards)}")
        print(f"Environment: burned_area={mean_burned_area:.1f}, fuel_breaks_used={mean_fuel_breaks:.1f}, ep_length={mean_episode_length:.1f}")
        
        # Detailed curriculum phase detection
        if global_step < 15000:
            curriculum_phase = "🚫 ANTI-BLOB"
            phase_progress = global_step / 15000 * 100
        elif global_step < 30000:
            curriculum_phase = "🎯 GEOMETRIC"
            phase_progress = (global_step - 15000) / 15000 * 100
        elif global_step < 60000:
            curriculum_phase = "🔄 TRANSITION" 
            phase_progress = (global_step - 30000) / 30000 * 100
        else:
            curriculum_phase = "🎖️ PERFORMANCE"
            phase_progress = 100.0
            
        print(f"Curriculum: {curriculum_phase} ({phase_progress:.1f}%)")
        print(f"Loss Components: total={mean_loss:.4f}, geometric={mean_geometric_loss:.4f}, performance={mean_performance_loss:.4f}")
        print(f"Performance Metrics: improvement={mean_immediate_improvement:.4f}, efficiency={mean_total_efficiency:.4f}, wasteful={mean_wasteful_penalty:.4f}")
        print(f"Bonuses: milestones={mean_milestone_bonus:.4f}")
        
        # Show spatial metrics only during geometric phase
        if global_step < 60000:
            print(f"Spatial Metrics: connectivity={mean_connectivity:.3f}, edge_ratio={mean_edge_ratio:.3f}, compactness={mean_compactness:.3f}")
            print(f"Spatial Forces: blob_penalty={mean_blob_penalty:.4f}, connectivity_bonus={mean_connectivity_bonus:.4f}, line_bonus={mean_line_bonus:.4f}")
            
            # Visual indicators for spatial structure quality
            if mean_connectivity > 2.0:
                structure_indicator = "🔗 CONNECTED"
            elif mean_connectivity > 1.0:
                structure_indicator = "➖ PARTIAL"
            else:
                structure_indicator = "⚪ SCATTERED"
                
            if mean_edge_ratio > 3.0:
                shape_indicator = "📏 LINE-LIKE"
            elif mean_edge_ratio > 2.0:
                shape_indicator = "🔄 MIXED"
            else:
                shape_indicator = "🔵 BLOB-LIKE"
                
            # Special indicator for anti-blob phase
            if global_step < 15000:
                blob_strength = "💪 HEAVY" if mean_blob_penalty > 0.1 else "⚡ MODERATE"
                print(f"Anti-Blob Strength: {blob_strength} | Structure: {structure_indicator} | Shape: {shape_indicator}")
            else:
                print(f"Structure Quality: {structure_indicator} | Shape Quality: {shape_indicator}")
        
        print(f"Q-Values: mean={mean_q_value:.3f}")
        print("=" * 70)

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
            
        # Anti-local-minima mechanisms
        if episodes_since_improvement > 20:  # No improvement for 20 episodes
            print("⚠️  No improvement for 20 episodes - applying anti-local-minima measures:")
            
            # 1. Epsilon boost to increase exploration
            if eps < 0.3:
                eps = min(0.5, eps + 0.2)
                print(f"   📈 Boosting epsilon to {eps:.3f} for exploration")
            
            # 2. Reset target network to encourage new learning
            if episodes_since_improvement > 30:
                print("   🔄 Resetting target network to break learning stagnation")
                tgt.load_state_dict(model.state_dict())
                episodes_since_improvement = 0  # Reset counter after intervention

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
