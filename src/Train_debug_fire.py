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

# Utility function to safely convert values to scalars
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
        # It's an array-like object with multiple elements
        try:
            return float(np.mean(value))
        except:
            return fallback
    try:
        return float(value)
    except:
        return fallback

mp.set_start_method("spawn", force=True)

# -------- project path hack --------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import RasterManager  # noqa: E402

# DEBUG PARAMETERS FOR FIRE SIMULATION ISSUES
DEBUG_FIRE_SIM = True
MAX_BURNED_THRESHOLD = 500  # Values above this suggest fire sim failure
MIN_STEPS_FOR_COMPLETION = 20  # Episodes should take at least this many steps


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
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
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
def compute_q_loss(model, target_model, batch, gamma, k, weights=None, device="cpu"):
    """Enhanced Q-learning loss with optional importance sampling weights."""
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
        self._last_burned = None  # Track burned area
        self._env_id = id(self) % 1000  # Unique environment ID for debugging
        self._is_dummy = hasattr(env, 'dummy') or 'Dummy' in str(type(env))
        self._step_count = 0
        self._burned_history = []  # Track burned area history for debugging

    def reset(self, **kw):
        self._ret = 0
        self._len = 0
        self._last_burned = None
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
            
            # Handle both scalar and array rewards
            self._ret += safe_scalar(r)
            self._len += 1
            self._step_count += 1
            
            # Always ensure info dict exists and has burned area
            if info is None:
                info = {}
            info = dict(info)
            
            # Track burned area for reporting - handle arrays
            if "burned" in info:
                burned_val = safe_scalar(info["burned"], 100.0)
                self._last_burned = burned_val
                info["burned"] = burned_val
                
                # Track burned area history for debugging
                self._burned_history.append(burned_val)
                if len(self._burned_history) > 10:  # Keep last 10 values
                    self._burned_history.pop(0)
                    
            elif self._last_burned is not None:
                info["burned"] = self._last_burned
            else:
                info["burned"] = 100.0  # Default fallback
                
            # Add environment metadata for debugging
            info["env_id"] = self._env_id
            info["is_dummy"] = self._is_dummy
            info["step_count"] = self._step_count
            
            if d or tr:
                info["episode_return"] = self._ret
                info["episode_length"] = self._len
                burned_scalar = safe_scalar(info.get('burned', 100))
                burned_str = f"{burned_scalar:.1f}"
                if self._len > 1:  # Only print if episode had multiple steps
                    env_type = "DUMMY" if self._is_dummy else "REAL"
                    burned_range = f"[{min(self._burned_history):.1f}-{max(self._burned_history):.1f}]" if len(self._burned_history) > 1 else ""
                    print(f"Episode completed: Return={self._ret:.3f}, Length={self._len}, Burned={burned_str} {burned_range}, Type={env_type}, ID={self._env_id}")
                try:
                    obs, _ = self.env.reset()
                except Exception as e:
                    if self._error_count <= 3:
                        print(f"Environment reset failed: {e}")
                    self._error_count += 1
                    obs = self._get_dummy_obs()
                self._ret = 0
                self._len = 0
                self._last_burned = None
            
            return obs, r, d, tr, info
            
        except (KeyboardInterrupt, SystemExit):
            # Don't catch these - let them propagate
            raise
            
        except Exception as e:
            # Only catch truly unexpected errors, not simulation failures
            self._error_count += 1
            
            # Reduce spam - only print first few errors
            if self._error_count <= 3:
                print(f"Environment wrapper error #{self._error_count}: {type(e).__name__}: {e}")
            elif self._error_count == 4:
                print(f"Environment wrapper: Suppressing further error messages (total: {self._error_count})")
            
            # If we have too many errors, this environment is broken
            if self._error_count > self._max_errors:
                obs = self._get_dummy_obs()
                reward = -1.0
                done = True
                
                info = {
                    "episode_return": safe_scalar(self._ret), 
                    "episode_length": self._len, 
                    "burned": 200.0,  # High but not extreme
                    "error": True
                }
                self._ret = 0
                self._len = 0
                self._last_burned = None
                return obs, reward, done, False, info
            
            # For recoverable errors, try to continue with current state
            obs = self._get_dummy_obs()
            reward = -0.1  # Small penalty, not catastrophic
            done = False   # Don't end episode unless necessary
            
            info = {
                "burned": safe_scalar(self._last_burned, 150.0),
                "error": True
            }
            
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
            if isinstance(raster, dict) and 'slp' in raster:
                self.H, self.W = raster['slp'].shape
        except:
            pass
            
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8, self.H, self.W), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiBinary(self.H * self.W)
        
        self.steps_taken = 0
        
    def reset(self, **kwargs):
        self.steps_taken = 0
        obs = np.random.rand(8, self.H, self.W).astype(np.float32) * 0.1  # Small random values
        return obs, {}
        
    def step(self, action):
        self.steps_taken += 1
        
        # Simulate some fuel break placement
        action = np.asarray(action, dtype=np.int8).reshape(-1)
        placed = min(np.sum(action), self.kstep)  # Respect step limit
        
        # More realistic reward: cost of fuel breaks vs fire prevention benefit
        fuel_break_cost = 0.003 * placed
        fire_prevention_benefit = 0.01 * min(placed, 3)  # Diminishing returns
        reward = fire_prevention_benefit - fuel_break_cost
        
        # Episode ends when budget is reached or after reasonable number of steps
        max_steps = min(self.budget // self.kstep, 25)
        done = self.steps_taken >= max_steps
        
        # Create more realistic dummy observation with some structure
        obs = np.zeros((8, self.H, self.W), dtype=np.float32)
        
        # Add some realistic-looking data with small variations
        obs[0] = np.random.uniform(0.2, 0.8, (self.H, self.W))  # Slope
        obs[1] = np.random.uniform(0, 1, (self.H, self.W))      # Aspect (normalized)
        obs[2] = np.random.uniform(0.1, 0.7, (self.H, self.W))  # Canopy cover
        obs[3] = np.random.randint(1, 14, (self.H, self.W)) / 13.0  # Fuel model (normalized)
        
        # Fireline layers (mostly zeros with some existing breaks)
        for i in range(4, 8):
            obs[i] = np.random.choice([0.0, 1.0], (self.H, self.W), p=[0.97, 0.03])
        
        # More realistic burned area simulation with environment variation
        # Use environment-specific seed for consistency but variety across environments
        env_seed = hash(str(id(self))) % 10000  # Unique per environment instance
        np.random.seed(env_seed + self.steps_taken)  # Vary by step but keep consistent
        
        # Base burned area with environment variation
        base_burned = 170.0 + np.random.uniform(-25, 25)  # 145-195 base per environment
        reduction_per_break = 5.0 + np.random.uniform(-1, 1)  # Slightly varied effectiveness
        total_breaks_placed = self.steps_taken * placed  # Use actual placed breaks
        
        # Add some realistic noise
        noise = np.random.uniform(-8, 8)
        burned_area = max(75.0, min(220.0, base_burned - total_breaks_placed * reduction_per_break + noise))
        
        info = {
            "burned": burned_area,
            "new_cells": placed,
            "dummy": True
        }
        
        if done:
            episode_return = reward * self.steps_taken
            info["episode_return"] = episode_return
            info["episode_length"] = self.steps_taken
        
        return obs, reward, done, False, info


def make_env_with_raster(raster, budget, kstep, sims, seed):
    """Create environment with specific raster data - LAZY CREATION (no double creation)."""
    def thunk():
        # Set random seed for reproducibility in subprocess
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Validate raster data first
                if not isinstance(raster, dict):
                    raise ValueError("Raster must be a dictionary")
                
                required_keys = ['slp', 'asp', 'fbfm', 'fireline_north', 'fireline_east', 'fireline_south', 'fireline_west']
                for key in required_keys:
                    if key not in raster:
                        raise ValueError(f"Missing required raster key: {key}")
                    if not isinstance(raster[key], np.ndarray):
                        raise ValueError(f"Raster key {key} must be numpy array")
                    # Validate array properties
                    if raster[key].size == 0:
                        raise ValueError(f"Raster key {key} is empty")
                    if not np.isfinite(raster[key]).all():
                        print(f"Warning: Raster key {key} contains non-finite values, cleaning...")
                        raster[key] = np.nan_to_num(raster[key], nan=0.0, posinf=1.0, neginf=0.0)
                
                # Start with minimal simulations and increase if successful
                if attempt == 0:
                    effective_sims = 1  # Very conservative first attempt
                elif attempt == 1:
                    effective_sims = min(2, sims)  # Slightly more
                else:
                    effective_sims = min(3, sims)  # Full attempt
                
                # Only print when actually creating (not during env_fn setup)
                print(f"🔧 Creating environment (attempt {attempt + 1}/{max_retries}, sims={effective_sims})")
                
                env = FuelBreakEnv(
                    raster,
                    break_budget=budget,
                    break_step=kstep,
                    num_simulations=effective_sims,
                    seed=seed,
                )
                
                # MINIMAL testing - only check if environment can be created
                try:
                    obs, _ = env.reset()
                    if obs is None or obs.size == 0:
                        raise RuntimeError("Invalid environment")
                    # Skip step testing for speed - any issues will be caught during training
                except Exception as test_e:
                    raise RuntimeError(f"Environment validation failed: {test_e}")
                
                return RobustAutoResetWrapper(env)
                
            except Exception as e:
                print(f"❌ Environment creation attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying with different parameters...")
                    import time
                    time.sleep(0.1)  # Brief pause between retries
                else:
                    print(f"💥 All {max_retries} attempts failed!")
                    raise RuntimeError(f"Failed to create environment after {max_retries} attempts. Last error: {e}")
    
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
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# ---------- main ----------
def main():
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {DEVICE}...")
    
    # CUDA memory optimization
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Hyperparams
    EPISODES = 1000  # Increased episodes to utilize all rasters
    STEPS_PER_EP = 30  # Increased to allow episodes to complete naturally  # Increased steps per episode
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
    GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate larger batch size
    
    # Vectorized env parameters
    N_ENVS = 16  # Reduced from 32 to save memory
    BUDGET = 200
    K_STEPS = 10
    SIMS = 1  # Reduced to 1 for maximum stability   # Reduced for stability (fire simulation often fails with higher values)
    
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
        print("Creating dummy rasters for testing...")
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
    
    # Create environments with selected rasters - with robust error handling
    print(f"Creating {N_ENVS} environments...")
    env_fns = []
    successful_envs = 0
    
    for i, raster in enumerate(selected_rasters):
        try:
            env_fn = make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i)
            # Test the environment creation immediately
            test_env = env_fn()
            test_obs, _ = test_env.reset()
            print(f"Environment {i}: ✅ Created successfully")
            env_fns.append(env_fn)
            successful_envs += 1
        except Exception as e:
            print(f"Environment {i}: ❌ Failed to create: {type(e).__name__}: {e}")
            # If we can't create any environments, this is a critical error
            if successful_envs == 0 and i > N_ENVS // 2:
                raise RuntimeError(f"Failed to create more than half of the environments. "
                                 f"This indicates a serious issue with the environment setup. "
                                 f"Check pyretechnics installation and raster data.")
    
    if successful_envs == 0:
        raise RuntimeError("Failed to create any environments! Check your setup.")
    
    if successful_envs < N_ENVS:
        print(f"⚠️  Only created {successful_envs}/{N_ENVS} environments successfully")
        print(f"Adjusting N_ENVS to {successful_envs}")
        N_ENVS = successful_envs
    
    print(f"Creating AsyncVectorEnv with {N_ENVS} environments...")
    vec_env = AsyncVectorEnv(env_fns)

    print("Resetting all environments...")
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    actual_n_envs = obs.shape[0]
    _, C, H, W = obs.shape
    
    print(f"Successfully initialized {actual_n_envs} environments with observation shape: {obs.shape}")
    N_ENVS = actual_n_envs  # Update to actual number

    # Initialize model with memory considerations
    if USE_DUELING:
        model = DuelingQNet(H, W).to(DEVICE)
        tgt = DuelingQNet(H, W).to(DEVICE)
        print("Using Dueling DQN architecture")
    elif USE_ENHANCED_MODEL:
        if MEMORY_EFFICIENT:
            # Use lighter version of enhanced model for memory efficiency
            model = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
            tgt = EnhancedQNet(H, W, use_attention=False, use_residual=True, use_multiscale=False).to(DEVICE)
            print("Using Memory-Efficient Enhanced DQN architecture (residual only)")
        else:
            model = EnhancedQNet(H, W, use_attention=True, use_residual=True, use_multiscale=True).to(DEVICE)
            tgt = EnhancedQNet(H, W, use_attention=True, use_residual=True, use_multiscale=True).to(DEVICE)
            print("Using Full Enhanced DQN architecture with attention and residual connections")
    else:
        model = QNet(H, W).to(DEVICE)
        tgt = QNet(H, W).to(DEVICE)
        print("Using basic DQN architecture")
    
    tgt.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning rate scheduler
    if USE_LR_SCHEDULER:
        total_steps = EPISODES * STEPS_PER_EP * 10  # Approximate total training steps
        scheduler = CosineAnnealingWarmup(opt, warmup_steps=1000, total_steps=total_steps)
    
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
    reward_win = deque(maxlen=100)  # Episode returns
    step_reward_win = deque(maxlen=1000)  # Step-by-step rewards
    burned_area_win = deque(maxlen=1000)  # Burned areas
    
    # Training metrics
    best_avg_reward = float('-inf')
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
                make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS) 
                for i, raster in enumerate(selected_rasters)
            ]
            vec_env = AsyncVectorEnv(env_fns)
            reset_out = vec_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        episode_rewards = []
        
        for step in range(STEPS_PER_EP):
            acts = choose_actions_batch(model, obs, K_STEPS, eps, DEVICE)
            
            # Multi-layer protection against hanging environments
            step_success = False
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    # Start the async step
                    vec_env.step_async(acts)
                    
                    # Use threading-based timeout (more reliable than signal)
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def step_wait_thread():
                        try:
                            result = vec_env.step_wait()
                            result_queue.put(result)
                        except Exception as e:
                            exception_queue.put(e)
                    
                    # Start the step_wait in a separate thread
                    step_thread = threading.Thread(target=step_wait_thread)
                    step_thread.daemon = True
                    step_thread.start()
                    
                    # Wait for result with timeout (reduced to 10 seconds)
                    step_thread.join(timeout=10.0)
                    
                    if step_thread.is_alive():
                        # Thread is still running - environment hung
                        print(f"Environment step timed out (retry {retry + 1}/{max_retries})")
                        raise TimeoutError("Environment step timed out after 10 seconds")
                    
                    # Check for exceptions
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    
                    # Get the result
                    if not result_queue.empty():
                        out = result_queue.get()
                        step_success = True
                        break
                    else:
                        raise RuntimeError("No result from environment step")
                
                except (EOFError, BrokenPipeError, ConnectionResetError, TimeoutError, RuntimeError) as e:
                    print(f"🚨 Environment error (retry {retry + 1}/{max_retries}): {type(e).__name__}: {e}")
                    
                    # For EOFError and pipe issues, immediately recreate environments
                    if isinstance(e, (EOFError, BrokenPipeError, ConnectionResetError)):
                        print("🔄 Pipe corruption detected - immediately recreating all environments")
                        break  # Skip retries, go straight to recreation
                    
                    if retry == max_retries - 1:
                        # Final retry failed - recreate environments
                        print("⚠️  All retries failed. Recreating environments...")
                        break
                    
                    # Quick retry with brief pause for non-pipe errors
                    import time
                    time.sleep(1)
                    continue
            
            # If step failed, recreate environments
            if not step_success:
                # Force close the hanging environment
                try:
                    vec_env.close()
                except:
                    pass
                
                # Aggressive process cleanup
                import multiprocessing
                
                # Try to use psutil for better process management if available
                try:
                    import psutil
                    current_process = psutil.Process(os.getpid())
                    for child in current_process.children(recursive=True):
                        try:
                            child.terminate()
                            child.wait(timeout=2)
                        except:
                            try:
                                child.kill()
                            except:
                                pass
                except ImportError:
                    # Fallback without psutil
                    print("psutil not available, using basic process cleanup")
                
                # Clean up multiprocessing
                for p in multiprocessing.active_children():
                    try:
                        p.terminate()
                        p.join(timeout=1)
                        if p.is_alive():
                            p.kill()
                    except:
                        pass
                
                # Wait for cleanup
                import time
                time.sleep(2)
                
                # Track AsyncVectorEnv failures
                if not hasattr(main, '_async_failures'):
                    main._async_failures = 0
                main._async_failures += 1
                
                print(f"🔄 Recreating environments (failure #{main._async_failures})")
                
                # If AsyncVectorEnv keeps failing, fall back to synchronous
                if main._async_failures >= 3:
                    print("⚠️  AsyncVectorEnv failing repeatedly. Consider using train_sync.py for stability.")
                    print("🔄 Attempting one more AsyncVectorEnv recreation with reduced complexity...")
                    
                    # Reduce complexity for stability
                    effective_n_envs = max(4, N_ENVS // 2)  # Reduce environment count
                    effective_sims = 1  # Minimal simulations
                    print(f"🔧 Reducing to {effective_n_envs} environments with {effective_sims} simulations")
                else:
                    effective_n_envs = N_ENVS
                    effective_sims = SIMS
                
                # Recreate environments with fresh rasters
                try:
                    selected_rasters = raster_manager.get_random_rasters(effective_n_envs)
                    env_fns = [
                        make_env_with_raster(raster, BUDGET, K_STEPS, effective_sims, seed=i + ep * N_ENVS + step + main._async_failures) 
                        for i, raster in enumerate(selected_rasters)
                    ]
                    vec_env = AsyncVectorEnv(env_fns)
                    reset_out = vec_env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    
                    # Update N_ENVS if we reduced it
                    if effective_n_envs != N_ENVS:
                        N_ENVS = effective_n_envs
                        print(f"✅ Successfully recreated {N_ENVS} environments")
                    else:
                        print(f"✅ Successfully recreated all {N_ENVS} environments")
                    
                except Exception as recreation_error:
                    print(f"❌ Environment recreation failed: {recreation_error}")
                    print("💡 Recommendation: Stop training and use train_sync.py instead")
                    print("🔄 Attempting to continue with minimal setup...")
                    
                    # Last resort: create minimal environment setup
                    try:
                        selected_rasters = raster_manager.get_random_rasters(2)  # Minimal 2 environments
                        env_fns = [
                            make_env_with_raster(raster, BUDGET, K_STEPS, 1, seed=i + ep * 1000) 
                            for i, raster in enumerate(selected_rasters)
                        ]
                        vec_env = AsyncVectorEnv(env_fns)
                        reset_out = vec_env.reset()
                        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                        N_ENVS = 2
                        print(f"⚠️  Running with minimal setup: {N_ENVS} environments")
                    except Exception as final_error:
                        print(f"💥 Critical failure: {final_error}")
                        print("🛑 Training cannot continue. Please use train_sync.py")
                        raise
                
                # Skip this step and continue
                continue
            
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
            # Enhanced episode completion tracking
            completed_episodes = 0
            for i in range(N_ENVS):
                if dones[i]:
                    completed_episodes += 1
                    
                    # Force episode return calculation if missing
                    info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                    if info_i and "episode_return" not in info_i:
                        # Calculate episode return from step reward
                        episode_return = safe_scalar(rews[i])
                        info_i["episode_return"] = episode_return
                        info_i["episode_length"] = step + 1
                        print(f"[env {i}] 🔧 Added missing episode_return: {episode_return:.3f}")


            # Store transitions
            for i in range(N_ENVS):
                buf.push(obs[i], acts[i], rews[i], nxt[i], dones[i])
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                
                # Always track step rewards (not just episode returns)
                step_reward_win.append(safe_scalar(rews[i]))
                
                # Track burned area if available
                if info_i and "burned" in info_i:
                    burned_area_win.append(safe_scalar(info_i["burned"]))
                
                
                # IMPROVED: Track episode completion with forced episode returns
                if dones[i]:
                    info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                    
                    # Get episode return - create if missing
                    if info_i and "episode_return" in info_i:
                        episode_reward = safe_scalar(info_i['episode_return'])
                    else:
                        # Calculate cumulative episode return from step reward
                        episode_reward = safe_scalar(rews[i])
                        if info_i:
                            info_i["episode_return"] = episode_reward
                            info_i["episode_length"] = step + 1
                    
                    episode_rewards.append(episode_reward)
                    reward_win.append(episode_reward)
                    
                    # Enhanced logging with fire simulation diagnostics
                    burned_val = info_i.get('burned', None) if info_i else None
                    burned_scalar = safe_scalar(burned_val, fallback=None)
                    burned_str = f"{burned_scalar:.1f}" if burned_scalar is not None else 'N/A'
                    
                    # Check if this looks like a fire simulation failure
                    fire_sim_failed = burned_scalar is not None and burned_scalar > 600
                    status = "🔥 FIRE_SIM_FAILED" if fire_sim_failed else "✅ COMPLETED"
                    
                    # Enhanced diagnostics
                    is_dummy_val = info_i.get('is_dummy', False) if info_i else False
                    env_type = "DUMMY" if safe_scalar(is_dummy_val, fallback=False) else "REAL"
                    env_id = safe_scalar(info_i.get('env_id', i) if info_i else i, fallback=i)
                    
                    print(f"[env {i}] {status} Episode: R={episode_reward:.3f} "
                          f"Burned={burned_str} Type={env_type} ID={env_id} Step={step+1}"))
                try:
                    vec_env.close()
                except:
                    pass
                
                # Brief cleanup
                import time
                time.sleep(1)
                
                # Recreate with fresh rasters
                selected_rasters = raster_manager.get_random_rasters(N_ENVS)
                env_fns = [
                    make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS + 9999) 
                    for i, raster in enumerate(selected_rasters)
                ]
                vec_env = AsyncVectorEnv(env_fns)
                reset_out = vec_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                print(f"✅ Proactively recreated {N_ENVS} environments")
        
        
        # Episode completion summary
        if episode_rewards:
            print(f"📊 Episode Summary: {len(episode_rewards)}/{N_ENVS} environments completed naturally")
        else:
            print(f"⚠️  Episode Summary: 0/{N_ENVS} environments completed - all timed out!")
            print("💡 Suggestion: Increase STEPS_PER_EP or check fire simulation")
        # Episode statistics with multiple metrics
        mean_loss = float(np.mean(loss_win)) if loss_win else float("nan")
        
        
        # IMPROVED: Episode returns with fallback handling
        if reward_win:
            mean_episode_reward = float(np.mean(reward_win))
            print(f"📈 Episode rewards available: {len(reward_win)} episodes, mean={mean_episode_reward:.3f}")
        else:
            # Fallback: Use step rewards if no episodes completed
            if step_reward_win:
                mean_episode_reward = float(np.mean(step_reward_win))
                print(f"⚠️  No episode completions - using step reward as fallback: {mean_episode_reward:.3f}")
            else:
                mean_episode_reward = float("nan")
                print(f"❌ No rewards available at all!")
        
        # Step rewards (always available)
        mean_step_reward = float(np.mean(step_reward_win)) if step_reward_win else float("nan")
        
        # Burned area (fire spread metric)
        mean_burned_area = float(np.mean(burned_area_win)) if burned_area_win else float("nan")
        
        # Calculate burned area statistics for debugging
        if burned_area_win:
            recent_burned = list(burned_area_win)[-50:]  # Last 50 values
            min_burned = min(recent_burned)
            max_burned = max(recent_burned)
            std_burned = np.std(recent_burned)
            
            # Check for suspicious patterns
            if max_burned > 500:  # Very high burned areas suggest fallback values
                high_count = sum(1 for b in recent_burned if b > 400)
                if high_count > len(recent_burned) * 0.3:  # More than 30% are high
                    print(f"⚠️  Warning: {high_count}/{len(recent_burned)} recent burned areas > 400 (possible fallback values)")
            
            # Check for identical values (suggests dummy environments)
            from collections import Counter
            value_counts = Counter([round(b, 1) for b in recent_burned])
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            if most_common_count > len(recent_burned) * 0.5:  # More than 50% identical
                print(f"⚠️  Warning: {most_common_count}/{len(recent_burned)} burned areas are identical ({most_common_value}) - possible dummy environments")
        else:
            min_burned = max_burned = std_burned = float("nan")
        
        # Use step reward as primary metric if episode rewards not available
        primary_reward = mean_episode_reward if not np.isnan(mean_episode_reward) else mean_step_reward
        
        current_lr = opt.param_groups[0]['lr'] if USE_LR_SCHEDULER else LR
        
        print(f"[MetaEp {ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} "
              f"loss={mean_loss:.4f} ep_reward={mean_episode_reward:.3f} "
              f"step_reward={mean_step_reward:.4f} burned_area={mean_burned_area:.1f} "
              f"[{min_burned:.0f}-{max_burned:.0f}±{std_burned:.0f}] lr={current_lr:.2e}")
        
        # Track best performance using primary reward metric
        if not np.isnan(primary_reward) and primary_reward > best_avg_reward:
            best_avg_reward = primary_reward
            episodes_since_improvement = 0
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/qnet_best.pt")
            print(f"🎉 New best model saved! Reward: {primary_reward:.4f}")
        else:
            episodes_since_improvement += 1

        # Regular checkpointing
        if ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_reward': best_avg_reward,
                'loss': mean_loss,
            }, f"checkpoints/qnet_ep{ep}.pt")
            
            # Clear CUDA cache periodically
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_reward': best_avg_reward,
    }, "checkpoints/qnet_final.pt")
    
    vec_env.close()
    print(f"Training completed! Best average reward: {best_avg_reward:.3f}")


if __name__ == "__main__":
    main()
