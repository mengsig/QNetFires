import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

from src.scripts.Simulate import Simulate

class FireEnv(gym.Env):
    """
    Gym environment for a single-run fire‐spread simulation.
    
    Action: a binary mask of shape (H, W), indicating fuel-break cells.
    Observation: final 'fire_type' matrix of shape (H, W) (0=unburned, 1=burned, etc.).
    Reward: negative acres burned (so agent learns to minimize spread).
    Episode: one step (reset → step → done).
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 slope: np.ndarray,
                 aspect: np.ndarray,
                 dem: np.ndarray,
                 cc: np.ndarray,
                 cbd: np.ndarray,
                 cbh: np.ndarray,
                 ch: np.ndarray,
                 fuel_model: np.ndarray,
                 ignite_point: tuple = None):
        super().__init__()
        # store the static inputs
        self._static_args = (slope, aspect, dem, cc, cbd, cbh, ch, fuel_model)
        self.H, self.W = slope.shape

        # Action: binary fuel-break mask
        self.action_space = spaces.MultiBinary(self.H * self.W)
        # Observation: final fire_type matrix (integers 0/1/…)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.H, self.W),
            dtype=np.uint8
        )

        # Ignite point default: center if none
        self.ignite_point = ignite_point or (self.H//2, self.W//2)

        # placeholder for last sim
        self.sim = None
        self.last_firemap = None
        self.seed()

    def seed(self, seed=None):
        """Seed RNGs for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset environment and return initial observation (all zeros)."""
        # Re-instantiate the simulator
        self.sim = Simulate(*self._static_args)
        self.sim.set_space_time_cubes()
        # blank observation until the run
        self.last_firemap = np.zeros((self.H, self.W), dtype=np.uint8)
        return self.last_firemap

    def step(self, action):
        """
        action: flat binary array of length H*W. Reshape to (H, W).
        """
        # 1) apply fuel breaks
        mask = action.reshape((self.H, self.W)).astype(bool)
        self.sim.set_fuel_breaks(mask)

        # 2) run simulations with configurable parameters
        x0, y0 = self.ignite_point
        num_sims = getattr(self, 'num_simulations', 100)
        max_duration = getattr(self, 'max_duration', None)
        
        # Use sequential fire simulations (avoid threading overhead)
        self.sim.run_many_simulations(num_sims, max_duration)
        obs = self.sim.get_burned()
        # cast to uint8 for the observation
        # 3) compute reward
        acres = self.sim.get_loss()
        reward = -acres

        # 4) this is a single-step episode
        done = True

        # 5) info dict
        info = {
            "acres_burned": acres,
            "total_cells": self.H * self.W,
            "num_simulations": num_sims,
            "max_duration": max_duration
        }
        self.last_firemap = obs

        return obs, reward, done, info

    def render(self, mode="human"):
        """Show the last fire map."""
        if mode == "human":
            plt.imshow(self.last_firemap, cmap="hot", origin="lower")
            plt.title("Fire Spread (1=burned)")
            plt.colorbar(label="fire_type")
            plt.show()
        elif mode == "rgb_array":
            # return an RGB array
            cmap = plt.get_cmap("hot")
            rgba = cmap(self.last_firemap / self.last_firemap.max())
            return (rgba[:, :, :3] * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        plt.close("all")

