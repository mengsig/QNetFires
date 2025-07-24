import gym
import numpy as np
from gym import spaces
from Simulate import Simulate


class FuelBreakEnv(gym.Env):
    """
    Multi-step env. Agent places fuel breaks iteratively.
    Observation: 8 channels (firelines N/E/S/W, slp, asp, fbfm, cc) with 0 where breaks placed.
    Action: pick ONE cell per step (Discrete) or many (MultiBinary) – shown here: one per step.
    Episode ends when break_budget is reached.
    Reward shown here only at the end (after budget used) = -normalized acres burned.
    """

    metadata = {"render.modes": []}

    def __init__(
        self, raster, break_budget, break_step=None, num_simulations=10, seed=None
    ):
        super().__init__()

        # store arrays for the simulator (NEVER zero these)
        self.raw_args = (
            raster["slp"],
            raster["asp"],
            raster["dem"],
            raster["cc"],
            raster["cbd"],
            raster["cbh"],
            raster["ch"],
            raster["fbfm"],
        )

        H, W = self.raw_args[0].shape
        self.H, self.W = H, W
        self.break_budget = break_budget
        if break_step is None:
            self.break_step = break_budget // 5
        else:
            self.break_step = break_step
        self.num_simulations = num_simulations

        zero = np.zeros((H, W), dtype=np.float32)
        self.firelines = (
            raster.get("fireline_north", zero),
            raster.get("fireline_east", zero),
            raster.get("fireline_south", zero),
            raster.get("fireline_west", zero),
        )

        # ===== Build the BASE tensor once (what the agent "could" see before masking) =====
        # Order: [slp, asp, cc, fbfm, fn, fe, fs, fw]   (8, H, W)
        slp, asp, dem, cc, cbd, cbh, ch, fbfm = self.raw_args
        fn, fe, fs, fw = self.firelines
        self._base = np.stack([slp, asp, cc, fbfm, fn, fe, fs, fw], axis=0).astype(
            np.float32
        )

        # ===== Spaces =====
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self._base.shape, dtype=np.float32
        )
        # Change to MultiBinary if you want many cells per step:
        # self.action_space = spaces.MultiBinary(self.H * self.W)
        self.action_space = spaces.Discrete(self.H * self.W)

        # bookkeeping
        self._break_mask = None  # (H, W) bool
        self._steps_used = 0

        self._build_sim()

    # ---------- helpers ----------
    def _build_sim(self):
        self.sim = Simulate(*self.raw_args)
        self.sim.set_space_time_cubes()

    def _make_obs(self):
        # zero out at break locations for the agent view
        obs = self._base.copy()
        if self._break_mask is not None:
            obs[:, self._break_mask] = 0.0
            obs[0, self._break_mask] = 91

        # clean nans/inf if any slipped through
        self.obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return self.obs

    # ---------- gym api ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_sim()
        self._break_mask = np.zeros((self.H, self.W), dtype=bool)
        self._used = 0
        self._last_burned = None  # for incremental reward
        return self._make_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int8).reshape(-1)
        assert action.size == self.H * self.W

        # only allow k new placements this step
        new_cells = np.flatnonzero(action)
        # drop already-broken cells
        new_cells = new_cells[~self._break_mask.flat[new_cells]]

        if new_cells.size > self.break_step:
            # you can either clip, sample first k, or give negative reward. Here: clip.
            new_cells = new_cells[: self.break_step]

        # apply them
        self._break_mask.flat[new_cells] = True
        self._used += new_cells.size

        # simulate to get incremental reward
        self.sim.set_fuel_breaks(self._break_mask)
        self.sim.average_acres_burned = 0
        self.sim.run_many_simulations(self.num_simulations)
        burned = self.sim.average_acres_burned

        if self._last_burned is None:
            # first step, no prior baseline -> 0 shaping or full negative
            incremental = burned
        else:
            incremental = burned - self._last_burned

        self._last_burned = burned

        # reward: improvement (negative burned). Example shaping:
        reward = -incremental / float(self.H * self.W)

        done = self._used >= self.break_budget
        obs = self._make_obs()

        return obs, reward, done, False, {"burned": burned, "new_cells": new_cells.size}

    def render(self, save_string=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow(self.sim.burned_mean, cmap="hot", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.1)
        cbar.set_label("Burned fraction")
        fig.savefig(f"{save_string or 'test'}.png", dpi=300)

    def render_observables(self, save_string=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(self.sim.burned_mean, cmap="hot", vmin=0, vmax=1)
        ax[1][0].imshow(self.obs[1], cmap="Greens", vmin=0, vmax=1)
        ax[0][1].imshow(self.obs[4], cmap="hot")
        ax[1][1].imshow(self.obs[2])
        fig.savefig(f"{save_string or 'test'}.png", dpi=300)
