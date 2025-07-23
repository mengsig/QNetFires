#!/usr/bin/env python3
# train_dqn_vec.py
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from gym.vector import AsyncVectorEnv  # or SyncVectorEnv

from Env import FuelBreakEnv
from Model import QNet

# ---------- project path ----------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import load_all_rasters  # noqa: E402

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
    action_t = torch.from_numpy(np.stack(batch.action)).long().to(DEVICE)  # 0/1
    reward_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(DEVICE)
    done_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(DEVICE)

    B, _, H, W = obs_t.shape
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


# ---------- Vector helpers ----------
def make_env(rasters, budget, kstep, sims):
    def _thunk():
        return FuelBreakEnv(
            rasters,
            break_budget=budget,
            break_step=kstep,
            num_simulations=sims,
        )

    return _thunk


def choose_actions_batch(model, obs_np, k, eps, invalid_masks=None):
    """
    obs_np: (N,C,H,W)
    invalid_masks: (N,H,W) bool or None
    Returns actions: (N,H*W) int8
    """
    N, C, H, W = obs_np.shape
    HxW = H * W
    actions = np.zeros((N, HxW), dtype=np.int8)

    with torch.no_grad():
        q_all = model(torch.from_numpy(obs_np).to(DEVICE)).cpu().numpy()  # (N,H*W)

    for i in range(N):
        if random.random() < eps:
            cand = np.arange(HxW)
            if invalid_masks is not None:
                cand = cand[~invalid_masks[i].flatten()]
            idx = np.random.choice(cand, size=min(k, cand.size), replace=False)
        else:
            qi = q_all[i]
            if invalid_masks is not None:
                qi = qi.copy()
                qi[invalid_masks[i].flatten()] = -1e9
            idx = np.argpartition(qi, -k)[-k:]
        actions[i, idx] = 1

    return actions


def extract_invalid_masks(infos, key="invalid_mask"):
    if not infos or key not in infos[0]:
        return None
    return np.stack([info[key] for info in infos], axis=0)


def squash_rewards(rewards, n_envs, reduce="sum"):
    """Ensure rewards -> (n_envs,)"""
    rewards = np.asarray(rewards)
    if rewards.shape == (n_envs,):
        return rewards
    # try reshape
    if rewards.size % n_envs != 0:
        raise ValueError(
            f"Rewards shape {rewards.shape} cannot be reshaped to (n_envs, -1)"
        )
    rewards = rewards.reshape(n_envs, -1)
    if reduce == "sum":
        return rewards.sum(axis=1)
    elif reduce == "mean":
        return rewards.mean(axis=1)
    elif reduce == "max":
        return rewards.max(axis=1)
    else:
        raise ValueError(f"Unknown reduce '{reduce}'")


# ---------- main ----------
def main():
    # Hyperparams
    EPISODES = 500
    STEPS_PER_EP = 50  # steps per meta-episode
    BUFFER_CAP = 50_000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 1e-4
    START_EPS = 1.0
    END_EPS = 0.05
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 1000
    SAVE_EVERY = 10

    # Vector env params
    N_ENVS = 16
    BUDGET = 100
    K_STEPS = 5
    SIMS = 25

    # Build envs
    raster_dict = load_all_rasters("cropped_raster", 1)
    raster_sets = [raster_dict] * N_ENVS  # replace with your list of different rasters
    env_fns = [make_env(r, BUDGET, K_STEPS, SIMS) for i, r in enumerate(raster_sets)]
    vec_env = AsyncVectorEnv(env_fns)

    obs, _ = vec_env.reset()  # (N,C,H,W)
    N, C, H, W = obs.shape

    model = QNet(H, W).to(DEVICE)
    target_model = QNet(H, W).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = ReplayBuffer(BUFFER_CAP)

    global_step = 0
    eps = START_EPS

    ep_returns = np.zeros(N_ENVS, dtype=np.float32)
    ep_lengths = np.zeros(N_ENVS, dtype=np.int32)

    for meta_ep in range(1, EPISODES + 1):
        print(f"META-EP: {meta_ep}/{EPISODES}")
        for step in range(STEPS_PER_EP):
            print(f"{step}/{STEPS_PER_EP}", end="\r", flush=True)

            # invalid masks from previous infos (None on first step)
            invalid_masks = extract_invalid_masks(
                globals().get("infos", None), "invalid_mask"
            )

            actions = choose_actions_batch(model, obs, K_STEPS, eps, invalid_masks)
            next_obs, rewards, dones, _, infos = vec_env.step(actions)

            # --- FIX REWARDS SHAPE HERE ---
            rewards = squash_rewards(rewards, N_ENVS, reduce="sum")

            # store transitions
            for i in range(N_ENVS):
                buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

            ep_returns += rewards
            ep_lengths += 1

            # reset finished envs
            if np.any(dones):
                try:
                    resets = vec_env.reset_done(dones)  # gym>=0.26
                except AttributeError:
                    idxs = np.where(dones)[0].tolist()
                    resets = vec_env.reset(indices=idxs)
                next_obs[dones] = resets

                finished = np.where(dones)[0]
                for i in finished:
                    print(f"[env {i}] return={ep_returns[i]:.3f} len={ep_lengths[i]}")
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            obs = next_obs
            global_step += N_ENVS

            # eps decay
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac

            # optimize
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_q_loss(model, target_model, batch, GAMMA, K_STEPS)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % TARGET_SYNC_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())

        print(
            f"[MetaEp {meta_ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} buffer={len(buffer)}"
        )

        if meta_ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/qnet_ep{meta_ep}.pt")

    torch.save(model.state_dict(), "checkpoints/qnet_final.pt")
    print("Training finished.")


if __name__ == "__main__":
    main()
