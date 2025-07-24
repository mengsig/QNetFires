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

from Env import FuelBreakEnv
from Model import QNet

mp.set_start_method("spawn", force=True)

# -------- project path hack --------
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

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


# ---------- Loss ----------
def compute_q_loss(model, target_model, batch, gamma, k):
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(DEVICE)
    next_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(DEVICE)
    a_t = torch.from_numpy(np.stack(batch.action)).long().to(DEVICE)
    r_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(DEVICE)
    d_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(DEVICE)

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


def make_env(raster_root, budget, kstep, sims, seed):
    def thunk():
        # load rasters in subprocess, not in parent
        from src.utils.loadingUtils import load_all_rasters

        rasters = load_all_rasters(raster_root, 1)
        env = FuelBreakEnv(
            rasters,
            break_budget=budget,
            break_step=kstep,
            num_simulations=sims,
            seed=seed,
        )
        return AutoResetWrapper(env)

    return thunk


# ---------- Action helper ----------
def choose_actions_batch(model, obs_np, k, eps):
    N, C, H, W = obs_np.shape
    HxW = H * W
    acts = np.zeros((N, HxW), dtype=np.int8)
    with torch.no_grad():
        qs = model(torch.from_numpy(obs_np).to(DEVICE)).cpu().numpy()
    for i in range(N):
        if random.random() < eps:
            idx = np.random.choice(HxW, k, replace=False)
        else:
            qi = qs[i]
            idx = np.argpartition(qi, -k)[-k:]
        acts[i, idx] = 1
    return acts


# ---------- main ----------
def main():
    # Hyperparams
    EPISODES = 500  # Outer training loops ("meta-episodes"); each runs STEPS_PER_EP interaction steps
    STEPS_PER_EP = 1  # Env interaction steps per meta-episode (each step advances ALL N_ENVS envs once)
    BUFFER_CAP = 50_000  # Max number of transitions kept in the replay buffer (old ones are dropped FIFO)
    BATCH_SIZE = (
        32  # How many transitions you sample from the buffer for one gradient update
    )
    GAMMA = 0.99  # Discount factor for future rewards in the Bellman target
    LR = 1e-4  # Learning rate for the optimizer (Adam here)
    START_EPS = 1.0  # Initial epsilon for ε-greedy action selection (full exploration)
    END_EPS = 0.05  # Final/minimum epsilon after decay (some exploration kept)
    EPS_DECAY_STEPS = 50_000  # Number of env-steps (sum over all envs) to anneal epsilon from START_EPS to END_EPS
    TARGET_SYNC_EVERY = (
        100  # How many env-steps between copying online net weights to the target net
    )
    SAVE_EVERY = 10  # Save a model checkpoint every this many meta-episodes

    # Vectorized env parameters
    N_ENVS = (
        32  # Number of environments run in parallel (processes with AsyncVectorEnv)
    )
    BUDGET = 200  # FuelBreakEnv: total number of cells you’re allowed to place across an episode
    K_STEPS = 10  # FuelBreakEnv: how many new fuel-break cells you can place per step (action size)
    SIMS = 25  # FuelBreakEnv: number of Monte-Carlo fire simulations per step to estimate reward

    # Build many rasters; adjust to your paths
    raster_root = "cropped_raster"  # or a list of 16 different roots

    env_fns = [
        make_env(raster_root, BUDGET, K_STEPS, SIMS, seed=i) for i in range(N_ENVS)
    ]
    vec_env = AsyncVectorEnv(env_fns)

    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    N_ENVS = obs.shape[0]
    _, C, H, W = obs.shape

    model = QNet(H, W).to(DEVICE)
    tgt = QNet(H, W).to(DEVICE)
    tgt.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=LR)

    buf = ReplayBuffer(BUFFER_CAP)
    global_step = 0
    eps = START_EPS
    loss_win = deque(maxlen=1000)

    for ep in range(1, EPISODES + 1):
        print(f"META-EP: {ep}/{EPISODES}")
        for _ in range(STEPS_PER_EP):
            acts = choose_actions_batch(model, obs, K_STEPS, eps)
            vec_env.step_async(acts)
            out = vec_env.step_wait()
            # 5-tuple => (obs,rews,dones,truncs,infos)
            if len(out) == 5:
                nxt, rews, dones, truncs, infos = out
                dones = np.logical_or(dones, truncs)
            else:
                nxt, rews, dones, infos = out

            rews = np.asarray(rews, dtype=np.float32)
            if rews.shape != (N_ENVS,):
                rews = rews.reshape(N_ENVS, -1).sum(axis=1)
            dones = np.asarray(dones, dtype=bool)

            for i in range(N_ENVS):
                buf.push(obs[i], acts[i], rews[i], nxt[i], dones[i])
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                if info_i and "episode_return" in info_i:
                    print(
                        f"[env {i}] R={info_i['episode_return']:.3f} L={info_i['episode_length']}"
                    )

            obs = nxt
            global_step += N_ENVS
            frac = min(1.0, global_step / EPS_DECAY_STEPS)
            eps = START_EPS - (START_EPS - END_EPS) * frac

            if len(buf) >= BATCH_SIZE:
                batch = buf.sample(BATCH_SIZE)
                loss = compute_q_loss(model, tgt, batch, GAMMA, K_STEPS)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_win.append(loss.item())
                if global_step % TARGET_SYNC_EVERY == 0:
                    tgt.load_state_dict(model.state_dict())

        mean_loss = float(np.mean(loss_win)) if loss_win else float("nan")
        print(
            f"[MetaEp {ep}] steps={STEPS_PER_EP * N_ENVS} eps={eps:.3f} mean_loss={mean_loss:.4f}"
        )

        if ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/qnet_ep{ep}.pt")

    torch.save(model.state_dict(), "checkpoints/qnet_final.pt")
    print("Done.")


if __name__ == "__main__":
    main()
