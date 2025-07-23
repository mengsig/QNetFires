# train_dqn.py
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from Env import FuelBreakEnv
from Model import QNet

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import load_all_rasters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Transition = namedtuple("Transition", "obs action reward next_obs done")


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


def choose_action_eps_greedy(model, obs_np, k, eps, invalid_mask=None):
    """Return MultiBinary vector of length H*W with exactly k ones."""
    HxW = obs_np.shape[-2] * obs_np.shape[-1]
    if random.random() < eps:
        # random K among still-valid
        cand = np.arange(HxW)
        if invalid_mask is not None:
            cand = cand[~invalid_mask.flatten()]
        idx = np.random.choice(cand, size=min(k, cand.size), replace=False)
        act = np.zeros(HxW, dtype=np.int8)
        act[idx] = 1
        return act
    with torch.no_grad():
        q = model(torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE))  # (1, H*W)
        q = q.squeeze(0).cpu().numpy()  # (H*W,)
        if invalid_mask is not None:
            q[invalid_mask.flatten()] = -1e9
        idx = np.argpartition(q, -k)[-k:]
        act = np.zeros(HxW, dtype=np.int8)
        act[idx] = 1
        return act


def compute_q_loss(model, target_model, batch, gamma, k):
    # shapes: obs: (B,C,H,W), action: (B,H*W), reward: (B,), next_obs: (B,C,H,W)
    obs_t = torch.from_numpy(np.stack(batch.obs)).float().to(DEVICE)
    next_obs_t = torch.from_numpy(np.stack(batch.next_obs)).float().to(DEVICE)
    action_t = torch.from_numpy(np.stack(batch.action)).long().to(DEVICE)  # 0/1
    reward_t = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(DEVICE)
    done_t = torch.from_numpy(np.array(batch.done, dtype=np.uint8)).to(DEVICE)

    B, _, H, W = obs_t.shape
    HxW = H * W

    q_all = model(obs_t)  # (B,H*W)
    # gather selected k cells’ Qs -> mean to get one scalar per transition
    act_mask = action_t.bool()
    q_selected = torch.sum(q_all * act_mask.float(), dim=1) / k

    with torch.no_grad():
        # Double DQN
        next_q_all = model(next_obs_t)  # online
        best_idx = torch.argmax(next_q_all, dim=1)  # (B,)
        target_q_all = target_model(next_obs_t)  # target
        next_q = target_q_all.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        target = reward_t + gamma * (1 - done_t.float()) * next_q

    loss = nn.MSELoss()(q_selected, target)
    return loss


def main():
    # ----- Hyperparams -----
    EPISODES = 500
    STEPS_PER_EP = 100  # or env.break_budget // env.break_step
    BUFFER_CAP = 50_000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 1e-4
    START_EPS = 1.0
    END_EPS = 0.05
    EPS_DECAY_STEPS = 50_000
    TARGET_SYNC_EVERY = 1000
    SAVE_EVERY = 10
    K_STEPS = None  # set later from env.break_step

    # ----- Env / Model -----
    raster_dict = load_all_rasters("cropped_raster", 1)
    env = FuelBreakEnv(raster_dict, break_budget=100, break_step=5, num_simulations=10)
    K_STEPS = env.break_step

    C, H, W = env.observation_space.shape
    model = QNet(H, W).to(DEVICE)
    target_model = QNet(H, W).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = ReplayBuffer(BUFFER_CAP)

    global_step = 0
    eps = START_EPS

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        for step in range(STEPS_PER_EP):
            # mask out already chosen cells (zeros in obs channels? or env._break_mask)
            invalid_mask = env._break_mask  # bool (H,W)

            action = choose_action_eps_greedy(model, obs, K_STEPS, eps, invalid_mask)
            next_obs, reward, done, _, info = env.step(action)

            buffer.push(obs, action, reward, next_obs, done)
            ep_reward += reward

            obs = next_obs
            global_step += 1

            # epsilon decay
            eps = max(
                END_EPS,
                START_EPS - (START_EPS - END_EPS) * (global_step / EPS_DECAY_STEPS),
            )

            # optimize
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_q_loss(model, target_model, batch, GAMMA, K_STEPS)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # target net sync
                if global_step % TARGET_SYNC_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())

            if done:
                break

        print(f"[Ep {ep}] reward={ep_reward:.4f} eps={eps:.3f} steps={step + 1}")

        if ep % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/qnet_ep{ep}.pt")

    torch.save(model.state_dict(), "checkpoints/qnet_final.pt")
    print("Training finished.")


if __name__ == "__main__":
    main()
