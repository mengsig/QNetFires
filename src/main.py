import os
import sys
import torch
import numpy as np
from Env import FuelBreakEnv
from Model import QNet

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import load_all_rasters

# ----- config -----
BREAK_BUDGET = 100
NUM_SIMS = 100
WEIGHTS_PATH = "checkpoints/qnet.pt"  # put your file here (or skip loading)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def choose_action_greedy(model, obs_np):
    """For Discrete: pick argmax Q over flattened grid."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
        q = model(obs_t)  # assume (1, H*W)
        action = int(torch.argmax(q, dim=1).item())
    return action


def choose_action_topk(model, obs_np, k):
    with torch.no_grad():
        q = model(torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE))  # (1, H*W)
        vals, idxs = torch.topk(q.squeeze(0), k)
        action = np.zeros(q.numel(), dtype=np.int8)
        action[idxs.cpu().numpy()] = 1
    return action  # shape (H*W,), 0/1


def main():
    raster_dict = load_all_rasters("cropped_raster", 1)

    env = FuelBreakEnv(raster_dict, break_budget=BREAK_BUDGET, num_simulations=NUM_SIMS)

    # load model
    H, W = env.H, env.W
    model = QNet(H, W).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()

    # ---- run one greedy episode ----
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = choose_action_topk(model, obs, k=env.break_step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        env.render(f"temp_{steps}")
        env.render_observables(f"temp_{steps}_obs")
        print(f"step {steps}: action={action}, reward={reward:.6f}")

    print(f"Episode finished in {steps} steps. Total reward: {total_reward:.6f}")
    if "burned" in info:
        print("Burned acres:", info["burned"])

    env.render("prediction_run")


if __name__ == "__main__":
    main()
