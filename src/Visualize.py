#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------- Project imports ----------
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from Env import FuelBreakEnv
from Model import QNet
from src.utils.loadingUtils import load_all_rasters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def choose_action_topk_greedy(model, obs_np, k, invalid_mask=None):
    """Pure greedy K-pick based on Q values."""
    with torch.no_grad():
        q = model(torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE))  # (1, H*W)
        q = q.squeeze(0).cpu().numpy()  # (H*W,)
        if invalid_mask is not None:
            q[invalid_mask.flatten()] = -1e9
        idx = np.argpartition(q, -k)[-k:]
        act = np.zeros(q.size, dtype=np.int8)
        act[idx] = 1
        return act, q


def run_episode(env, model, out_dir=None, make_gif=False):
    obs, _ = env.reset()
    H, W = env.observation_space.shape[-2:]
    k = env.break_step

    # these store what we’ll visualize
    placed_indices = []
    rewards = []
    q_maps = []  # only if you want per-step heatmaps
    frames = []  # for GIF

    done = False
    step = 0

    while not done:
        invalid = env._break_mask  # bool (H,W)
        action_vec, q = choose_action_topk_greedy(model, obs, k, invalid)
        next_obs, reward, done, _, info = env.step(action_vec)
        rewards.append(reward)

        # keep track of which cells we picked at this step
        mask = action_vec.reshape(H, W).astype(bool)
        env.firelines[0][mask] = np.inf
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            placed_indices.append((y, x, step))  # store order

        q_maps.append(q.reshape(H, W))

        if make_gif and out_dir is not None:
            frames.append(render_frame(env, mask, step))  # returns np.array image

        obs = next_obs
        step += 1

    return {
        "placed_indices": placed_indices,
        "rewards": rewards,
        "q_maps": q_maps,
        "frames": frames,
    }


def render_static_firebreak_plot(env, placed_indices, raster_for_bg, out_path=None):
    """
    raster_for_bg: 2D array you want to show under the breaks (e.g. fireline intensity).
                   If None, we’ll use the first channel of obs.
    placed_indices: list[(y,x,order)]
    """
    H, W = raster_for_bg.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(raster_for_bg, cmap="inferno")  # change cmap as desired
    fig.colorbar(im, ax=ax, shrink=0.7, label="Fireline intensity")  # label if relevant

    # scatter breaks, colored by order
    order = np.array([o for (_, _, o) in placed_indices])
    ys = np.array([y for (y, _, _) in placed_indices])
    xs = np.array([x for (_, x, _) in placed_indices])

    sc = ax.scatter(xs, ys, c=order, s=18, cmap="cool", edgecolors="k", linewidths=0.2)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, label="Placement order")
    ax.set_title("Fuel breaks over Fireline Intensity")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def render_frame(env, step_mask, step_no):
    """
    Return an RGB frame (numpy array) showing the background + this step’s breaks.
    Useful for GIFs. Uses matplotlib but returns the pixel buffer.
    """
    # Decide what to draw in background. Example: env.fire_intensity or channel 0
    # TODO: adapt to your env naming:
    bg = getattr(env, "fire_intensity", None)
    bg = env.firelines[0]
    if bg is None:
        # fallback first obs chan
        bg = env.obs[:, :, 0] if env.obs.ndim == 3 else env.obs[0]

    cmap = plt.get_cmap("hot")
    cmap.set_bad("purple")
    fig, ax = plt.subplots()
    ax.imshow(bg, cmap=cmap)
    ys, xs = np.where(step_mask)
    ax.scatter(xs, ys, s=20, edgecolors="k", facecolors="none")
    ax.set_title(f"Step {step_no}")
    ax.set_axis_off()
    fig.canvas.draw()
    # convert to image array
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def write_gif(frames, out_path, fps=2):
    try:
        import imageio
    except ImportError:
        print("imageio not installed. pip install imageio to save GIFs.")
        return
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved GIF to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize trained DQN fuel breaks.")
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/qnet_final.pt",
        help="Path to model checkpoint .pt",
    )
    p.add_argument(
        "--outdir", type=str, default="viz_out", help="Where to save figures/gifs"
    )
    p.add_argument("--gif", action="store_true", help="Save gif of steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument(
        "--kstep", type=int, default=5, help="break_step if you want override"
    )
    p.add_argument("--sims", type=int, default=10, help="num_simulations")
    p.add_argument(
        "--bg_raster_key",
        type=str,
        default="fireline_intensity",
        help="key in raster_dict to use as background; 'none' to use obs channel 0",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raster_dict = load_all_rasters("cropped_raster", 1)
    env = FuelBreakEnv(
        raster_dict,
        break_budget=args.budget,
        break_step=args.kstep,
        num_simulations=args.sims,
    )

    C, H, W = env.observation_space.shape
    model = QNet(H, W).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()

    result = run_episode(env, model, out_dir=args.outdir, make_gif=args.gif)

    # Background raster
    if args.bg_raster_key.lower() != "none" and args.bg_raster_key in raster_dict:
        bg = raster_dict[args.bg_raster_key]
    else:
        # fallback: first channel of current obs
        bg = env.obs[:, :, 0] if env.obs.ndim == 3 else env.obs[0]

    static_path = os.path.join(args.outdir, "fuelbreaks_overlay.png")
    render_static_firebreak_plot(env, result["placed_indices"], bg, static_path)

    if args.gif and result["frames"]:
        gif_path = os.path.join(args.outdir, "episode.gif")
        write_gif(result["frames"], gif_path, fps=2)

    # quick printout
    print(f"Total reward: {sum(result['rewards']):.3f}")
    print(f"Steps taken: {len(result['rewards'])}")
    print("Done.")


if __name__ == "__main__":
    main()
