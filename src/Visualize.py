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
from Model import QNet, EnhancedQNet, DuelingQNet
from src.utils.loadingUtils import load_all_rasters, RasterManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_from_checkpoint(ckpt_path, H, W, model_type="auto"):
    """
    Load model from checkpoint with automatic model type detection.
    """
    print(f"Loading model from {ckpt_path}...")

    # Load checkpoint
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                print(
                    f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
                )
                if "best_reward" in checkpoint:
                    print(f"Best reward: {checkpoint['best_reward']:.3f}")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try to determine model type from state dict if auto
    if model_type == "auto":
        if any("attention" in key for key in state_dict.keys()):
            model_type = "enhanced"
        elif any("value_head" in key for key in state_dict.keys()):
            model_type = "dueling"
        else:
            model_type = "basic"
        print(f"Auto-detected model type: {model_type}")

    # Create appropriate model
    if model_type == "enhanced":
        model = EnhancedQNet(
            H, W, use_attention=True, use_residual=True, use_multiscale=True
        )
        print("Using Enhanced QNet")
    elif model_type == "enhanced_memory":
        model = EnhancedQNet(
            H, W, use_attention=False, use_residual=True, use_multiscale=False
        )
        print("Using Memory-Efficient Enhanced QNet")
    elif model_type == "dueling":
        model = DuelingQNet(H, W)
        print("Using Dueling QNet")
    else:
        model = QNet(H, W)
        print("Using Basic QNet")

    model = model.to(DEVICE)

    # Try loading state dict with error handling
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying non-strict loading...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


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
    """Run a complete episode and collect visualization data."""
    try:
        obs, _ = env.reset()
    except Exception as e:
        print(f"Environment reset failed: {e}")
        return None

    H, W = obs.shape[-2:]
    k = env.break_step

    # Storage for visualization
    placed_indices = []
    rewards = []
    q_maps = []
    frames = []
    burned_areas = []

    done = False
    step = 0
    total_reward = 0

    print("Running episode...")
    while not done and step < 100:  # Safety limit
        try:
            invalid = env._break_mask  # bool (H,W)
            action_vec, q = choose_action_topk_greedy(model, obs, k, invalid)
            print(obs.shape)
            next_obs, reward, done, _, info = env.step(action_vec)

            total_reward += reward
            rewards.append(reward)
            burned_areas.append(info.get("burned", 0))

            # Track placed fuel breaks
            mask = action_vec.reshape(H, W).astype(bool)
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                placed_indices.append((y, x, step))

            q_maps.append(q.reshape(H, W))

            if make_gif and out_dir is not None:
                try:
                    frames.append(render_frame(env, mask, step, obs))
                except Exception as e:
                    print(f"Frame rendering failed: {e}")

            obs = next_obs
            step += 1

            if step % 5 == 0:
                print(f"Step {step}, Reward: {reward:.3f}, Total: {total_reward:.3f}")

        except Exception as e:
            print(f"Step {step} failed: {e}")
            break

    print(f"Episode completed: {step} steps, Total reward: {total_reward:.3f}")

    return {
        "placed_indices": placed_indices,
        "rewards": rewards,
        "q_maps": q_maps,
        "frames": frames,
        "burned_areas": burned_areas,
        "total_reward": total_reward,
        "steps": step,
    }


def render_static_firebreak_plot(env, placed_indices, raster_for_bg, out_path=None):
    """Render static plot showing fuel break placement over background raster."""
    if len(placed_indices) == 0:
        print("No fuel breaks to visualize")
        return

    H, W = raster_for_bg.shape
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Background raster with fuel breaks
    ax1 = axes[0]
    im1 = ax1.imshow(raster_for_bg, cmap="inferno", aspect="equal")
    fig.colorbar(im1, ax=ax1, shrink=0.7, label="Background intensity")

    # Plot fuel breaks colored by placement order
    if placed_indices:
        order = np.array([o for (_, _, o) in placed_indices])
        ys = np.array([y for (y, _, _) in placed_indices])
        xs = np.array([x for (_, x, _) in placed_indices])

        sc = ax1.scatter(
            xs,
            ys,
            c=order,
            s=30,
            cmap="cool",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.8,
        )
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.7, label="Placement order")

    ax1.set_title("Fuel Breaks Placement")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")

    # Right plot: Final fuel break mask
    ax2 = axes[1]
    fuel_break_mask = np.zeros((H, W), dtype=bool)
    if placed_indices:
        for y, x, _ in placed_indices:
            fuel_break_mask[y, x] = True

    # Show original raster
    ax2.imshow(raster_for_bg, cmap="inferno", alpha=0.6, aspect="equal")
    # Overlay fuel breaks
    ax2.imshow(fuel_break_mask, cmap="Greens", alpha=0.8, aspect="equal")
    ax2.set_title("Final Fuel Break Network")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def render_frame(env, step_mask, step_no, obs):
    """Render a single frame for GIF creation."""
    # Use first channel of observation as background
    bg = obs[1] if obs.ndim == 3 else obs[0, :, :]
    print(bg.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg, cmap="Greens", aspect="equal")

    # Show fuel breaks placed in this step
    if np.any(step_mask):
        ys, xs = np.where(step_mask)
        ax.scatter(
            xs,
            ys,
            s=50,
            c="cyan",
            marker="s",
            edgecolors="black",
            linewidths=1,
            alpha=0.9,
        )

    ax.set_title(f"Step {step_no}", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    # Convert to image array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def render_q_value_heatmap(q_maps, placed_indices, out_path=None):
    """Render Q-value evolution over time."""
    if not q_maps:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Show Q-values at different steps
    steps_to_show = [
        0,
        len(q_maps) // 4,
        len(q_maps) // 2,
        3 * len(q_maps) // 4,
        len(q_maps) - 1,
    ]
    steps_to_show = [s for s in steps_to_show if s < len(q_maps)]

    for i, step_idx in enumerate(steps_to_show[:6]):
        if i >= len(axes):
            break

        ax = axes[i]
        q_map = q_maps[step_idx]

        im = ax.imshow(q_map, cmap="RdYlBu_r", aspect="equal")

        # Show fuel breaks placed up to this step
        placed_up_to_step = [(y, x) for y, x, s in placed_indices if s <= step_idx]
        if placed_up_to_step:
            ys, xs = zip(*placed_up_to_step)
            ax.scatter(xs, ys, c="black", s=20, marker="x", alpha=0.8)

        ax.set_title(f"Q-values at Step {step_idx}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(len(steps_to_show), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved Q-value heatmap to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def write_gif(frames, out_path, fps=2):
    """Write frames to GIF file."""
    try:
        import imageio

        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Saved GIF to {out_path}")
    except ImportError:
        print("imageio not installed. pip install imageio to save GIFs.")
    except Exception as e:
        print(f"Failed to save GIF: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize trained DQN fuel breaks.")
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/qnet_best.pt",
        help="Path to model checkpoint .pt",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "basic", "enhanced", "enhanced_memory", "dueling"],
        help="Model architecture type",
    )
    p.add_argument(
        "--outdir", type=str, default="viz_out", help="Where to save figures/gifs"
    )
    p.add_argument("--gif", action="store_true", help="Save gif of steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--kstep", type=int, default=10, help="break_step")
    p.add_argument(
        "--sims", type=int, default=5, help="num_simulations (reduced for viz)"
    )
    p.add_argument(
        "--raster_idx", type=int, default=1, help="Which raster to use (1-500)"
    )
    p.add_argument(
        "--bg_raster_key",
        type=str,
        default="fireline_north",
        help="key in raster_dict to use as background",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("QAgent Visualization Tool")
    print("=" * 40)

    # Load raster data
    print("Loading raster data...")
    try:
        raster_manager = RasterManager("cropped_raster", 500)
        all_rasters = raster_manager.load_all_rasters()

        if len(all_rasters) == 0:
            print("No rasters found, creating dummy raster...")
            raster_dict = {
                "slp": np.random.rand(50, 50).astype(np.float32),
                "asp": np.random.rand(50, 50).astype(np.float32),
                "dem": np.random.rand(50, 50).astype(np.float32),
                "cc": np.random.rand(50, 50).astype(np.float32),
                "cbd": np.random.rand(50, 50).astype(np.float32),
                "cbh": np.random.rand(50, 50).astype(np.float32),
                "ch": np.random.rand(50, 50).astype(np.float32),
                "fbfm": np.random.randint(1, 14, (50, 50)).astype(np.float32),
                "fireline_north": np.random.rand(50, 50).astype(np.float32),
                "fireline_east": np.random.rand(50, 50).astype(np.float32),
                "fireline_south": np.random.rand(50, 50).astype(np.float32),
                "fireline_west": np.random.rand(50, 50).astype(np.float32),
            }
        else:
            raster_idx = min(args.raster_idx - 1, len(all_rasters) - 1)
            raster_dict = all_rasters[raster_idx]
            print(f"Using raster {raster_idx + 1}/{len(all_rasters)}")

    except Exception as e:
        print(f"Error loading rasters: {e}")
        print("Using dummy raster data...")
        raster_dict = {
            "slp": np.random.rand(50, 50).astype(np.float32),
            "asp": np.random.rand(50, 50).astype(np.float32),
            "dem": np.random.rand(50, 50).astype(np.float32),
            "cc": np.random.rand(50, 50).astype(np.float32),
            "cbd": np.random.rand(50, 50).astype(np.float32),
            "cbh": np.random.rand(50, 50).astype(np.float32),
            "ch": np.random.rand(50, 50).astype(np.float32),
            "fbfm": np.random.randint(1, 14, (50, 50)).astype(np.float32),
            "fireline_north": np.random.rand(50, 50).astype(np.float32),
            "fireline_east": np.random.rand(50, 50).astype(np.float32),
            "fireline_south": np.random.rand(50, 50).astype(np.float32),
            "fireline_west": np.random.rand(50, 50).astype(np.float32),
        }

    # Create environment
    print("Creating environment...")
    try:
        env = FuelBreakEnv(
            raster_dict,
            break_budget=args.budget,
            break_step=args.kstep,
            num_simulations=args.sims,
            seed=args.seed,
        )
        obs, _ = env.reset()
        H, W = obs.shape[-2:]
        print(f"Environment created: {H}x{W} landscape")
    except Exception as e:
        print(f"Environment creation failed: {e}")
        return

    # Load model
    try:
        model = load_model_from_checkpoint(args.ckpt, H, W, args.model_type)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run episode
    result = run_episode(env, model, out_dir=args.outdir, make_gif=args.gif)

    if result is None:
        print("Episode failed to run")
        return

    # Generate visualizations
    print("Generating visualizations...")

    # Background raster for visualization
    if args.bg_raster_key in raster_dict:
        bg = raster_dict[args.bg_raster_key]
    else:
        print(f"Background key '{args.bg_raster_key}' not found, using slope")
        bg = raster_dict.get("slp", obs[0])

    # Static fuel break plot
    static_path = os.path.join(args.outdir, "fuelbreaks_overlay.png")
    render_static_firebreak_plot(env, result["placed_indices"], bg, static_path)

    # Q-value heatmap
    if result["q_maps"]:
        qvalue_path = os.path.join(args.outdir, "qvalue_evolution.png")
        render_q_value_heatmap(result["q_maps"], result["placed_indices"], qvalue_path)

    # GIF creation
    if args.gif and result["frames"]:
        gif_path = os.path.join(args.outdir, "episode.gif")
        write_gif(result["frames"], gif_path, fps=2)

    # Summary plot
    if result["rewards"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(result["rewards"], "b-", linewidth=2)
        ax1.set_title("Reward per Step")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)

        if result["burned_areas"]:
            ax2.plot(result["burned_areas"], "r-", linewidth=2)
            ax2.set_title("Burned Area per Step")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Burned Area")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        summary_path = os.path.join(args.outdir, "training_summary.png")
        fig.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved summary plot to {summary_path}")

    # Print results
    print("\n" + "=" * 40)
    print("EPISODE RESULTS:")
    print(f"Total reward: {result['total_reward']:.3f}")
    print(f"Steps taken: {result['steps']}")
    print(f"Fuel breaks placed: {len(result['placed_indices'])}")
    print(
        f"Average reward per step: {result['total_reward'] / max(1, result['steps']):.3f}"
    )
    if result["burned_areas"]:
        print(f"Final burned area: {result['burned_areas'][-1]:.1f}")
    print(f"Visualizations saved to: {args.outdir}")
    print("Done!")


if __name__ == "__main__":
    main()
